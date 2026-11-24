"""Download close-approach data from the NASA/JPL CAD API.

API address : https://ssd-api.jpl.nasa.gov/cad.api
Docs        : https://ssd-api.jpl.nasa.gov/doc/cad.html

How the API works (simple version)
----------------------------------
We send a web request (HTTP GET) with some settings ("query parameters"), for example
body=ALL and dist-max=0.1. The API answers with JSON (like a C++ map of key -> value):

    {"signature": {...}, "count": 3, "total": 104497,
     "fields": ["des", "orbit_id", "jd", ...],
     "data":   [["2001 AB", "12", "2451545.5", ...], ...]}

"fields" are the column names and "data" is the list of rows.
The API gives at most `limit` rows per request, so we ask page by page using
`limit` (page size) and `limit-from` (row number to start at; it starts at 1, not 0).

Important: one row = one close-approach EVENT (one object passing one planet/moon on
one date). The same object can appear in many rows.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

# Address of the official API and its documentation.
CAD_URL = "https://ssd-api.jpl.nasa.gov/cad.api"
DOC_URL = "https://ssd-api.jpl.nasa.gov/doc/cad.html"

# HTTP status codes that mean "server busy / temporary problem": worth trying again.
RETRY_STATUS_CODES = [429, 500, 502, 503, 504]


# --------------------------------------------------------------------------- #
# 1. The list of queries we may use
# --------------------------------------------------------------------------- #
def build_query_profiles():
    """Return an ordered dict {profile_name: query_parameters}, from narrow to broad.

    The notebook tests them in this order and uses the FIRST one that gives at least
    100,000 events. So the dataset is never wider than needed.
    """
    # Each line is: (profile name, target body, only NEOs?, maximum distance in au)
    # body="ALL"   -> approaches to every planet/moon, not just Earth
    # neo="true"   -> only Near-Earth Objects; "false" -> filter not applied (all objects)
    # dist-max     -> keep approaches closer than this distance (au = Sun-Earth distance)
    settings = [
        ("earth_neo_0.05au", "Earth", "true", "0.05"),
        ("all_bodies_neo_0.05au", "ALL", "true", "0.05"),
        ("all_bodies_neo_0.1au", "ALL", "true", "0.1"),
        ("all_bodies_neo_0.2au", "ALL", "true", "0.2"),
        ("all_bodies_neo_0.5au", "ALL", "true", "0.5"),
        ("all_bodies_neo_1au", "ALL", "true", "1"),
        ("all_bodies_all_objects_0.2au", "ALL", "false", "0.2"),
        ("all_bodies_all_objects_0.5au", "ALL", "false", "0.5"),
        ("all_bodies_all_objects_1au", "ALL", "false", "1"),
    ]

    # Build the full parameter dictionary for each profile.
    # The settings shared by every profile: years 1900-2200, sorted by date,
    # and ask the API to also send the diameter and full-name columns.
    profiles = {}
    for name, body, neo, dist_max in settings:
        params = {
            "date-min": "1900-01-01",
            "date-max": "2200-01-01",
            "sort": "date",
            "diameter": "true",
            "fullname": "true",
            "body": body,
            "neo": neo,
            "dist-max": dist_max,
        }
        profiles[name] = params
    return profiles


# --------------------------------------------------------------------------- #
# 2. Sending one request
# --------------------------------------------------------------------------- #
def request_json(params, max_retries=3, timeout_seconds=60):
    """Send one GET request to the API and return the answer as a Python dict.

    If the server is busy or the network fails, wait and try again
    (wait 2 s, then 4 s, then 8 s). Other errors (for example a wrong parameter)
    stop immediately, because trying again would not help.
    """
    for attempt in range(max_retries + 1):
        # Try the request. Network problems raise an exception, which we catch.
        try:
            response = requests.get(CAD_URL, params=params, timeout=timeout_seconds)
        except (requests.Timeout, requests.ConnectionError) as error:
            print(f"Network problem (attempt {attempt + 1}): {error}")
        else:
            # 200 = OK: convert the JSON text into a Python dict and return it.
            if response.status_code == 200:
                return response.json()
            # Not a temporary problem: stop with a clear error message.
            if response.status_code not in RETRY_STATUS_CODES:
                raise RuntimeError(f"HTTP {response.status_code} from CAD API: {response.text[:500]}")
            print(f"Server busy, HTTP {response.status_code} (attempt {attempt + 1})")

        # Wait before the next attempt (exponential back-off: 2, 4, 8 seconds).
        if attempt < max_retries:
            time.sleep(2 * (2 ** attempt))

    raise RuntimeError(f"CAD request failed after {max_retries + 1} attempts")


# --------------------------------------------------------------------------- #
# 3. Counting how many rows a query would return (without downloading them)
# --------------------------------------------------------------------------- #
def get_total(params):
    """Return how many events match the query (uses the API option total-only=true)."""
    # Ask only for the count, not the rows.
    query = dict(params)
    query["total-only"] = "true"
    payload = request_json(query)
    if "total" in payload:
        return int(payload["total"])

    # Fallback described in the API docs: a normal request with limit=1 also reports "total".
    query = dict(params)
    query["limit"] = 1
    payload = request_json(query)
    if "total" in payload:
        return int(payload["total"])

    # No "total" at all means nothing matched.
    return 0


# --------------------------------------------------------------------------- #
# 4. Downloading all rows, page by page
# --------------------------------------------------------------------------- #
def fetch_close_approaches(params, batch_size=10_000, pause_seconds=1.0, max_rows=2_000_000):
    """Download ALL rows that match `params` and return (raw_dataframe, metadata).

    We download everything (not just the first pages) because rows are sorted by
    date: stopping early would keep only the oldest years.
    """
    # Remember when the download started (UTC time). Used later to mark past/future events.
    started = datetime.now(timezone.utc)

    # First ask how many rows to expect, and stop if the number is 0 or too large.
    total = get_total(params)
    if total == 0:
        raise RuntimeError("Query matches zero records; widen the filters.")
    if total > max_rows:
        raise RuntimeError(f"Query matches {total:,} records, more than max_rows={max_rows:,}.")

    # Download page by page until we have `total` rows.
    # Like a C++ loop: while (rows.size() < total) { get next page; append; }
    all_rows = []
    fields = None
    start_row = 1                      # the API numbers rows from 1
    while len(all_rows) < total:
        query = dict(params)
        query["limit"] = batch_size
        query["limit-from"] = start_row
        payload = request_json(query)
        page_rows = payload.get("data", [])

        # An empty page before reaching the total: stop.
        if len(page_rows) == 0:
            if len(all_rows) < total and (total - len(all_rows)) > max(100, int(total * 0.005)):
                print(f"WARNING: empty page at row {start_row} before reaching the total; stopping.")
            break

        # The column names must be the same on every page, otherwise rows would not line up.
        if fields is None:
            fields = payload["fields"]
        elif payload["fields"] != fields:
            raise RuntimeError("Column list changed between pages; refusing to merge.")

        all_rows.extend(page_rows)
        start_row = start_row + len(page_rows)
        print(f"  downloaded {len(all_rows):,} / {total:,}")

        # If fewer rows than batch_size were returned, this was the last page.
        if len(page_rows) < batch_size:
            break

        # Be polite to the server: short pause between requests.
        if len(all_rows) < total:
            time.sleep(pause_seconds)

    if fields is None or len(all_rows) == 0:
        raise RuntimeError("No data pages were retrieved.")

    # Put the rows in a table (DataFrame = a 2-D table, like vector<vector<string>> with column names).
    df = pd.DataFrame(all_rows, columns=fields)

    # Safety check: we expect all rows, but allow a tiny difference due to live database updates.
    if len(df) != total:
        diff = abs(len(df) - total)
        if diff <= max(100, int(total * 0.005)):
            print(f"  Note: downloaded {len(df):,} rows vs API initial estimate of {total:,} "
                  f"({len(df) - total:+d} row difference due to live database updates).")
        else:
            raise RuntimeError(f"Downloaded {len(df):,} rows but the API reported {total:,}. Not saved.")

    # Metadata = facts about this download, saved next to the data file.
    metadata = {
        "retrieved_utc": started.isoformat(timespec="seconds"),
        "endpoint": CAD_URL,
        "documentation": DOC_URL,
        "query_parameters": dict(params),
        "total_reported": total,
        "rows_downloaded": len(df),
        "rows_match_reported_total": len(df) == total,
    }
    counts = count_events(df)
    for key in counts:
        metadata[key] = counts[key]
    return df, metadata


def count_events(df):
    """Count events, unique objects, exact duplicate rows and repeated event keys."""
    # An "event key" identifies one event: same object (des), same time (jd), same body.
    # The body column only exists when the query used body=ALL.
    key = []
    for col in ["des", "jd", "body"]:
        if col in df.columns:
            key.append(col)

    return {
        "event_records": len(df),
        "unique_designations": int(df["des"].nunique()),
        "exact_duplicate_rows": int(df.duplicated(keep="first").sum()),
        "repeated_event_key_rows": int(df.duplicated(subset=key, keep="first").sum()),
        "event_key_columns": key,
    }


# --------------------------------------------------------------------------- #
# 5. Saving and loading the raw file
# --------------------------------------------------------------------------- #
def save_raw(df, metadata, raw_dir):
    """Save raw rows as data/raw/cad_raw_<time>.csv.gz plus a .metadata.json file.

    The file name contains the download time, so an old download is never overwritten.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Turn "2026-11-18T16:38:50+00:00" into "20261118T163850Z" for the file name.
    stamp = metadata["retrieved_utc"].replace("-", "").replace(":", "").replace("+0000", "Z")
    data_path = raw_dir / f"cad_raw_{stamp}.csv.gz"
    meta_path = raw_dir / f"cad_raw_{stamp}.metadata.json"
    if data_path.exists() or meta_path.exists():
        raise FileExistsError(f"Refusing to overwrite raw data: {data_path}")

    # Save the table (compressed CSV) and the metadata (JSON text file).
    df.to_csv(data_path, index=False, compression="gzip")
    metadata["raw_file"] = data_path.name
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, default=str)
    return data_path, meta_path


def latest_raw_file(raw_dir):
    """Return the newest raw file (names contain the time, so sorting by name = by time)."""
    files = sorted(Path(raw_dir).glob("cad_raw_*.csv.gz"))
    if len(files) == 0:
        raise FileNotFoundError(f"No raw files in {raw_dir}. Run notebook 01 with REUSE_LATEST_RAW = False first.")
    return files[-1]


def load_raw(path):
    """Load a raw CSV with every column as text; only empty cells become missing (NaN)."""
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
