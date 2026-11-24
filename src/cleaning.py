"""Cleaning, checking and summarising the close-approach data.

Rules we follow
---------------
* The raw file is never changed. Cleaning returns a NEW table.
* Missing stays missing. A missing diameter means "unknown", never 0.
* Only EXACT duplicate rows are deleted. Strange values are FLAGGED in columns that
  start with `qc_` (quality check), not deleted, because unusual can still be real.
* The only value we change: a diameter <= 0 (or a negative diameter error) is physically
  impossible, so it becomes missing. The count is reported.
* Every step counts what it did in a `report` dictionary (like a C++ map<string, int>).
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Columns that hold numbers (they arrive from the API as text).
NUMERIC_COLUMNS = ["jd", "dist", "dist_min", "dist_max", "v_rel", "v_inf", "h", "diameter", "diameter_sigma"]
# Columns that must exist, otherwise cleaning cannot work.
REQUIRED_COLUMNS = ["des", "jd", "cd", "dist", "v_rel"]

# Absolute magnitude H outside this range is suspicious (only flagged, never deleted).
H_MIN = -2.0
H_MAX = 35.0
# The two date columns (jd and cd) may differ by at most this many minutes.
JD_CD_TOLERANCE_MINUTES = 2.0
# Julian date of 1970-01-01 00:00 (the "Unix epoch"), used to convert jd to a date.
UNIX_EPOCH_JD = 2440587.5

# Month names -> numbers. Done by hand so it works on any computer language setting.
MONTH_NUMBERS = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
                 "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}

# Pattern of the calendar date, e.g. "2028-Jun-26 05:23" -> 5 pieces: year, month, day, hour, minute.
CD_PATTERN = r"^\s*(\d{4})-([A-Za-z]{3})-(\d{2}) (\d{2}):(\d{2})\s*$"
# Pattern of the time uncertainty, e.g. "13:02", "2_09:08" (2 days 9 h 8 min) or "< 00:01".
T_SIGMA_PATTERN = re.compile(r"^\s*(?P<prefix>[<>~]?)\s*(?:(?P<d>\d+)_)?(?P<h>\d+):(?P<m>\d{2})\s*$")


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #
def parse_cd(series):
    """Convert calendar dates like '2028-Jun-26 05:23' into real datetime values.

    Text that does not match the pattern becomes NaT ("Not a Time" = missing date).
    """
    # Split each date string into 5 text pieces (one column per piece).
    parts = series.astype("string").str.extract(CD_PATTERN)

    # Turn the month name into a number ("jun" -> "Jun" -> 6).
    month = parts[1].str.title().map(MONTH_NUMBERS)

    # Build a small table with year, month, day, hour, minute as numbers.
    date_parts = pd.DataFrame({
        "year": pd.to_numeric(parts[0], errors="coerce").astype("float64"),
        "month": pd.to_numeric(month, errors="coerce").astype("float64"),
        "day": pd.to_numeric(parts[2], errors="coerce").astype("float64"),
        "hour": pd.to_numeric(parts[3], errors="coerce").astype("float64"),
        "minute": pd.to_numeric(parts[4], errors="coerce").astype("float64"),
    })

    # pandas combines the 5 columns into one datetime column. Impossible dates -> NaT.
    return pd.to_datetime(date_parts, errors="coerce")


def parse_t_sigma(value):
    """Convert one time-uncertainty text to (minutes, is_bound).

    "13:02" -> (782.0, False);  "2_09:08" -> (3428.0, False);  "< 00:01" -> (1.0, True)
    is_bound is True when the text starts with < or >, meaning the number is only a limit.
    """
    # Missing value -> missing result.
    if pd.isna(value):
        return (np.nan, False)

    # Text that does not match the pattern -> missing result.
    match = T_SIGMA_PATTERN.match(str(value))
    if match is None:
        return (np.nan, False)

    # days * 1440 + hours * 60 + minutes  (1440 minutes in a day)
    days = int(match.group("d") or 0)
    minutes = days * 1440 + int(match.group("h")) * 60 + int(match.group("m"))
    is_bound = match.group("prefix") in ["<", ">"]
    return (float(minutes), is_bound)


# --------------------------------------------------------------------------- #
# Cleaning steps (each small function does ONE step and writes its counts in `report`)
# --------------------------------------------------------------------------- #
def strip_text(df, source_columns, report):
    """Step 1: remove spaces around text and turn empty text into missing."""
    for col in source_columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype("string").str.strip().replace("", pd.NA)

    # Count missing values per column BEFORE number conversion.
    missing = {}
    for col in source_columns:
        missing[col] = int(df[col].isna().sum())
    report["missing_before_conversion"] = missing
    return df


def remove_exact_duplicates(df, source_columns, report):
    """Step 2: delete rows that are identical to an earlier row in every source column."""
    is_duplicate = df.duplicated(subset=source_columns, keep="first")
    report["exact_duplicate_rows_removed"] = int(is_duplicate.sum())
    return df.loc[~is_duplicate].copy()


def convert_numbers(df, report):
    """Step 3: convert text columns to numbers. Text that is not a number becomes NaN."""
    failures = {}
    nonfinite = {}
    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            continue
        was_missing = df[col].isna()
        # errors="coerce" means: if conversion fails, put NaN instead of crashing.
        converted = pd.to_numeric(df[col], errors="coerce").astype("float64")
        # A failure = the value was present before but is NaN now.
        failures[col] = int((~was_missing & converted.isna()).sum())
        # Infinity ("inf") is not a real measurement -> set it to missing too.
        is_infinite = np.isinf(converted)
        nonfinite[col] = int(is_infinite.sum())
        converted[is_infinite] = np.nan
        df[col] = converted
    report["numeric_conversion_failures"] = failures
    report["nonfinite_values_set_missing"] = nonfinite
    return df


def add_dates(df, report):
    """Step 4: parse the calendar date and check it against the Julian date."""
    df["approach_time"] = parse_cd(df["cd"])
    report["date_parse_failures"] = int(df["approach_time"].isna().sum())

    # Convert Julian date to a datetime: days since 1970-01-01 = jd - 2440587.5
    jd_time = pd.to_datetime(df["jd"] - UNIX_EPOCH_JD, unit="D", errors="coerce")

    # Difference between the two dates in minutes. More than 2 minutes -> flag.
    diff_minutes = (df["approach_time"] - jd_time).dt.total_seconds().abs() / 60.0
    df["qc_jd_cd_mismatch"] = (diff_minutes > JD_CD_TOLERANCE_MINUTES).fillna(False)
    report["jd_cd_mismatch_rows"] = int(df["qc_jd_cd_mismatch"].sum())
    report["jd_cd_tolerance_minutes"] = JD_CD_TOLERANCE_MINUTES
    return df


def add_t_sigma(df, report):
    """Step 5: convert the time-uncertainty text (t_sigma_f) into minutes."""
    if "t_sigma_f" not in df.columns:
        return df

    # Plain loop over every value (like a C++ for-each over a vector).
    minutes_list = []
    bound_list = []
    for value in df["t_sigma_f"]:
        minutes, is_bound = parse_t_sigma(value)
        minutes_list.append(minutes)
        bound_list.append(is_bound)

    df["t_sigma_minutes"] = pd.Series(minutes_list, index=df.index, dtype="float64")
    df["t_sigma_is_bound"] = pd.Series(bound_list, index=df.index, dtype=bool)
    report["t_sigma_unparsed"] = int((df["t_sigma_f"].notna() & df["t_sigma_minutes"].isna()).sum())
    return df


def add_quality_flags(df, report):
    """Step 6: add True/False `qc_*` columns for strange values, and fix impossible diameters."""
    # Distance must be positive.
    df["qc_nonpositive_dist"] = (df["dist"] <= 0).fillna(False)

    # dist_min <= dist <= dist_max should hold (tiny tolerance for rounding).
    tol = 1e-12
    if "dist_min" in df.columns and "dist_max" in df.columns:
        too_low = df["dist_min"] > df["dist"] * (1 + tol)
        too_high = df["dist_max"] < df["dist"] * (1 - tol)
        df["qc_dist_range_inconsistent"] = (too_low | too_high).fillna(False)

    # Speeds must not be negative.
    negative = df["v_rel"] < 0
    if "v_inf" in df.columns:
        negative = negative | (df["v_inf"] < 0)
    df["qc_negative_velocity"] = negative.fillna(False)

    # Absolute magnitude H outside [-2, 35] is suspicious.
    if "h" in df.columns:
        df["qc_h_implausible"] = ((df["h"] < H_MIN) | (df["h"] > H_MAX)).fillna(False)

    # Diameter <= 0 is impossible -> set to missing (NOT to zero!).
    invalid = {}
    if "diameter" in df.columns:
        report["diameter_missing_before_validity_check"] = int(df["diameter"].isna().sum())
        bad_diameter = df["diameter"] <= 0
        invalid["nonpositive_diameter_set_missing"] = int(bad_diameter.sum())
        df.loc[bad_diameter, "diameter"] = np.nan

    # A negative diameter uncertainty is impossible -> set to missing.
    if "diameter_sigma" in df.columns:
        bad_sigma = df["diameter_sigma"] < 0
        invalid["negative_diameter_sigma_set_missing"] = int(bad_sigma.sum())
        df.loc[bad_sigma, "diameter_sigma"] = np.nan

    # Count how many rows each qc_ flag marked.
    for col in df.columns:
        if col.startswith("qc_"):
            invalid[col] = int(df[col].sum())
    report["validity_checks"] = invalid
    return df


def add_body_column(df, report):
    """Step 7: make sure a `body` column exists.

    The API only sends `body` when we ask for body=ALL. Otherwise every row is an
    Earth approach (the API default), so we fill in "Earth".
    """
    report["body_column_in_raw"] = "body" in df.columns
    if "body" not in df.columns:
        df["body"] = "Earth"
        report["body_note"] = ("Raw data has no 'body' column (API omits it unless body=ALL); "
                               "the API default body is Earth, so 'Earth' was assigned.")
    return df


def flag_repeated_event_keys(df, report):
    """Step 8: flag (not delete) rows that share the same object + time + body.

    Such rows may be two orbit solutions for one event. Choosing one is a scientific
    decision, so we keep both and only mark them.
    """
    key = []
    for col in ["des", "jd", "body"]:
        if col in df.columns:
            key.append(col)

    # keep=False marks EVERY row of a repeated group; keep="first" marks only the extra copies.
    repeated = df.duplicated(subset=key, keep=False)
    df["dup_event_key"] = repeated
    report["event_key_columns"] = key
    report["rows_with_repeated_event_key"] = int(repeated.sum())
    report["repeated_event_key_extra_rows"] = int(df.duplicated(subset=key, keep="first").sum())
    return df


# --------------------------------------------------------------------------- #
# Main cleaning function: runs all steps in order
# --------------------------------------------------------------------------- #
def clean_close_approaches(raw):
    """Clean the raw table and return (clean_df, report)."""
    # Stop early if an essential column is missing.
    for col in REQUIRED_COLUMNS:
        if col not in raw.columns:
            raise ValueError(f"Missing required column: {col}")

    # Work on a copy; remember each row's position in the raw file (to trace it back later).
    report = {"rows_raw": len(raw)}
    source_columns = list(raw.columns)
    df = raw.copy()
    df.insert(0, "source_row_id", np.arange(len(df), dtype="int64"))

    # Run the steps one after another.
    df = strip_text(df, source_columns, report)
    df = remove_exact_duplicates(df, source_columns, report)
    df = convert_numbers(df, report)
    df = add_dates(df, report)
    df = add_t_sigma(df, report)
    df = add_quality_flags(df, report)
    df = add_body_column(df, report)
    df = flag_repeated_event_keys(df, report)

    # Tidy up: text columns as text type, rows sorted by time then object name.
    for col in ["des", "orbit_id", "t_sigma_f", "body"]:
        if col in df.columns:
            df[col] = df[col].astype("string")
    # mergesort is a "stable" sort: rows with equal keys keep their original order.
    df = df.sort_values(["approach_time", "des"], kind="mergesort").reset_index(drop=True)

    # Final counts for the report.
    report["rows_clean"] = len(df)
    report["unique_designations"] = int(df["des"].nunique())
    report["approach_time_min"] = str(df["approach_time"].min())
    report["approach_time_max"] = str(df["approach_time"].max())
    missing_after = {}
    for col in df.columns:
        missing_after[col] = int(df[col].isna().sum())
    report["missing_after_cleaning"] = missing_after
    if "diameter" in df.columns:
        report["diameter_missing_after_cleaning"] = int(df["diameter"].isna().sum())
    return df, report


# --------------------------------------------------------------------------- #
# Summary helpers used by the notebooks
# --------------------------------------------------------------------------- #
def missing_summary(df):
    """Table: for each column its type, missing count, missing % and number of distinct values."""
    counts = df.isna().sum()
    n_rows = max(len(df), 1)            # avoid division by zero on an empty table
    out = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing_count": counts,
        "missing_pct": (counts / n_rows * 100).round(3),
        "n_unique": df.nunique(dropna=True),
    })
    # Most-missing columns first.
    return out.sort_values(["missing_pct", "missing_count"], ascending=False)


def numeric_summary(df, columns):
    """Descriptive statistics for each numeric column (count, mean, std, percentiles, skew, kurtosis)."""
    rows = {}
    for col in columns:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        q = s.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        rows[col] = {
            "count": int(s.size),
            "mean": s.mean(),
            "std": s.std(),
            "min": s.min(),
            "p01": q[0.01], "p05": q[0.05], "q1": q[0.25], "median": q[0.5],
            "q3": q[0.75], "p95": q[0.95], "p99": q[0.99],
            "max": s.max(),
            "skew": s.skew(),          # > 0 means a long tail to the right
            "kurtosis": s.kurt(),      # > 0 means heavier tails than a normal distribution
        }
    return pd.DataFrame(rows).T


def iqr_bounds(series, k=1.5):
    """Tukey fences: (Q1 - k*IQR, Q3 + k*IQR). Values outside are called outliers."""
    s = series.dropna()
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    return float(q1 - k * iqr), float(q3 + k * iqr)


def iqr_outlier_summary(df, columns, k=1.5, log10_columns=None):
    """Count values outside the Tukey fences for each column (nothing is deleted).

    Columns listed in `log10_columns` are tested on log10 scale, because their values
    span many orders of magnitude (e.g. distances from 0.01 to 100 LD).
    """
    if log10_columns is None:
        log10_columns = set()

    rows = []
    for col in columns:
        s = df[col].dropna()
        use_log = col in log10_columns
        if use_log:
            s = s[s > 0]               # log10 only works for positive numbers
            values = np.log10(s)
        else:
            values = s

        lo, hi = iqr_bounds(values, k)
        outside = (values < lo) | (values > hi)
        rows.append({
            "column": col,
            "scale": "log10" if use_log else "linear",
            "n": int(values.size),
            "lower_fence": lo,
            "upper_fence": hi,
            "n_low": int((values < lo).sum()),
            "n_high": int((values > hi).sum()),
            "pct_outside": round(float(outside.mean() * 100), 3),
        })
    return pd.DataFrame(rows).set_index("column")


def date_span_years(times):
    """Length of the date range in years.

    We convert to plain Python datetimes first, because pandas timestamps overflow
    for differences longer than about 292 years (our window 1900-2200 is 300 years).
    """
    values = times.dropna()
    if len(values) == 0:
        return float("nan")
    first = values.min().to_pydatetime()
    last = values.max().to_pydatetime()
    return (last - first).days / 365.25


# --------------------------------------------------------------------------- #
# Saving and loading the cleaned table
# --------------------------------------------------------------------------- #
def save_clean(df, report, processed_dir, name="close_approaches_clean"):
    """Save the cleaned table (.csv.gz) and the cleaning report (.json)."""
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    data_path = processed_dir / f"{name}.csv.gz"
    report_path = processed_dir / f"{name}_report.json"
    df.to_csv(data_path, index=False, compression="gzip")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    return data_path, report_path


def load_processed(path):
    """Load a cleaned CSV and restore the column types that CSV files cannot store."""
    text_types = {"des": "string", "orbit_id": "string", "body": "string",
                  "t_sigma_f": "string", "fullname": "string"}
    df = pd.read_csv(path, parse_dates=["approach_time"], dtype=text_types)

    # True/False columns may come back as text "True"/"False": turn them into booleans.
    for col in df.columns:
        is_flag = col.startswith("qc_") or col in ["dup_event_key", "t_sigma_is_bound"]
        if is_flag and df[col].dtype == object:
            df[col] = df[col].astype(str).str.lower().eq("true")
    return df
