"""Feature engineering: new columns built from the cleaned data.

IMPORTANT: the categories made here (proximity, velocity, size, groups A-D) are simple
DESCRIPTIVE groups with round-number thresholds. They are NOT impact-risk ratings and do
NOT predict impacts (that is the job of NASA/JPL's Sentry system).

Constants
---------
AU_KM  = 149,597,870.7 km   1 astronomical unit (IAU 2012, exact)
LD_KM  = 384,400 km         1 lunar distance (Earth-Moon distance, JPL value)
Diameter from brightness H with an ASSUMED albedo p (reflectivity):
        D[km] = 1329 / sqrt(p) * 10^(-H/5)
This is an estimate, not a measurement: the real albedo is usually unknown.
"""

import numpy as np
import pandas as pd

AU_KM = 149_597_870.7
LD_KM = 384_400.0
DIAMETER_H_CONSTANT_KM = 1329.0
ALBEDO_LOW = 0.05             # dark object  -> larger size estimate
ALBEDO_CNEOS_DEFAULT = 0.14   # NASA CNEOS default
ALBEDO_HIGH = 0.25            # bright object -> smaller size estimate

# ---- category rules (round numbers, chosen for readability, not physics) ----
# Proximity bins in lunar distances (Earth only): [0,1), [1,5), [5,20), [20, inf)
PROXIMITY_BINS_LD = [0.0, 1.0, 5.0, 20.0, np.inf]
PROXIMITY_LABELS = ["<1 LD", "1-5 LD", "5-20 LD", ">=20 LD"]
NOT_APPLICABLE_BODY = "N/A (non-Earth body)"

# Velocity bins in km/s: [0,10), [10,20), [20, inf)
VELOCITY_BINS_KMS = [0.0, 10.0, 20.0, np.inf]
VELOCITY_LABELS = ["Slow (<10 km/s)", "Moderate (10-20 km/s)", "Fast (>=20 km/s)"]

# Size proxy from H (larger H = fainter = usually smaller). H <= 22 is roughly >= 140 m.
H_LARGER_MAX = 22.0
H_MEDIUM_MAX = 26.0
SIZE_LABELS = ["Larger (H<=22)", "Medium (22<H<=26)", "Small (H>26)", "H unknown"]

GROUP_LABELS = {
    "A": "A: <5 LD & larger (H<=22)",
    "B": "B: <5 LD & medium/small (H>22)",
    "C": "C: <5 LD & H unknown",
    "D": "D: Earth, >=5 LD",
    "N": NOT_APPLICABLE_BODY,
}


# --------------------------------------------------------------------------- #
# Unit conversions
# --------------------------------------------------------------------------- #
def au_to_km(au):
    """Astronomical units -> kilometres."""
    return au * AU_KM


def au_to_ld(au):
    """Astronomical units -> lunar distances (au * km_per_au / km_per_LD)."""
    return au * AU_KM / LD_KM


def diameter_from_h(h, albedo):
    """Estimated diameter in km from absolute magnitude H and an assumed albedo.

    Works for a single number, a numpy array or a pandas column. Missing H -> missing result.
    """
    if albedo <= 0:
        raise ValueError("albedo must be positive")
    if not isinstance(h, pd.Series):
        h = np.asarray(h, dtype="float64")
    return DIAMETER_H_CONSTANT_KM / np.sqrt(albedo) * np.power(10.0, -h / 5.0)


# --------------------------------------------------------------------------- #
# Groups of new columns
# --------------------------------------------------------------------------- #
def add_distance_features(df):
    """Add distance in km and lunar distances (LD), log10 distance and uncertainty range."""
    out = df.copy()
    out["dist_km"] = au_to_km(out["dist"])
    out["dist_ld"] = au_to_ld(out["dist"])

    # log10 only for positive distances (others become NaN). Log is used because
    # distances span several orders of magnitude.
    positive_ld = out["dist_ld"].where(out["dist_ld"] > 0)
    out["log10_dist_ld"] = np.log10(positive_ld)

    # The 3-sigma minimum / maximum distance, also in LD.
    if "dist_min" in out.columns:
        out["dist_min_ld"] = au_to_ld(out["dist_min"])
    if "dist_max" in out.columns:
        out["dist_max_ld"] = au_to_ld(out["dist_max"])

    # Width of the uncertainty range: absolute (LD) and relative to the distance.
    if "dist_min" in out.columns and "dist_max" in out.columns:
        out["dist_range_ld"] = out["dist_max_ld"] - out["dist_min_ld"]
        positive_dist = out["dist"].where(out["dist"] > 0)
        out["dist_range_rel"] = (out["dist_max"] - out["dist_min"]) / positive_dist
    return out


def add_velocity_features(df):
    """Add v_gain_kms = v_rel - v_inf: speed gained from the planet's gravity."""
    out = df.copy()
    if "v_inf" in out.columns:
        out["v_gain_kms"] = out["v_rel"] - out["v_inf"]
    return out


def add_time_features(df, reference_time=None):
    """Add year, month, decade, and is_future (approach after the download time)."""
    out = df.copy()
    out["year"] = out["approach_time"].dt.year
    out["month"] = out["approach_time"].dt.month
    out["decade"] = (out["year"] // 10) * 10        # 1987 -> 1980 (integer division)
    if reference_time is not None:
        out["is_future"] = out["approach_time"] > pd.Timestamp(reference_time)
    return out


def add_size_features(df):
    """Add True/False "is this value known?" columns and diameter estimates from H."""
    out = df.copy()

    # Availability flags (missing is never treated as 0).
    if "diameter" in out.columns:
        out["has_diameter"] = out["diameter"].notna()
    else:
        out["has_diameter"] = False
    if "diameter_sigma" in out.columns:
        out["has_diameter_sigma"] = out["diameter_sigma"].notna()
    else:
        out["has_diameter_sigma"] = False
    if "h" in out.columns:
        out["has_h"] = out["h"].notna()
    else:
        out["has_h"] = False

    # Relative diameter uncertainty = sigma / diameter.
    if "diameter" in out.columns and "diameter_sigma" in out.columns:
        positive_diameter = out["diameter"].where(out["diameter"] > 0)
        out["diameter_rel_uncertainty"] = out["diameter_sigma"] / positive_diameter

    # Three size ESTIMATES from H, one per assumed albedo (bright, default, dark).
    if "h" in out.columns:
        out["diameter_est_km_albedo_0p25"] = diameter_from_h(out["h"], ALBEDO_HIGH)
        out["diameter_est_km_albedo_0p14"] = diameter_from_h(out["h"], ALBEDO_CNEOS_DEFAULT)
        out["diameter_est_km_albedo_0p05"] = diameter_from_h(out["h"], ALBEDO_LOW)
    return out


# --------------------------------------------------------------------------- #
# Descriptive categories (NOT risk levels)
# --------------------------------------------------------------------------- #
def earth_mask(df):
    """True for rows whose body is Earth (lunar distance only makes sense for Earth)."""
    if "body" in df.columns:
        return df["body"].astype("string").str.lower().eq("earth")
    return pd.Series(True, index=df.index)


def add_proximity_class(out, is_earth):
    """Bin the distance in LD. Non-Earth rows get 'N/A (non-Earth body)'."""
    # pd.cut puts each value into a bin, like a C++ if/else-if ladder on thresholds.
    # right=False means bins are [left, right): 1.0 LD goes into "1-5 LD".
    prox = pd.cut(out["dist_ld"], bins=PROXIMITY_BINS_LD, labels=PROXIMITY_LABELS,
                  right=False, include_lowest=True)
    prox = prox.astype("object").where(is_earth, NOT_APPLICABLE_BODY)
    out["proximity_class"] = pd.Categorical(prox, categories=PROXIMITY_LABELS + [NOT_APPLICABLE_BODY],
                                            ordered=True)
    return out


def add_velocity_class(out):
    """Bin the relative velocity v_rel at 10 and 20 km/s."""
    out["velocity_class"] = pd.cut(out["v_rel"], bins=VELOCITY_BINS_KMS, labels=VELOCITY_LABELS,
                                   right=False, include_lowest=True)
    return out


def add_size_class(out):
    """Bin H at 22 and 26 (H unknown when missing)."""
    if "h" in out.columns:
        h = out["h"]
    else:
        h = pd.Series(np.nan, index=out.index)

    # Start with "H unknown" for every row, then overwrite where H is known.
    size = pd.Series(SIZE_LABELS[3], index=out.index, dtype="object")
    size[h <= H_LARGER_MAX] = SIZE_LABELS[0]
    size[(h > H_LARGER_MAX) & (h <= H_MEDIUM_MAX)] = SIZE_LABELS[1]
    size[h > H_MEDIUM_MAX] = SIZE_LABELS[2]
    out["size_class"] = pd.Categorical(size, categories=SIZE_LABELS, ordered=True)
    return out


def add_exploratory_group(out, is_earth):
    """Combine proximity (closer than 5 LD or not) with the size class into groups A-D."""
    near = out["proximity_class"].isin(["<1 LD", "1-5 LD"])
    larger = out["size_class"].eq(SIZE_LABELS[0])
    unknown = out["size_class"].eq(SIZE_LABELS[3])

    # Default: non-Earth. Earth rows start as D, then near rows become A, B or C.
    group = pd.Series(GROUP_LABELS["N"], index=out.index, dtype="object")
    group[is_earth] = GROUP_LABELS["D"]
    group[near & larger] = GROUP_LABELS["A"]
    group[near & ~larger & ~unknown] = GROUP_LABELS["B"]
    group[near & unknown] = GROUP_LABELS["C"]
    out["exploratory_group"] = pd.Categorical(group, categories=list(GROUP_LABELS.values()), ordered=False)
    return out


def categorize_events(df):
    """Add the 4 descriptive category columns."""
    out = df.copy()
    if "dist_ld" not in out.columns:
        out = add_distance_features(out)
    is_earth = earth_mask(out)
    out = add_proximity_class(out, is_earth)
    out = add_velocity_class(out)
    out = add_size_class(out)
    out = add_exploratory_group(out, is_earth)
    return out


def build_features(df, reference_time=None):
    """Run every feature step in order and return the new table."""
    out = add_distance_features(df)
    out = add_velocity_features(out)
    out = add_time_features(out, reference_time=reference_time)
    out = add_size_features(out)
    out = categorize_events(out)
    return out
