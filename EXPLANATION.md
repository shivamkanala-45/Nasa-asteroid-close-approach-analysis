# EXPLANATION: How this project works (for beginners)

Written for someone who knows **C++, DSA, probability and statistics** but is new to Python data libraries.

Contents:
1. [The project in one paragraph](#1-the-project-in-one-paragraph)
2. [Project flow in simple English](#2-project-flow-in-simple-english)
3. [The data: what each column means](#3-the-data-what-each-column-means)
4. [Python data tools for a C++ programmer](#4-python-data-tools-for-a-c-programmer)
5. [Every library and function used](#5-every-library-and-function-used)
6. [Statistics and ML used, as math](#6-statistics-and-ml-used-as-math)
7. [How the simplified version was checked (and what could differ)](#7-how-the-simplified-version-was-checked-and-what-could-differ)
8. [10 likely viva questions with short answers](#8-10-likely-viva-questions-with-short-answers)

---

## 1. The project in one paragraph

NASA/JPL computes when asteroids and comets pass close to Earth and other planets, in the past and in the future. We download at least 100,000 of these **close-approach events** from NASA's public API, clean them carefully (we never invent or silently delete data), and explore them. We ask how far, how fast and how bright the objects are, what is missing, how variables relate, and how the counts change over time. At the end we try two simple machine-learning tools, K-means (grouping) and Isolation Forest (finding unusual events), plus simple if/else categories. **No impact prediction is made anywhere.**

---

## 2. Project flow in simple English

```
NASA CAD API
    │  notebook 01  (EDA step 1: loading)
    ▼
Try queries from narrow to broad ──► first one with ≥ 100,000 events
    │
    ▼
Download page by page (10,000 rows per request) ──► data/raw/cad_raw_<time>.csv.gz  (never edited)
    │  notebook 01  (EDA step 2: cleaning)
    ▼
Clean: strip text → drop exact duplicates → text→numbers → parse dates → flag strange values
    │                                                   ──► data/processed/close_approaches_clean.csv.gz
    │  notebook 02  (EDA steps 3-6)
    ▼
Add features (km, lunar distances, year, size estimates, categories)
Summary statistics → plots (1 variable, categories, 2 variables, many variables, time) → outliers
    │                                                   ──► close_approaches_features.csv.gz, figures, results JSON
    │  notebook 03  (ML extension)
    ▼
Earth events only → scale features → K-means → Isolation Forest → rule-based groups A-D
```

**Notebook 01: Collect and clean**
1. **Pick a query.** NASA's default query (Earth only, next 60 days) is tiny. We have a ladder of 9 queries, each wider than the last: more planets, a larger distance limit, then all objects instead of only near-Earth objects. For each one we ask only "how many rows would you return?". We pick the **first** one that reaches 100,000. This keeps the dataset as narrow as possible.
2. **Download.** The API gives at most 10,000 rows per request, so we loop page by page. We download everything, because rows are sorted by date and stopping early would keep only old years. We then check that we got exactly as many rows as promised.
3. **Save raw.** The download is saved untouched, with a JSON "metadata" file: when, which query, and how many rows.
4. **Clean.** There are 8 small steps, each counted in a `report`. Exact duplicate rows are removed. Strange values are **flagged** with `qc_*` columns, **not deleted**. Impossible diameters (≤ 0) become missing. A missing diameter is never set to 0.
5. **Inspect.** We look at missing values, duplicates and consistency checks (for example dist_min ≤ dist ≤ dist_max).

**Notebook 02: The EDA**
1. Load the cleaned table and look at its shape, types, missing values and basic statistics.
2. Check *who* has a diameter. Bright objects have one much more often, so diameter data is **biased**.
3. Add new columns (features), such as distance in lunar distances.
4. Summary statistics: mean, median, percentiles, skewness, kurtosis. Then show that log10 makes skewed variables more symmetric.
5. Plots:
   - histograms and boxplots of each variable;
   - counts per planet/moon;
   - distance vs speed and brightness vs diameter;
   - correlation heatmaps and a pair plot;
   - events per year, decade and month.
6. Outliers with the IQR rule. They are flagged, not deleted.
7. Save the feature table and the key numbers.

**Notebook 03: ML extension**
1. Use only Earth events and 3 features: log distance, speed and brightness H. Standardise them.
2. **K-means:** try k = 2…10, pick the k with the best silhouette score, describe each cluster with medians, and check stability with a second seed.
3. **Isolation Forest:** flag the 1% most unusual events, compare them with the rest, and test 0.5% and 2%.
4. **Rule-based groups A-D:** count events per group and compare with the clusters.

---

## 3. The data: what each column means

### Columns from NASA (raw, all arrive as text)
| Column | Meaning | Unit |
|---|---|---|
| `des` | object designation (its ID), e.g. `2001 WN5` | text |
| `orbit_id` | which orbit solution NASA used | text |
| `jd` | time of closest approach as a Julian date (days since 4713 BC) | days |
| `cd` | same time as calendar text, e.g. `2028-Jun-26 05:23` | TDB time |
| `dist` | nominal (best-estimate) closest distance | au |
| `dist_min`, `dist_max` | smallest / largest possible distance (3-sigma) | au |
| `v_rel` | speed relative to the planet at closest approach | km/s |
| `v_inf` | speed far away from the planet ("before gravity pulls it") | km/s |
| `t_sigma_f` | 3-sigma uncertainty of the time, e.g. `2_09:08` = 2 d 9 h 8 min | text |
| `body` | planet/moon approached (only sent when we ask for all bodies) | text |
| `h` | absolute magnitude: brightness. **Larger H = fainter = usually smaller** | magnitude |
| `diameter`, `diameter_sigma` | measured size and its uncertainty. **Mostly missing** | km |
| `fullname` | full readable name | text |

### Columns we add
| Column | Meaning |
|---|---|
| `source_row_id` | row number in the raw file (to trace back) |
| `approach_time` | `cd` converted to a real datetime |
| `t_sigma_minutes`, `t_sigma_is_bound` | time uncertainty in minutes; True if the text was "< x" or "> x" |
| `qc_jd_cd_mismatch` | jd and cd differ by > 2 minutes |
| `qc_nonpositive_dist` | dist ≤ 0 |
| `qc_dist_range_inconsistent` | dist outside [dist_min, dist_max] |
| `qc_negative_velocity` | a negative speed |
| `qc_h_implausible` | H outside [-2, 35] |
| `dup_event_key` | another row has the same (des, jd, body) |
| `dist_km`, `dist_ld`, `log10_dist_ld` | distance in km, in lunar distances, and log10 of LD |
| `dist_min_ld`, `dist_max_ld`, `dist_range_ld`, `dist_range_rel` | uncertainty range in LD, and relative to the distance |
| `v_gain_kms` | `v_rel − v_inf`: speed gained from the planet's gravity |
| `year`, `month`, `decade`, `is_future` | time parts; is_future = after the download time |
| `has_diameter`, `has_diameter_sigma`, `has_h` | is the value known? |
| `diameter_est_km_albedo_0p05/0p14/0p25` | size **estimated** from H for 3 assumed albedos |
| `h_band` | H in 4 bands (≤18, 18-22, 22-26, >26) |
| `proximity_class`, `velocity_class`, `size_class`, `exploratory_group` | simple categories (not risk) |
| `log10_diameter`, `log10_t_sigma` | log versions for the correlation matrix |
| `iqr_outlier_dist_ld`, `iqr_outlier_v_rel`, `iqr_outlier_h`, `any_qc_flag` | outlier and quality flags |

### Key concepts
* **au** (astronomical unit) = 149,597,870.7 km (Sun–Earth distance). **LD** (lunar distance) = 384,400 km (Earth–Moon). So 1 au ≈ 389 LD.
* **Absolute magnitude H** is brightness at a standard distance. The magnitude scale is reversed and logarithmic: +5 magnitudes = 100× fainter.
* **Size from H:** D = 1329 / √p × 10^(−H/5) km, where p = albedo (how reflective the surface is). p is usually unknown, so D is only an estimate. A dark object (p = 0.05) gives about 2.2× the size of a bright one (p = 0.25), since √(0.25/0.05) ≈ 2.2.
* **Event vs object:** 1 object can have 50 events. Counting events ≠ counting asteroids.

---

## 4. Python data tools for a C++ programmer

| Python / pandas | C++ analogy |
|---|---|
| `list` `[1, 2, 3]` | `std::vector<T>` |
| `dict` `{"a": 1}` | `std::unordered_map<string, T>` |
| `tuple` `(a, b)` returned from a function | `std::pair` / `std::tuple` |
| `pd.Series` | a `vector<T>` where every element also has a label (the index) |
| `pd.DataFrame` | a table: `vector<Row>` where each column is a typed vector with a name, like a `struct` per row |
| `df["v_rel"]` | take one column (a Series) |
| `df["v_rel"] * 2` (a "vectorised" operation) | a `for` loop over the column, already compiled in C, so it is much faster than a Python loop |
| `df[df["v_rel"] > 20]` (boolean mask) | `vector<bool> mask`; keep rows where `mask[i]` is true (like `std::copy_if`) |
| `df.groupby("body")["v_rel"].median()` | build `map<body, vector<double>>`, then compute the median of each vector |
| `pd.cut(x, bins)` | for each x, `upper_bound` in the sorted thresholds → bin label |
| `NaN` / `NaT` / `pd.NA` | a "missing" marker (like `std::optional` being empty). `NaN == NaN` is false! |
| `.dropna()` | remove the missing elements before computing |
| `def f(x):` | a function. No types are written; Python checks types at runtime |
| `for col in columns:` | range-based `for (auto& col : columns)` |
| `import pandas as pd` | `#include` plus a namespace alias |
| `try: … except Error:` | `try { … } catch (Error&) { … }` |
| `display(table)` | Jupyter's pretty-print of a table (like `cout`, but formatted) |

**Why not write every loop by hand?** For 100,000+ rows, a Python `for` loop is roughly 50-100× slower than the vectorised pandas version. So we use loops for small things (over columns, profiles or k values) and vectorised operations for "do this to every row".

---

## 5. Every library and function used

### 5.1 Libraries
| Library | One-line meaning |
|---|---|
| `json` | read/write JSON files (text format of key-value data, like a nested map) |
| `re` | regular expressions: match text patterns such as dates |
| `sys` | access to Python itself; we use `sys.path` to make `src/` importable |
| `time` | `time.sleep(s)` pauses the program for s seconds |
| `pathlib.Path` | file paths that work on Windows and Linux (`ROOT / "data" / "raw"`) |
| `datetime` | dates and times (download timestamp in UTC) |
| `requests` | send HTTP requests to a web API and receive the answer |
| `pandas` (`pd`) | tables (DataFrame) and columns (Series): load, clean, group, summarise |
| `numpy` (`np`) | fast math on arrays (log10, sqrt, random numbers, NaN) |
| `scipy.stats` | statistics; here `spearmanr` (Spearman correlation) |
| `matplotlib.pyplot` (`plt`) | basic plotting: figures, axes, lines, bars, histograms |
| `seaborn` (`sns`) | statistical plots on top of matplotlib: boxplot, heatmap, pairplot, scatterplot |
| `sklearn` (scikit-learn) | machine learning: KMeans, IsolationForest, PCA, StandardScaler, metrics |

### 5.2 Our own functions (`src/`)

**`src/nasa_api.py`**
| Function | What it does |
|---|---|
| `build_query_profiles()` | returns the 9 queries, ordered from narrow to broad |
| `request_json(params)` | sends one GET request and returns the JSON as a dict; retries 3× on network/server errors |
| `get_total(params)` | asks the API how many rows a query matches, without downloading them |
| `fetch_close_approaches(params, ...)` | downloads all pages and returns (table, metadata); checks the row count |
| `count_events(df)` | counts events, unique objects, exact duplicates and repeated event keys |
| `save_raw(df, metadata, raw_dir)` | saves the raw CSV + metadata JSON with a time-stamped name (never overwrites) |
| `latest_raw_file(raw_dir)` | finds the newest raw file |
| `load_raw(path)` | loads a raw CSV with all columns as text |

**`src/cleaning.py`**
| Function | What it does |
|---|---|
| `parse_cd(series)` | calendar text → datetime (handles month names by hand, so it works with any Windows language setting) |
| `parse_t_sigma(value)` | `"2_09:08"` → (3428 minutes, False) |
| `strip_text(...)` | step 1: trim spaces; empty text → missing |
| `remove_exact_duplicates(...)` | step 2: delete rows identical to an earlier row |
| `convert_numbers(...)` | step 3: text → float; failures → NaN (counted) |
| `add_dates(...)` | step 4: parse `cd`, compare with `jd`, flag > 2 min differences |
| `add_t_sigma(...)` | step 5: time uncertainty → minutes |
| `add_quality_flags(...)` | step 6: add `qc_*` flags; diameter ≤ 0 → missing |
| `add_body_column(...)` | step 7: add `body = "Earth"` if the API did not send it |
| `flag_repeated_event_keys(...)` | step 8: flag rows with the same (des, jd, body) |
| `clean_close_approaches(raw)` | runs steps 1-8 and returns (clean table, report) |
| `missing_summary(df)` | table of missing count / % / distinct values per column |
| `numeric_summary(df, cols)` | count, mean, std, min, percentiles, max, skew and kurtosis per column |
| `iqr_bounds(series)` | Tukey fences (Q1 − 1.5·IQR, Q3 + 1.5·IQR) |
| `iqr_outlier_summary(df, cols, ...)` | number of values outside the fences per column |
| `date_span_years(times)` | length of the date range in years (avoids a pandas 292-year overflow) |
| `save_clean(...)` / `load_processed(path)` | save / load the cleaned table and report |

**`src/features.py`**
| Function | What it does |
|---|---|
| `au_to_km`, `au_to_ld` | unit conversions |
| `diameter_from_h(h, albedo)` | size estimate D = 1329/√p · 10^(−H/5) |
| `add_distance_features(df)` | dist_km, dist_ld, log10_dist_ld, uncertainty range |
| `add_velocity_features(df)` | v_gain_kms = v_rel − v_inf |
| `add_time_features(df, t)` | year, month, decade, is_future |
| `add_size_features(df)` | has_* flags, relative diameter uncertainty, 3 size estimates |
| `earth_mask(df)` | True for Earth rows |
| `add_proximity_class`, `add_velocity_class`, `add_size_class`, `add_exploratory_group` | the 4 rule-based categories |
| `categorize_events(df)` | adds all 4 categories |
| `build_features(df, t)` | runs all feature steps |

**`src/visualization.py`**
| Function | What it does |
|---|---|
| `apply_style()` | one common look for every plot |
| `save_figure(fig, name)` | saves a PNG in `reports/figures/` |
| `plot_histogram(...)` | histogram, optionally with log-spaced bins |
| `plot_missingness_bar(...)` | bar chart of % missing per column |
| `plot_missingness_matrix(...)` | picture of missing cells for a random sample of rows |
| `plot_box_by_group(...)` | one boxplot per category |
| `plot_density_scatter(...)` | hexbin: a scatter plot for very many points (colour = count) |
| `plot_corr_heatmap(...)` | coloured correlation matrix with numbers |
| `plot_counts(...)` | bar or line chart of counts |
| `describe_series(...)` *(in notebook 02)* | one computed sentence: n, median, mean, skewness and shape |

### 5.3 Library functions used in the code
| Function | One-line meaning |
|---|---|
| `requests.get(url, params, timeout)` | send an HTTP GET request; `.status_code` = 200 means OK, `.json()` = parse the answer |
| `pd.read_csv(path, ...)` / `df.to_csv(path, ...)` | load / save a table from/to a CSV file (`.gz` = compressed) |
| `pd.DataFrame(rows, columns=...)` | build a table from a list of rows |
| `df.shape`, `df.dtypes`, `df.columns` | (rows, columns), type of each column, column names |
| `df.head(n)`, `df.sample(n, random_state=42)` | first n rows / n random rows (seed 42 = same rows every time) |
| `df.copy()` | independent copy (like copying a vector, not taking a reference) |
| `df.isna()`, `.notna()`, `.dropna()` | missing? / not missing? / drop missing values |
| `df.duplicated(subset, keep)` | True for rows that repeat an earlier row |
| `pd.to_numeric(x, errors="coerce")` | text → number; impossible text → NaN |
| `pd.to_datetime(...)` | numbers/text → datetime |
| `series.str.strip()`, `.str.extract(pattern)`, `.str.lower()`, `.str.title()` | text operations on every row of a column |
| `series.map(dict)` | replace every value using a lookup table |
| `series.astype(type)` | convert the column type |
| `series.where(cond)` | keep values where cond is True, put NaN elsewhere |
| `df.loc[mask, col] = value` | set values in the rows where mask is True |
| `df.insert(pos, name, values)` | add a column at a given position |
| `df.sort_values(cols, kind="mergesort")` | sort rows; mergesort is stable |
| `df.reset_index(drop=True)` | renumber rows 0…n−1 |
| `series.value_counts()` | how often each value occurs |
| `series.nunique()` | number of distinct values |
| `df.describe()` | count, mean, std, min, quartiles, max |
| `series.mean()`, `.median()`, `.std()`, `.quantile(q)`, `.skew()`, `.kurt()` | summary statistics |
| `df.groupby(col).agg(...)` / `.size()` | group rows by a column, then summarise each group |
| `pd.cut(x, bins, labels)` | put numbers into labelled intervals |
| `pd.Categorical(...)` | a column with a fixed, optionally ordered, set of labels |
| `pd.crosstab(a, b)` | count table of two categorical columns |
| `df.corr(method="spearman" / "pearson")` | correlation matrix |
| `df.nlargest(n, col)` | n rows with the largest values in col |
| `df.select_dtypes(include="number")` | keep only numeric columns |
| `series.dt.year`, `.dt.month`, `.dt.total_seconds()` | parts of datetimes / time differences |
| `np.log10`, `np.sqrt`, `np.power`, `np.isinf`, `np.arange`, `np.linspace`, `np.logspace` | elementwise math; ranges of numbers |
| `np.random.default_rng(42).choice(n, size, replace=False)` | random sample of row positions (seeded) |
| `stats.spearmanr(x, y)` | Spearman correlation ρ and p-value |
| `plt.subplots(rows, cols)` | create a figure with a grid of plots |
| `ax.hist`, `ax.plot`, `ax.bar`, `ax.barh`, `ax.scatter`, `ax.hexbin` | histogram, line, bars, horizontal bars, points, hex-density |
| `ax.set_xscale("log")`, `ax.set_title`, `ax.set_xlabel`, `ax.legend`, `ax.axvline` | axis scale, labels, legend, vertical line |
| `fig.tight_layout()`, `fig.savefig()`, `plt.show()` | fix spacing, save, display |
| `sns.boxplot`, `sns.heatmap`, `sns.pairplot`, `sns.scatterplot` | boxplot, coloured matrix, all pairwise scatter plots, scatter plot coloured by group |
| `StandardScaler().fit(X)` / `.transform(X)` | learn mean/std, then compute z = (x − mean)/std |
| `KMeans(n_clusters, n_init=10, random_state).fit(X)` | K-means; n_init=10 means 10 random starts, keep the best |
| `km.predict(X)`, `km.inertia_`, `km.labels_`, `km.cluster_centers_` | cluster of each point, total within-cluster squared distance, labels, centres |
| `silhouette_score(X, labels)` | average silhouette (−1…1, higher = better separated) |
| `adjusted_rand_score(a, b)` | agreement of two groupings (1 = identical, ~0 = random) |
| `PCA(n_components=2).fit(X)`, `.transform`, `.explained_variance_ratio_` | project to 2 axes for drawing; share of variance kept |
| `IsolationForest(n_estimators=200, contamination=0.01).fit(X)` | anomaly detector; `.predict` = −1 for anomaly, `.score_samples` = normality score |

---

## 6. Statistics and ML used, as math

* **Mean** x̄ = Σxᵢ/n. **Median** = middle value after sorting (O(n log n), or O(n) with nth_element). The median is robust: one huge value cannot move it.
* **Quartiles** Q1, Q3 = 25th and 75th percentiles. **IQR** = Q3 − Q1.
* **Skewness** = E[(X−μ)³]/σ³. It is > 0 for a long right tail (distance, diameter). **Kurtosis** (excess) = E[(X−μ)⁴]/σ⁴ − 3. It is > 0 for heavy tails.
* **Log transform:** if values span 0.01 to 100, log10 maps them to −2 to 2, which spreads them evenly. A multiplicative pattern becomes additive.
* **Tukey outliers:** x < Q1 − 1.5·IQR or x > Q3 + 1.5·IQR. For a normal distribution this flags about 0.7% of values.
* **Pearson r** = cov(X,Y)/(σₓσᵧ), which measures a *linear* relationship. **Spearman ρ** = Pearson r computed on the **ranks**, which measures any *monotonic* relationship and is robust to skew and outliers.
* **Standardisation (z-score):** z = (x − μ)/σ, giving mean 0 and std 1.
* **K-means** minimises inertia = Σ over clusters Σ over points ‖x − centreₖ‖². Lloyd's algorithm alternates between assigning each point to its nearest centre and recomputing each centre as the mean. It converges to a local minimum, so we use 10 random starts.
* **Silhouette** of a point: s = (b − a)/max(a, b). Here a = mean distance to points in its own cluster, and b = mean distance to points in the nearest other cluster.
* **Adjusted Rand index:** the share of point pairs on which two groupings agree, corrected for chance.
* **PCA:** finds the orthogonal directions of largest variance (eigenvectors of the covariance matrix). Here we use it only to draw 3-D data on 2 axes.
* **Isolation Forest:** random binary trees that split on a random feature at a random value. An unusual point is isolated quickly (short path). The score is based on the average path length over 200 trees. `contamination = 0.01` sets the threshold so 1% are flagged.
* **Jaccard overlap** = |A ∩ B| / |A ∪ B|, used to compare the flagged sets at 0.5%, 1% and 2%.

---

## 7. How the simplified version was checked (and what could differ)

**What I tested.** NASA's API was not reachable from the build machine, so I built a **fake local copy of the API** with random, CAD-like data (175,000 rows). I did not include this fake data in the project. Then:

1. I ran the **original** project (notebooks 01 → 02 → 03) against the fake API.
2. I ran the **simplified** project against the same fake API. It chose the same query and downloaded exactly the same rows.
3. I ran the simplified notebooks on the **same raw file and the same download time** as the original, and compared:

| Output | Result |
|---|---|
| Cleaned table (`close_approaches_clean.csv.gz`, 104,467 rows) | **identical** |
| Feature table (`close_approaches_features.csv.gz`) | **identical** |
| Cleaning report (all counts) | **identical** |
| `results_summary.json`: 17 kept EDA numbers + all 11 ML numbers | **identical** |
| 20 figures made by both versions | **identical pixel-for-pixel** |
| All notebook 03 printed outputs | **identical** |
| All 3 notebooks run start to finish | no errors, no warnings |

I also tested the cleaning and feature functions on a small, hand-made table with tricky values. It had no `body` column, an impossible date, `inf`, bad text, `<`/`>`/`~` time strings, a negative distance and a zero diameter. The results were identical to the original's.

**Differences you should know about (none change the kept results):**

1. **Removed analyses** (README section 4): those sections, figures and JSON keys are no longer produced.
2. **Text trimming:** on pandas 3 (the version `pip` installs today), the original's "strip spaces" step only worked on the `fullname` column, because of a type-check detail. The simplified version strips spaces in all text columns, as the original intended. With NASA's data this gave the same values; the only visible change is that `cd` shows type `string` instead of `str` in one dtype table. If NASA ever sent spaces around a designation, the simplified version would treat `" 2001 AB"` and `"2001 AB"` as the same object. That is the correct behaviour.
3. **Safety checks removed from the download:** the original ran a pagination self-test before downloading. The simplified version still checks that the number of downloaded rows equals the number NASA promised, so missing rows are caught. A page that overlaps another page but has the right length would no longer be detected. NASA's documented paging makes this very unlikely.
4. **`get_total` edge case:** if NASA's reply contained neither a `total` nor a `count` of 0, the original stopped with an error. The simplified version treats it as 0 matching events, and that profile appears as 0 in the query table.
5. **Project-folder detection:** the notebooks find the project folder when Jupyter is started from the project folder or from `notebooks/` (the two usual ways). The original also searched higher parent folders.

---

## 8. 10 likely viva questions with short answers

**1. What is EDA, and what are its steps in your project?**
EDA (Exploratory Data Analysis) means understanding a dataset before modelling it. My steps are: loading (API download), cleaning (types, duplicates, flags), summary statistics, visualizations (one variable, two variables, many variables, time), outliers, and insights. Notebook 03 adds a small ML extension.

**2. Why is an "event" not the same as an "asteroid"?**
Each row is one close approach. The same asteroid can pass Earth many times between 1900 and 2200, so it appears in many rows. That is why I report events and unique designations separately. It also means rows are not independent observations.

**3. Why didn't you fill missing diameters with 0 or the mean?**
Zero is physically impossible, and the mean would invent data. Diameter is **missing not at random**: bright, well-studied objects have a diameter much more often (section 3 of notebook 02 shows this by H band). So missing means "unknown", and any analysis using diameter describes a biased subset.

**4. Why do you use a log scale for distance?**
Distances span several orders of magnitude and are strongly right-skewed. After log10 the skewness drops a lot (the table in section 5 shows the before/after values), so plots and correlations are not dominated by a few huge values.

**5. What is the difference between Pearson and Spearman correlation? Why use Spearman?**
Pearson measures a straight-line relationship and is sensitive to outliers and skew. Spearman is Pearson computed on ranks, so it captures any monotonic relationship and is robust. Our variables are skewed, so Spearman fits better. Correlation does not imply causation.

**6. How did you detect outliers, and why didn't you delete them?**
I used the Tukey IQR rule: below Q1 − 1.5·IQR or above Q3 + 1.5·IQR (on a log scale for skewed variables). Unusual values can be real, for example a very fast comet. So I flag them and check how many also have a cleaning quality flag; I don't delete them.

**7. Why does the number of events per year rise in recent decades? Are more asteroids coming?**
No. It reflects **discovery/observation bias**: modern telescope surveys have found many more objects, and future approaches can only be listed for objects already known today. Event counts measure our knowledge, not the real encounter rate.

**8. Why did you standardise the features before K-means?**
K-means uses Euclidean distance. Without scaling, the feature with the largest numeric range (H, about 20 units) would dominate log distance (about 3 units). The z-score gives every feature mean 0 and std 1, so each contributes equally.

**9. How did you choose the number of clusters k?**
I tried k = 2 to 10 and used two measures. The elbow (inertia) shows where adding clusters stops helping much. The silhouette score shows how well separated the clusters are; I picked the k with the highest silhouette. I checked stability by refitting with another seed and computing the adjusted Rand index (1 = identical).

**10. How does Isolation Forest work, and what does "contamination = 0.01" mean?**
It builds 200 random trees that split on random features at random values. An unusual point gets isolated after few splits (short path length), so it gets a high anomaly score. Contamination = 0.01 means the model flags the 1% most unusual events **by design**. So "1% anomalies" is a setting, not a finding, and an anomaly is not necessarily dangerous or an error.

*Bonus. Is this an impact-risk model?* No. It never estimates impact probability, and groups A-D are round-number descriptive categories. Real impact monitoring is NASA/JPL's Sentry system.
