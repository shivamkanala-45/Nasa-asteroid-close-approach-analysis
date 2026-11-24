"""Plotting helpers, so every chart has the same style and is saved the same way.

matplotlib = the basic plotting library (figure = window, ax/axes = one plot inside it).
seaborn    = built on matplotlib; nicer statistical plots (boxplot, heatmap) in one call.
Every figure is saved as a PNG in reports/figures/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

RANDOM_SEED = 42
# reports/figures inside the project folder (this file is in <project>/src/).
FIGURE_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"


def apply_style():
    """Set one readable style for all plots."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.figsize": (8, 4.5),
        "figure.dpi": 100,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "savefig.bbox": "tight",
        "figure.max_open_warning": 100,
    })


def save_figure(fig, filename, dpi=150):
    """Save a figure as reports/figures/<filename>.png and return the path."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    if not filename.lower().endswith(".png"):
        filename = filename + ".png"
    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def plot_histogram(series, title, xlabel, log_x=False, bins=60, ax=None):
    """Histogram of one column. log_x=True uses log-spaced bins (positive values only)."""
    # Create a new plot if none was given.
    if ax is None:
        fig, ax = plt.subplots()

    if log_x:
        # Bins of equal width on a log scale, e.g. 0.01-0.1, 0.1-1, 1-10 ...
        s = series.dropna()
        s = s[s > 0]
        edges = np.logspace(np.log10(s.min()), np.log10(s.max()), bins + 1)
        ax.hist(s, bins=edges, color="#4C72B0", edgecolor="white", linewidth=0.3)
        ax.set_xscale("log")
        xlabel = f"{xlabel} (log scale)"
    else:
        ax.hist(series.dropna(), bins=bins, color="#4C72B0", edgecolor="white", linewidth=0.3)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of events")
    return ax


def plot_missingness_bar(summary, title="Missing values by column"):
    """Horizontal bar chart of missing % per column (input: cleaning.missing_summary)."""
    data = summary.sort_values("missing_pct")
    fig, ax = plt.subplots(figsize=(8, max(3, 0.32 * len(data))))
    ax.barh(data.index, data["missing_pct"], color="#C44E52")
    ax.set_xlabel("Missing (%)")
    ax.set_title(title)
    ax.set_xlim(0, max(100, data["missing_pct"].max()))
    return fig


def plot_missingness_matrix(df, columns, sample_n=1500):
    """Picture of missing cells (dark = missing) for a random sample of rows."""
    sample = df[list(columns)].sample(min(sample_n, len(df)), random_state=RANDOM_SEED)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.heatmap(sample.isna().to_numpy(), cmap=["#E8E8E8", "#222222"], cbar=False, ax=ax,
                xticklabels=list(columns), yticklabels=False)
    ax.set_title(f"Missingness pattern (random sample of {len(sample):,} rows; dark = missing)")
    ax.set_ylabel("Sampled events")
    return fig


def plot_box_by_group(df, value, group, ylabel, title, log_y=False, order=None):
    """One boxplot of `value` for each category of `group` (e.g. distance per body)."""
    data = df[[value, group]].dropna()
    if log_y:
        data = data[data[value] > 0]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.boxplot(data=data, x=group, y=value, order=order, ax=ax, color="#8FB8DE", fliersize=1.5)
    if log_y:
        ax.set_yscale("log")
        ylabel = ylabel + " (log scale)"
    ax.set_ylabel(ylabel)
    ax.set_xlabel(group)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    return fig


def plot_density_scatter(df, x, y, xlabel, ylabel, title, log_x=False, log_y=False, gridsize=60):
    """Hexbin plot: a scatter plot for very many points, colour = how many points per cell."""
    data = df[[x, y]].dropna()
    if log_x:
        data = data[data[x] > 0]
    if log_y:
        data = data[data[y] > 0]

    x_scale = "log" if log_x else "linear"
    y_scale = "log" if log_y else "linear"
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    hb = ax.hexbin(data[x], data[y], gridsize=gridsize, bins="log", mincnt=1, cmap="viridis",
                   xscale=x_scale, yscale=y_scale)
    fig.colorbar(hb, ax=ax, label="Events per cell (log)")
    ax.set_xlabel(xlabel + (" (log scale)" if log_x else ""))
    ax.set_ylabel(ylabel + (" (log scale)" if log_y else ""))
    ax.set_title(f"{title} (n={len(data):,})")
    return fig


def plot_corr_heatmap(corr, title):
    """Coloured table of correlations (-1 blue ... +1 red) with the numbers written in."""
    size = len(corr)
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * size), max(5, 0.8 * size)))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True,
                linewidths=0.5, ax=ax, cbar_kws={"label": "Correlation"})
    ax.set_title(title)
    return fig


def plot_counts(counts, title, xlabel, ylabel="Number of events", kind="bar", figsize=(10, 4.5)):
    """Bar chart (or line chart) of counts, e.g. events per decade."""
    fig, ax = plt.subplots(figsize=figsize)
    if kind == "line":
        ax.plot(counts.index, counts.values, color="#4C72B0", linewidth=1.5)
    else:
        ax.bar(counts.index.astype(str), counts.values, color="#4C72B0")
        ax.tick_params(axis="x", rotation=45)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig
