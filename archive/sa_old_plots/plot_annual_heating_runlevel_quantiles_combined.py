#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================
# Main input: run-level KPI table created by your extraction script.
# Each row is treated as one run; sample_id * seed combinations are NOT
# averaged before calculating P5 / P50 / P95.
INPUT_CSV = Path(r"../new_plots/sa_results/run_level_kpis.csv")

OUTPUT_DIR = Path(r"../new_plots/output/annual_heating_runlevel_quantiles")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

KPI = "annual_heating_kWh"
VARIANT_ORDER = ["V1", "V1_20HH", "V2", "V2_20HH", "V3", "V4", "V5", "V6", "V7", "V8"]

SHOW_PLOTS = True
DPI = 300

# Same quantile colors across all variants
COLOR_P05 = "#c9d6ea"
COLOR_P50 = "#7ea6d8"
COLOR_P95 = "#2f5f98"
LEGEND_EDGE_COLOR = "#bdc1c5"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 8,
    "axes.titleweight": "normal",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})
# ============================================================


# ============================================================
# BASIC HELPERS
# ============================================================
def add_grid(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", alpha=0.30)
    ax.set_axisbelow(True)


def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def p5(x: pd.Series) -> float:
    return float(np.nanpercentile(x, 5))


def p50(x: pd.Series) -> float:
    return float(np.nanpercentile(x, 50))


def p95(x: pd.Series) -> float:
    return float(np.nanpercentile(x, 95))


def sort_variants(df: pd.DataFrame, col: str = "variant") -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.Categorical(out[col], categories=VARIANT_ORDER, ordered=True)
    return out.sort_values(col).reset_index(drop=True)


def style_legend(ax: plt.Axes) -> None:
    legend = ax.legend(frameon=True)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor(LEGEND_EDGE_COLOR)
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_alpha(1.0)


def format_value_labels(
    ax: plt.Axes,
    bars,
    values: np.ndarray,
    fmt: str = ",.0f",
    rotation: int = 0,
    ha: str = "center",
) -> None:
    ymax = ax.get_ylim()[1]
    offset = 0.012 * ymax
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            f"{value:{fmt}}",
            ha=ha,
            va="bottom",
            rotation=rotation,
            fontsize=8,
        )


# ============================================================
# RUN-LEVEL QUANTILE LOGIC
# ============================================================
def build_variant_quantiles_from_runs(df_runs: pd.DataFrame) -> pd.DataFrame:
    """
    Build P5 / P50 / P95 directly from run_level_kpis.csv.

    Important:
    Each row is treated as one run. Therefore, different seeds remain separate
    observations. This is the same main logic as plot_annual_heating_runlevel_quantiles.py.
    """
    rows: list[dict[str, float | int | str]] = []
    id_cols = [c for c in ["building_id", "weather_key", "sample_id", "seed"] if c in df_runs.columns]

    for variant, grp in df_runs.groupby("variant", observed=False):
        if grp.empty:
            continue

        s = pd.to_numeric(grp[KPI], errors="coerce").dropna()
        if s.empty:
            continue

        if id_cols:
            n_runs = int(grp[id_cols].drop_duplicates().shape[0])
        else:
            n_runs = int(s.shape[0])

        rows.append(
            {
                "variant": str(variant),
                "p05": p5(s),
                "p50": p50(s),
                "p95": p95(s),
                "mean": float(s.mean()),
                "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
                "n_runs": n_runs,
            }
        )

    out = sort_variants(pd.DataFrame(rows))
    return out[out["variant"].notna()].copy()


def load_and_prepare_run_level_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    df_runs = pd.read_csv(INPUT_CSV)

    required = {"variant", KPI}
    missing = required.difference(df_runs.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    df_runs = sort_variants(df_runs)
    df_runs = df_runs[df_runs["variant"].notna()].copy()

    df_variant = build_variant_quantiles_from_runs(df_runs)
    return df_runs, df_variant


# ============================================================
# PLOTS FROM plot_annual_heating_runlevel_quantiles.py
# ============================================================
def plot_quantile_interval(df_variant: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))

    labels: list[str] = []
    x: list[int] = []
    p50_vals: list[float] = []
    yerr_low: list[float] = []
    yerr_high: list[float] = []
    q_rows = []

    for v in VARIANT_ORDER:
        rows = df_variant.loc[df_variant["variant"] == v]
        if rows.empty:
            continue
        row = rows.iloc[0]
        labels.append(v)
        x.append(len(labels) - 1)
        p50_vals.append(float(row["p50"]))
        yerr_low.append(float(row["p50"] - row["p05"]))
        yerr_high.append(float(row["p95"] - row["p50"]))
        q_rows.append(row)

    ax.errorbar(x, p50_vals, yerr=[yerr_low, yerr_high], fmt="o", capsize=5)

    for xi, row in zip(x, q_rows):
        ax.text(xi, row["p50"], f"{row['p50']:,.0f}", ha="center", va="bottom", fontsize=9)
        ax.text(xi, row["p05"], f"{row['p05']:,.0f}", ha="center", va="top", fontsize=8)
        ax.text(xi, row["p95"], f"{row['p95']:,.0f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Annual heating demand: P5 / P50 / P95 across all runs")
    ax.set_xlabel("Variant")
    ax.set_ylabel("Annual heating demand [kWh]")
    add_grid(ax)

    out_path = OUTPUT_DIR / "01_quantile_interval_annual_heating_runlevel.pdf"
    savefig(fig, out_path)
    return out_path


def plot_boxplot_runs(df_runs: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))

    data = [
        pd.to_numeric(df_runs.loc[df_runs["variant"] == v, KPI], errors="coerce").dropna().values
        for v in VARIANT_ORDER
    ]
    labels = [v for v, arr in zip(VARIANT_ORDER, data) if len(arr) > 0]
    data = [arr for arr in data if len(arr) > 0]

    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_title("Annual heating demand: raw run distribution")
    ax.set_xlabel("Variant")
    ax.set_ylabel("Annual heating demand [kWh]")
    add_grid(ax)

    out_path = OUTPUT_DIR / "02_boxplot_annual_heating_runs.pdf"
    savefig(fig, out_path)
    return out_path


# ============================================================
# PLOTS FROM plot_annual_heating_quantile_bars and stacked.py
# ============================================================
def plot_grouped_quantile_bars(df_variant: pd.DataFrame) -> Path:
    labels = df_variant["variant"].astype(str).tolist()
    x = np.arange(len(labels), dtype=float)
    width = 0.24

    p05_vals = df_variant["p05"].to_numpy(dtype=float)
    p50_vals = df_variant["p50"].to_numpy(dtype=float)
    p95_vals = df_variant["p95"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10.5, 5.4))

    bars_p05 = ax.bar(x - width, p05_vals, width=width, label="P5", color=COLOR_P05)
    bars_p50 = ax.bar(x, p50_vals, width=width, label="P50", color=COLOR_P50)
    bars_p95 = ax.bar(x + width, p95_vals, width=width, label="P95", color=COLOR_P95)

    ax.set_title("Annual heating demand by variant: P5 / P50 / P95")
    ax.set_xlabel("Variant")
    ax.set_ylabel("Annual heating demand [kWh]")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    add_grid(ax)
    style_legend(ax)

    format_value_labels(ax, bars_p05, p05_vals)
    format_value_labels(ax, bars_p50, p50_vals)
    format_value_labels(ax, bars_p95, p95_vals)

    out_path = OUTPUT_DIR / "03_annual_heating_quantile_grouped_bars_runlevel.pdf"
    savefig(fig, out_path)
    return out_path


def plot_stacked_quantile_bars(df_variant: pd.DataFrame) -> Path:
    labels = df_variant["variant"].astype(str).tolist()
    x = np.arange(len(labels), dtype=float)
    width = 0.55

    p05_vals = df_variant["p05"].to_numpy(dtype=float)
    p50_vals = df_variant["p50"].to_numpy(dtype=float)
    p95_vals = df_variant["p95"].to_numpy(dtype=float)

    # Stacked interpretation:
    # bottom segment reaches P5, middle segment reaches P50, top segment reaches P95.
    seg1 = p05_vals
    seg2 = p50_vals - p05_vals
    seg3 = p95_vals - p50_vals

    fig, ax = plt.subplots(figsize=(10.5, 5.4))

    ax.bar(x, seg1, width=width, label="P5", color=COLOR_P05)
    ax.bar(x, seg2, width=width, bottom=seg1, label="P50 - P5", color=COLOR_P50)
    ax.bar(x, seg3, width=width, bottom=seg1 + seg2, label="P95 - P50", color=COLOR_P95)

    ax.set_title("Annual heating demand by variant: stacked quantile band")
    ax.set_xlabel("Variant")
    ax.set_ylabel("Annual heating demand [kWh]")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    add_grid(ax)
    style_legend(ax)

    ymax = ax.get_ylim()[1]
    offset = 0.012 * ymax
    for xi, v05, v50, v95 in zip(x, p05_vals, p50_vals, p95_vals):
        ax.text(xi, v05 + offset, f"{v05:,.0f}", ha="center", va="bottom", fontsize=8)
        ax.text(xi, v50 + offset, f"{v50:,.0f}", ha="center", va="bottom", fontsize=8)
        ax.text(xi, v95 + offset, f"{v95:,.0f}", ha="center", va="bottom", fontsize=8)

    out_path = OUTPUT_DIR / "04_annual_heating_quantile_stacked_bars_runlevel.pdf"
    savefig(fig, out_path)
    return out_path


# ============================================================
# PLOT FROM plot_annual_heating_quantile_with error_bars.py
# ============================================================
def plot_p50_bar_with_p5_p95_errorbars(df_variant: pd.DataFrame) -> Path:
    labels = df_variant["variant"].astype(str).tolist()
    x = np.arange(len(labels), dtype=float)
    width = 0.58

    # Keep this in kWh so it is consistent with the run-level quantile plots.
    p05_vals = df_variant["p05"].to_numpy(dtype=float)
    p50_vals = df_variant["p50"].to_numpy(dtype=float)
    p95_vals = df_variant["p95"].to_numpy(dtype=float)

    yerr_low = p50_vals - p05_vals
    yerr_high = p95_vals - p50_vals

    fig, ax = plt.subplots(figsize=(10.5, 5.4))

    bars = ax.bar(
        x,
        p50_vals,
        width=width,
        color=COLOR_P50,
        edgecolor="none",
        label="P50",
        zorder=3,
    )

    ax.errorbar(
        x,
        p50_vals,
        yerr=[yerr_low, yerr_high],
        fmt="none",
        ecolor="black",
        elinewidth=1.2,
        capsize=5,
        capthick=1.2,
        zorder=4,
        label="P5-P95 range",
    )

    ax.set_title("Annual heating demand by variant: P50 with P5/P95 range")
    ax.set_xlabel("Variant")
    ax.set_ylabel("Annual heating demand [kWh]")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    add_grid(ax)
    style_legend(ax)

    format_value_labels(ax, bars, p50_vals, fmt=",.0f", rotation=0)

    out_path = OUTPUT_DIR / "05_annual_heating_p50_with_p5_p95_errorbars_runlevel.pdf"
    savefig(fig, out_path)
    return out_path


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    df_runs, df_variant = load_and_prepare_run_level_data()

    quantile_csv = OUTPUT_DIR / "annual_heating_variant_quantiles_runlevel.csv"
    df_variant.to_csv(quantile_csv, index=False)

    saved_paths: list[Path] = [quantile_csv]
    saved_paths.append(plot_quantile_interval(df_variant))
    saved_paths.append(plot_boxplot_runs(df_runs))
    saved_paths.append(plot_grouped_quantile_bars(df_variant))
    saved_paths.append(plot_stacked_quantile_bars(df_variant))
    saved_paths.append(plot_p50_bar_with_p5_p95_errorbars(df_variant))

    print("Saved:")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
