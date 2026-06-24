#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_CSV = (SCRIPT_DIR / "../results/sa_results/_analysis_plots_fixed/run_level_kpis.csv").resolve()
OUTPUT_DIR = (SCRIPT_DIR / "../results/sa_results/_analysis_plots_fixed/matrices").resolve()

# Change KPI here if needed
KPI = "annual_heating_kWh"
# KPI = "overheating_hours_meanTair_gt_26C"

VARIANT_ORDER = ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8"]

SHOW_PLOTS = True
DPI = 150

# "median_pct_diff" or "mean_pct_diff"
CELL_STAT = "median_pct_diff"
# ============================================================


def base_building_group_key(building_id: str) -> str:
    s = str(building_id).strip()
    m = re.match(r"^(.*?)(?:[_-]?)(\d+)$", s)
    if not m:
        return s
    prefix = str(m.group(1)).strip()
    return prefix or s


def sort_variants(df: pd.DataFrame, col: str = "variant") -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.Categorical(out[col], categories=VARIANT_ORDER, ordered=True)
    return out.sort_values(col)


def savefig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def build_sample_summary(df_runs: pd.DataFrame) -> pd.DataFrame:
    # one row = one sample for one variant, aggregated across seeds
    group_cols = ["base_building_group", "variant", "sample_id"]
    df = (
        df_runs
        .groupby(group_cols, dropna=False)[KPI]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
        .rename(
            columns={
                "mean": f"{KPI}_mean",
                "std": f"{KPI}_std",
                "min": f"{KPI}_min",
                "max": f"{KPI}_max",
            }
        )
    )
    return sort_variants(df)


def build_pairwise_percent_matrix(df_sample: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Pivot to matched sample means:
    # rows = (base_building_group, sample_id)
    # cols = variant
    # values = KPI mean across seeds
    pivot = df_sample.pivot_table(
        index=["base_building_group", "sample_id"],
        columns="variant",
        values=f"{KPI}_mean",
        aggfunc="first",
    )

    labels = [v for v in VARIANT_ORDER if v in pivot.columns]
    pivot = pivot[labels]

    mat = pd.DataFrame(index=labels, columns=labels, dtype=float)
    n_pairs = pd.DataFrame(index=labels, columns=labels, dtype=float)

    for row_v in labels:
        for col_v in labels:
            numerator = pivot[row_v] - pivot[col_v]
            denominator = pivot[col_v]

            valid = denominator.notna() & numerator.notna() & (denominator != 0)
            pct_diffs = (numerator[valid] / denominator[valid]) * 100.0

            n_pairs.loc[row_v, col_v] = int(len(pct_diffs))

            if len(pct_diffs) == 0:
                mat.loc[row_v, col_v] = np.nan
            elif CELL_STAT == "mean_pct_diff":
                mat.loc[row_v, col_v] = float(pct_diffs.mean())
            else:
                mat.loc[row_v, col_v] = float(pct_diffs.median())

    return mat, n_pairs


def plot_pairwise_percent_heatmap(mat: pd.DataFrame) -> None:
    labels = list(mat.index)
    arr = mat.values.astype(float)

    vmax = np.nanmax(np.abs(arr))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    vmin = -vmax

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    im = ax.imshow(arr, vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)

    ax.set_xlabel("Compared to column variant")
    ax.set_ylabel("Row variant")

    title_stat = "Median" if CELL_STAT == "median_pct_diff" else "Mean"
    ax.set_title(f"Pairwise % differences in {KPI} ({title_stat}: row vs column)")

    # annotate values
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = arr[i, j]
            if np.isnan(value):
                text = "NA"
            else:
                text = f"{value:.1f}%"
            ax.text(j, i, text, ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Percent difference [%]")

    savefig(fig, OUTPUT_DIR / f"{KPI}_pairwise_percent_heatmap.png")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    df_runs = pd.read_csv(INPUT_CSV)

    required = {"building_id", "variant", "sample_id", "seed", KPI}
    missing = required.difference(df_runs.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    if "base_building_group" not in df_runs.columns:
        df_runs["base_building_group"] = df_runs["building_id"].astype(str).apply(base_building_group_key)

    df_runs = sort_variants(df_runs)
    df_sample = build_sample_summary(df_runs)

    sample_csv = OUTPUT_DIR / f"{KPI}_sample_summary.csv"
    df_sample.to_csv(sample_csv, index=False)

    pairwise_pct_mat, pairwise_counts = build_pairwise_percent_matrix(df_sample)

    mat_csv = OUTPUT_DIR / f"{KPI}_pairwise_percent_matrix.csv"
    counts_csv = OUTPUT_DIR / f"{KPI}_pairwise_percent_counts.csv"

    pairwise_pct_mat.to_csv(mat_csv)
    pairwise_counts.to_csv(counts_csv)

    plot_pairwise_percent_heatmap(pairwise_pct_mat)

    print("Saved:")
    print(sample_csv)
    print(mat_csv)
    print(counts_csv)
    print(OUTPUT_DIR / f"{KPI}_pairwise_percent_heatmap.png")


if __name__ == "__main__":
    main()