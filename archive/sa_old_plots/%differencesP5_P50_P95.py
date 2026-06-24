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

KPI = "annual_heating_kWh"
# KPI = "overheating_hours_meanTair_gt_26C"

VARIANT_ORDER = ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8"]

SHOW_PLOTS = True
DPI = 150
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


def build_variant_quantiles(df_sample: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, grp in df_sample.groupby("variant", observed=False):
        s = grp[f"{KPI}_mean"].dropna()
        if s.empty:
            continue
        rows.append(
            {
                "variant": variant,
                "P5": float(np.nanpercentile(s, 5)),
                "P50": float(np.nanpercentile(s, 50)),
                "P95": float(np.nanpercentile(s, 95)),
                "n_samples": int(s.notna().sum()),
            }
        )
    out = pd.DataFrame(rows)
    return sort_variants(out)


def build_pairwise_percent_matrix(df_quant: pd.DataFrame, quantile_col: str) -> pd.DataFrame:
    labels = [v for v in VARIANT_ORDER if v in df_quant["variant"].values]
    qmap = dict(zip(df_quant["variant"], df_quant[quantile_col]))

    mat = pd.DataFrame(index=labels, columns=labels, dtype=float)

    for row_v in labels:
        for col_v in labels:
            row_val = qmap.get(row_v, np.nan)
            col_val = qmap.get(col_v, np.nan)

            if pd.isna(row_val) or pd.isna(col_val) or col_val == 0:
                mat.loc[row_v, col_v] = np.nan
            else:
                mat.loc[row_v, col_v] = (row_val - col_val) / col_val * 100.0

    return mat


def plot_heatmap(mat: pd.DataFrame, quantile_label: str, output_path: Path) -> None:
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
    ax.set_title(f"{KPI}: pairwise % differences for {quantile_label}")

    for i in range(len(labels)):
        for j in range(len(labels)):
            value = arr[i, j]
            text = "NA" if np.isnan(value) else f"{value:.1f}%"
            ax.text(j, i, text, ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Percent difference [%]")

    savefig(fig, output_path)


def plot_combined_heatmaps(mats: dict[str, pd.DataFrame], output_path: Path) -> None:
    keys = ["P5", "P50", "P95"]
    available = [k for k in keys if k in mats]

    fig, axes = plt.subplots(1, len(available), figsize=(5.5 * len(available), 5.5), constrained_layout=True)
    if len(available) == 1:
        axes = [axes]

    vmax = 0.0
    for k in available:
        arr = mats[k].values.astype(float)
        local_vmax = np.nanmax(np.abs(arr))
        if np.isfinite(local_vmax):
            vmax = max(vmax, local_vmax)
    if vmax == 0:
        vmax = 1.0
    vmin = -vmax

    last_im = None
    for ax, k in zip(axes, available):
        mat = mats[k]
        labels = list(mat.index)
        arr = mat.values.astype(float)

        im = ax.imshow(arr, vmin=vmin, vmax=vmax, aspect="auto")
        last_im = im

        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels)

        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        ax.set_title(k)

        for i in range(len(labels)):
            for j in range(len(labels)):
                value = arr[i, j]
                text = "NA" if np.isnan(value) else f"{value:.1f}%"
                ax.text(j, i, text, ha="center", va="center", fontsize=7)

    fig.suptitle(f"{KPI}: pairwise % differences for P5 / P50 / P95")

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes, shrink=0.9)
        cbar.set_label("Percent difference [%]")

    savefig(fig, output_path)


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

    # sample-level means across seeds
    df_sample = build_sample_summary(df_runs)
    df_sample.to_csv(OUTPUT_DIR / f"{KPI}_sample_summary.csv", index=False)

    # variant-level quantiles
    df_quant = build_variant_quantiles(df_sample)
    df_quant.to_csv(OUTPUT_DIR / f"{KPI}_variant_quantiles.csv", index=False)

    mats = {}
    for qcol in ["P5", "P50", "P95"]:
        mat = build_pairwise_percent_matrix(df_quant, qcol)
        mats[qcol] = mat
        mat.to_csv(OUTPUT_DIR / f"{KPI}_pairwise_percent_matrix_{qcol}.csv")
        plot_heatmap(
            mat,
            quantile_label=qcol,
            output_path=OUTPUT_DIR / f"{KPI}_pairwise_percent_heatmap_{qcol}.png",
        )

    plot_combined_heatmaps(
        mats,
        OUTPUT_DIR / f"{KPI}_pairwise_percent_heatmaps_P5_P50_P95.png",
    )

    print("Saved:")
    print(OUTPUT_DIR / f"{KPI}_sample_summary.csv")
    print(OUTPUT_DIR / f"{KPI}_variant_quantiles.csv")
    for qcol in ["P5", "P50", "P95"]:
        print(OUTPUT_DIR / f"{KPI}_pairwise_percent_matrix_{qcol}.csv")
        print(OUTPUT_DIR / f"{KPI}_pairwise_percent_heatmap_{qcol}.png")
    print(OUTPUT_DIR / f"{KPI}_pairwise_percent_heatmaps_P5_P50_P95.png")


if __name__ == "__main__":
    main()