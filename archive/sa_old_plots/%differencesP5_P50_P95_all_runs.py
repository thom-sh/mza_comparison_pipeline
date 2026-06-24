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
INPUT_CSV = (SCRIPT_DIR / "../new_plots/sa_results/run_level_kpis.csv").resolve()
OUTPUT_DIR = (SCRIPT_DIR / "../new_plots/output/matrices").resolve()

KPI = "annual_heating_kWh"
# KPI  = "peak_heating_kW"
# KPI = "overheating_hours_any_zone_gt_26C"

KPI_LABELS = {
    "annual_heating_kWh": "Annual heating demand",
    "peak_heating_kW": "Peak heating demand",
    "overheating_hours_any_zone_gt_26C": "Overheating hours",
    "overheating_hours_meanTair_gt_26C": "Overheating hours",
}

VARIANT_ORDER = ["V2", "V3", "V5", "V7"]

SHOW_PLOTS = True
DPI = 150

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 8,
    "axes.titleweight": "normal",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

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



def build_run_summary(df_runs: pd.DataFrame) -> pd.DataFrame:
    """
    Keep one row = one simulation run.

    Requested behavior:
    each variant uses all runs directly, i.e.
        n_runs(variant) = n_samples * n_seeds * n_buildings (if multiple buildings exist)

    No aggregation across seeds or samples is performed here.
    """
    cols = [c for c in [
        "base_building_group",
        "building_id",
        "variant",
        "sample_id",
        "seed",
        KPI,
    ] if c in df_runs.columns]

    return sort_variants(df_runs[cols].copy())



def build_variant_quantiles_from_runs(df_runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, grp in df_runs.groupby("variant", observed=False):
        s = pd.to_numeric(grp[KPI], errors="coerce").dropna()
        if s.empty:
            continue

        row = {
            "variant": variant,
            "P5": float(np.nanpercentile(s, 5)),
            "P50": float(np.nanpercentile(s, 50)),
            "P95": float(np.nanpercentile(s, 95)),
            "n_runs": int(s.notna().sum()),
        }

        if "sample_id" in grp.columns:
            row["n_unique_samples"] = int(grp["sample_id"].nunique(dropna=True))
        if "seed" in grp.columns:
            row["n_unique_seeds"] = int(grp["seed"].nunique(dropna=True))
        if "building_id" in grp.columns:
            row["n_unique_buildings"] = int(grp["building_id"].nunique(dropna=True))

        rows.append(row)

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



def plot_heatmap(mat: pd.DataFrame, quantile_label: str, output_path: Path, KPI_LABEL) -> None:
    labels = list(mat.index)
    arr = mat.values.astype(float)

    vmax = np.nanmax(np.abs(arr))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    vmin = -vmax

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    im = ax.imshow(arr, cmap="RdPu", vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            value = arr[i, j]
            text = "NA" if np.isnan(value) else f"{value:.1f}%"
            ax.text(j, i, text, ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"Difference in {KPI_LABEL} [%]")
    savefig(fig, output_path)



def plot_combined_heatmaps(mats: dict[str, pd.DataFrame], output_path: Path, KPI_LABEL) -> None:
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

        im = ax.imshow(arr, cmap="RdPu", vmin=vmin, vmax=vmax, aspect="auto")
        last_im = im

        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels)

        ax.set_title(k)

        for i in range(len(labels)):
            for j in range(len(labels)):
                value = arr[i, j]
                text = "NA" if np.isnan(value) else f"{value:.1f}%"
                ax.text(j, i, text, ha="center", va="center", fontsize=8)

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes, shrink=0.9)
        cbar.set_label(f"Difference in {KPI_LABEL} [%]")

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

    # one row = one run (sample x seed x building x variant)
    df_run_summary = build_run_summary(df_runs)
    df_run_summary.to_csv(OUTPUT_DIR / f"{KPI}_run_summary_all_runs.csv", index=False)

    # variant quantiles are computed directly from all runs
    df_quant = build_variant_quantiles_from_runs(df_runs)
    df_quant.to_csv(OUTPUT_DIR / f"{KPI}_variant_quantiles_all_runs.csv", index=False)

    mats = {}
    for qcol in ["P5", "P50", "P95"]:
        mat = build_pairwise_percent_matrix(df_quant, qcol)
        mats[qcol] = mat
        mat.to_csv(OUTPUT_DIR / f"{KPI}_pairwise_percent_matrix_{qcol}_all_runs.csv")
        plot_heatmap(
            mat,
            quantile_label=qcol,
            output_path=OUTPUT_DIR / f"{KPI}_pairwise_percent_heatmap_{qcol}_all_runs.pdf",
            KPI_LABEL=KPI_LABELS.get(KPI, KPI),
        )

    plot_combined_heatmaps(
        mats,
        OUTPUT_DIR / f"{KPI}_pairwise_percent_heatmaps_P5_P50_P95_all_runs.pdf",
        KPI_LABEL=KPI_LABELS.get(KPI, KPI),
    )

    print("Using all runs directly per variant (no seed aggregation).")
    print("Saved:")
    print(OUTPUT_DIR / f"{KPI}_run_summary_all_runs.csv")
    print(OUTPUT_DIR / f"{KPI}_variant_quantiles_all_runs.csv")
    for qcol in ["P5", "P50", "P95"]:
        print(OUTPUT_DIR / f"{KPI}_pairwise_percent_matrix_{qcol}_all_runs.csv")
        print(OUTPUT_DIR / f"{KPI}_pairwise_percent_heatmap_{qcol}_all_runs.pdf")
    print(OUTPUT_DIR / f"{KPI}_pairwise_percent_heatmaps_P5_P50_P95_all_runs.pdf")


if __name__ == "__main__":
    main()
