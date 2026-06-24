#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================
INPUT_CSV = Path(r"../results/sa_results/_analysis_plots_fixed/run_level_kpis.csv")
OUTPUT_DIR = Path(r"../results/sa_results/_analysis_plots_fixed/annual_heating_only")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

KPI = "annual_heating_kWh"
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


def add_grid(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", alpha=0.3)


def savefig(fig: plt.Figure, path: Path) -> None:
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


def build_sample_summary(df_runs: pd.DataFrame) -> pd.DataFrame:
    """
    One row = one sample for one variant, aggregated across seeds.
    """
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
        if grp.empty:
            continue

        s = grp[f"{KPI}_mean"].dropna()
        if s.empty:
            continue

        q05 = p5(s)
        q50 = p50(s)
        q95 = p95(s)

        rows.append(
            {
                "variant": variant,
                "p05": q05,
                "p50": q50,
                "p95": q95,
                "seg_1_p05": q05,
                "seg_2_p50_minus_p05": q50 - q05,
                "seg_3_p95_minus_p50": q95 - q50,
                "n_samples": int(s.notna().sum()),
            }
        )

    return sort_variants(pd.DataFrame(rows))


def plot_stacked_quantile_bar(df_variant: pd.DataFrame) -> None:
    if df_variant.empty:
        raise RuntimeError("No quantile data available for plotting.")

    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(df_variant))

    seg1 = df_variant["seg_1_p05"].values
    seg2 = df_variant["seg_2_p50_minus_p05"].values
    seg3 = df_variant["seg_3_p95_minus_p50"].values

    ax.bar(x, seg1, label="P5")
    ax.bar(x, seg2, bottom=seg1, label="P50 - P5")
    ax.bar(x, seg3, bottom=seg1 + seg2, label="P95 - P50")

    # annotate P5, P50, P95 values
    for i, row in df_variant.reset_index(drop=True).iterrows():
        ax.text(i, row["p05"], f"{row['p05']:,.0f}", ha="center", va="bottom", fontsize=8)
        ax.text(i, row["p50"], f"{row['p50']:,.0f}", ha="center", va="bottom", fontsize=8)
        ax.text(i, row["p95"], f"{row['p95']:,.0f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(df_variant["variant"])
    ax.set_title("Annual heating demand quantiles by variant")
    ax.set_xlabel("Variant")
    ax.set_ylabel("Annual heating demand [kWh]")
    ax.legend()
    add_grid(ax)

    savefig(fig, OUTPUT_DIR / "bar_chart_stacked_quantiles_p5_p50_p95.png")


def main() -> None:
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

    # aggregate across seeds first
    df_sample = build_sample_summary(df_runs)

    # compute variant-level quantiles
    df_variant = build_variant_quantiles(df_sample)
    df_variant.to_csv(OUTPUT_DIR / "annual_heating_variant_quantiles.csv", index=False)

    print(df_variant[["variant", "p05", "p50", "p95", "n_samples"]])

    plot_stacked_quantile_bar(df_variant)


if __name__ == "__main__":
    main()