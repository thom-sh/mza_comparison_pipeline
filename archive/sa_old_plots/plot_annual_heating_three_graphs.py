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
# Use the run-level KPI file created by sa_extract_and_plot_fixed.py
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


def add_grid(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.3)


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


def sort_variants(df: pd.DataFrame, col: str = "variant") -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.Categorical(out[col], categories=VARIANT_ORDER, ordered=True)
    return out.sort_values(col)


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
        if grp.empty:
            continue
        s = grp[f"{KPI}_mean"].dropna()
        if s.empty:
            continue
        rows.append(
            {
                "variant": variant,
                "p05": p5(s),
                "p50": p50(s),
                "p95": p95(s),
                "n_samples": int(s.notna().sum()),
            }
        )
    return sort_variants(pd.DataFrame(rows))


def plot_boxplot_sample_means(df_sample: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    data = [
        df_sample.loc[df_sample["variant"] == v, f"{KPI}_mean"].dropna().values
        for v in VARIANT_ORDER
    ]
    labels = [v for v, arr in zip(VARIANT_ORDER, data) if len(arr) > 0]
    data = [arr for arr in data if len(arr) > 0]

    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_title("Annual heating demand: boxplot of sample means")
    ax.set_xlabel("Variant")
    ax.set_ylabel("Annual heating demand [kWh]")
    add_grid(ax)

    savefig(fig, OUTPUT_DIR / "01_boxplot_annual_heating_sample_means.png")


def plot_quantile_interval(df_variant: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    labels, x, y, yerr_low, yerr_high = [], [], [], [], []
    for v in VARIANT_ORDER:
        rows = df_variant.loc[df_variant["variant"] == v]
        if rows.empty:
            continue
        row = rows.iloc[0]
        labels.append(v)
        x.append(len(labels) - 1)
        y.append(row["p50"])
        yerr_low.append(row["p50"] - row["p05"])
        yerr_high.append(row["p95"] - row["p50"])

    ax.errorbar(x, y, yerr=[yerr_low, yerr_high], fmt="o", capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Annual heating demand: P5 / P50 / P95 across samples")
    ax.set_xlabel("Variant")
    ax.set_ylabel("Annual heating demand [kWh]")
    add_grid(ax)

    savefig(fig, OUTPUT_DIR / "02_quantile_interval_annual_heating.png")


def plot_seed_std_boxplot(df_sample: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    data = [
        df_sample.loc[df_sample["variant"] == v, f"{KPI}_std"].dropna().values
        for v in VARIANT_ORDER
    ]
    labels = [v for v, arr in zip(VARIANT_ORDER, data) if len(arr) > 0]
    data = [arr for arr in data if len(arr) > 0]

    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_title("Annual heating demand: variability across seeds")
    ax.set_xlabel("Variant")
    ax.set_ylabel("Seed standard deviation [kWh]")
    add_grid(ax)

    savefig(fig, OUTPUT_DIR / "03_seed_std_boxplot_annual_heating.png")


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

    # 1) aggregate raw runs -> sample level
    df_sample = build_sample_summary(df_runs)
    df_sample.to_csv(OUTPUT_DIR / "annual_heating_sample_summary.csv", index=False)

    # 2) aggregate sample level -> variant quantiles
    df_variant = build_variant_quantiles(df_sample)
    df_variant.to_csv(OUTPUT_DIR / "annual_heating_variant_quantiles.csv", index=False)

    # 3) plots
    plot_boxplot_sample_means(df_sample)
    plot_quantile_interval(df_variant)
    plot_seed_std_boxplot(df_sample)

    print("Saved:")
    print(OUTPUT_DIR / "annual_heating_sample_summary.csv")
    print(OUTPUT_DIR / "annual_heating_variant_quantiles.csv")
    print(OUTPUT_DIR / "01_boxplot_annual_heating_sample_means.png")
    print(OUTPUT_DIR / "02_quantile_interval_annual_heating.png")
    print(OUTPUT_DIR / "03_seed_std_boxplot_annual_heating.png")


if __name__ == "__main__":
    main()
