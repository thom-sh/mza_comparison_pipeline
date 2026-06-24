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
    # One row = one sample for one variant, aggregated across seeds
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


def plot_combined_figure(df_sample: pd.DataFrame, df_variant: pd.DataFrame) -> None:
    # Keep only variants that actually exist in the data
    labels = []
    sample_mean_data = []
    seed_std_data = []
    q_rows = []

    for v in VARIANT_ORDER:
        rows_v_sample = df_sample.loc[df_sample["variant"] == v]
        rows_v_quant = df_variant.loc[df_variant["variant"] == v]

        sample_means = rows_v_sample[f"{KPI}_mean"].dropna().values if not rows_v_sample.empty else np.array([])
        seed_stds = rows_v_sample[f"{KPI}_std"].dropna().values if not rows_v_sample.empty else np.array([])

        if len(sample_means) == 0 or rows_v_quant.empty:
            continue

        labels.append(v)
        sample_mean_data.append(sample_means)
        seed_std_data.append(seed_stds)
        q_rows.append(rows_v_quant.iloc[0])

    x = np.arange(1, len(labels) + 1)

    fig, axes = plt.subplots(
        2, 1,
        figsize=(11, 9),
        sharex=True,
        constrained_layout=True
    )

    # --------------------------------------------------------
    # Panel A: boxplot of sample means + P5/P50/P95 overlay
    # --------------------------------------------------------
    ax = axes[0]
    ax.boxplot(sample_mean_data, positions=x, widths=0.55, showfliers=False)

    p05_vals = [row["p05"] for row in q_rows]
    p50_vals = [row["p50"] for row in q_rows]
    p95_vals = [row["p95"] for row in q_rows]

    # Connect quantiles across variants for readability
    ax.plot(x, p50_vals, marker="o", linestyle="-", linewidth=1.5, label="P50")
    ax.plot(x, p05_vals, marker="_", linestyle="None", markersize=14, markeredgewidth=2, label="P5")
    ax.plot(x, p95_vals, marker="_", linestyle="None", markersize=14, markeredgewidth=2, label="P95")

    ax.set_title("Annual heating demand: variants and input-uncertainty band")
    ax.set_ylabel("Annual heating demand [kWh]")
    ax.legend()
    add_grid(ax)

    # --------------------------------------------------------
    # Panel B: seed-std boxplot
    # --------------------------------------------------------
    ax = axes[1]
    ax.boxplot(seed_std_data, positions=x, widths=0.55, showfliers=False)
    ax.set_title("Annual heating demand: usage stochasticity across seeds")
    ax.set_xlabel("Variant")
    ax.set_ylabel("Seed standard deviation [kWh]")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    add_grid(ax)

    savefig(fig, OUTPUT_DIR / "04_combined_annual_heating_figure.png")


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
    df_sample = build_sample_summary(df_runs)
    df_variant = build_variant_quantiles(df_sample)

    df_sample.to_csv(OUTPUT_DIR / "annual_heating_sample_summary.csv", index=False)
    df_variant.to_csv(OUTPUT_DIR / "annual_heating_variant_quantiles.csv", index=False)

    plot_combined_figure(df_sample, df_variant)

    print("Saved:")
    print(OUTPUT_DIR / "annual_heating_sample_summary.csv")
    print(OUTPUT_DIR / "annual_heating_variant_quantiles.csv")
    print(OUTPUT_DIR / "04_combined_annual_heating_figure.png")


if __name__ == "__main__":
    main()
