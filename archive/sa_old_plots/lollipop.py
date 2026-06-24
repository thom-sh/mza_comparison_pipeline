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
REFERENCE_VARIANT = "V1"
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
    """
    One row = one sample for one variant, aggregated across seeds.
    This matches your existing annual-heating workflow.
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


def plot_sorted_median_annual_heating(df_variant: pd.DataFrame) -> None:
    """
    Sorted median annual heating demand by variant with P5-P95 interval.
    """
    if df_variant.empty:
        raise RuntimeError("No variant quantiles available for plotting.")

    df_plot = df_variant.sort_values("p50", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(df_plot["variant"], df_plot["p50"])

    lower_err = df_plot["p50"] - df_plot["p05"]
    upper_err = df_plot["p95"] - df_plot["p50"]

    ax.errorbar(
        x=df_plot["variant"],
        y=df_plot["p50"],
        yerr=[lower_err, upper_err],
        fmt="none",
        capsize=4,
        elinewidth=1,
    )

    for i, val in enumerate(df_plot["p50"]):
        ax.text(i, val, f"{val:,.0f}", ha="center", va="bottom", fontsize=9)

    ax.set_title("Median annual heating demand by variant")
    ax.set_xlabel("Variant")
    ax.set_ylabel("Annual heating demand [kWh]")
    add_grid(ax)

    savefig(fig, OUTPUT_DIR / "05_sorted_median_annual_heating.png")


def plot_lollipop_relative_to_reference(df_variant: pd.DataFrame, reference_variant: str) -> None:
    """
    Ranked lollipop chart of median annual heating demand difference
    relative to a reference variant.
    """
    if df_variant.empty:
        raise RuntimeError("No variant quantiles available for plotting.")

    ref_rows = df_variant.loc[df_variant["variant"] == reference_variant]
    if ref_rows.empty:
        raise RuntimeError(f"Reference variant {reference_variant} not found.")

    ref_value = float(ref_rows.iloc[0]["p50"])

    df_plot = df_variant.copy()
    df_plot["delta_kWh"] = df_plot["p50"] - ref_value
    df_plot = df_plot[df_plot["variant"] != reference_variant].copy()
    df_plot = df_plot.sort_values("delta_kWh", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9, 5))

    y_pos = np.arange(len(df_plot))
    ax.hlines(y=y_pos, xmin=0, xmax=df_plot["delta_kWh"], linewidth=2)
    ax.plot(df_plot["delta_kWh"], y_pos, "o")
    ax.axvline(0, linewidth=1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot["variant"])

    dx = 0.01 * max(1.0, float(np.nanmax(np.abs(df_plot["delta_kWh"].values))))
    for y, val in zip(y_pos, df_plot["delta_kWh"]):
        ax.text(val + dx, y, f"{val:+,.0f}", va="center", fontsize=9)

    ax.set_title(f"Median annual heating demand difference relative to {reference_variant}")
    ax.set_xlabel("Difference to reference [kWh]")
    ax.set_ylabel("Variant")
    add_grid(ax)

    savefig(fig, OUTPUT_DIR / f"06_lollipop_delta_vs_{reference_variant}.png")


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

    # 1) aggregate run-level results to sample-level means across seeds
    df_sample = build_sample_summary(df_runs)
    df_sample.to_csv(OUTPUT_DIR / "annual_heating_sample_summary.csv", index=False)

    # 2) compute variant-level quantiles across samples
    df_variant = build_variant_quantiles(df_sample)
    df_variant.to_csv(OUTPUT_DIR / "annual_heating_variant_quantiles.csv", index=False)

    # 3) plots
    plot_sorted_median_annual_heating(df_variant)
    plot_lollipop_relative_to_reference(df_variant, REFERENCE_VARIANT)

    print("Saved:")
    print(OUTPUT_DIR / "annual_heating_sample_summary.csv")
    print(OUTPUT_DIR / "annual_heating_variant_quantiles.csv")
    print(OUTPUT_DIR / "05_sorted_median_annual_heating.png")
    print(OUTPUT_DIR / f"06_lollipop_delta_vs_{REFERENCE_VARIANT}.png")


if __name__ == "__main__":
    main()