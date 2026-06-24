#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================
INPUT_CSV = Path(r"../results/sa_results/_analysis_plots_fixed/run_level_kpis.csv")
OUTPUT_DIR = Path(r"../results/sa_results/_analysis_plots_fixed/annual_heating_only")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

KPI = "annual_heating_kWh"
VARIANT_ORDER = ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8"]

# Choose "median" or "mean"
STAT = "median"

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


def build_variant_bar_values(df_sample: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, grp in df_sample.groupby("variant", observed=False):
        if grp.empty:
            continue

        s = grp[f"{KPI}_mean"].dropna()
        if s.empty:
            continue

        value = float(s.median()) if STAT == "median" else float(s.mean())

        rows.append(
            {
                "variant": variant,
                "value_kWh": value,
                "n_samples": int(s.notna().sum()),
            }
        )

    return sort_variants(pd.DataFrame(rows))


def plot_simple_bar_chart(df_variant: pd.DataFrame) -> None:
    if df_variant.empty:
        raise RuntimeError("No data available for plotting.")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(df_variant["variant"], df_variant["value_kWh"])

    for i, val in enumerate(df_variant["value_kWh"]):
        ax.text(i, val, f"{val:,.0f}", ha="center", va="bottom", fontsize=9)

    title_stat = "Median" if STAT == "median" else "Mean"
    ax.set_title(f"{title_stat} annual heating demand by variant")
    ax.set_xlabel("Variant")
    ax.set_ylabel("Annual heating demand [kWh]")
    add_grid(ax)

    out_name = f"bar_chart_annual_heating_{STAT}.png"
    savefig(fig, OUTPUT_DIR / out_name)


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

    # one value per variant
    df_variant = build_variant_bar_values(df_sample)
    df_variant.to_csv(OUTPUT_DIR / f"annual_heating_{STAT}_by_variant.csv", index=False)

    print(df_variant)
    plot_simple_bar_chart(df_variant)


if __name__ == "__main__":
    main()