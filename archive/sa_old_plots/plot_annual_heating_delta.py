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
REFERENCE_VARIANT = "V3"
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
    ax.grid(True, alpha=0.3)


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


def plot_delta_vs_reference(df_sample: pd.DataFrame) -> None:
    ref = df_sample.loc[
        df_sample["variant"] == REFERENCE_VARIANT,
        ["base_building_group", "sample_id", f"{KPI}_mean"],
    ].rename(columns={f"{KPI}_mean": f"{KPI}_ref"})

    if ref.empty:
        raise RuntimeError(f"No reference rows found for {REFERENCE_VARIANT}")

    df = df_sample.merge(
        ref,
        on=["base_building_group", "sample_id"],
        how="left",
        validate="many_to_one",
    )

    delta_col = f"{KPI}_delta_vs_{REFERENCE_VARIANT}"
    df[delta_col] = df[f"{KPI}_mean"] - df[f"{KPI}_ref"]

    variants_to_plot = [v for v in VARIANT_ORDER if v != REFERENCE_VARIANT]

    data = []
    labels = []
    for v in variants_to_plot:
        arr = df.loc[df["variant"] == v, delta_col].dropna().values
        if len(arr) > 0:
            data.append(arr)
            labels.append(v)

    if not data:
        raise RuntimeError("Delta plot is empty. Check sample alignment and reference variant.")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.axhline(0.0, linestyle="--", color="black", linewidth=1)
    ax.set_title(f"Annual heating demand: delta vs {REFERENCE_VARIANT}")
    ax.set_xlabel("Variant")
    ax.set_ylabel("Δ Annual heating demand [kWh]")
    add_grid(ax)

    savefig(fig, OUTPUT_DIR / f"05_delta_vs_{REFERENCE_VARIANT}_annual_heating.png")

    # optional export of the delta values
    export_cols = ["base_building_group", "variant", "sample_id", f"{KPI}_mean", f"{KPI}_ref", delta_col]
    df[export_cols].to_csv(
        OUTPUT_DIR / f"annual_heating_delta_vs_{REFERENCE_VARIANT}.csv",
        index=False
    )


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

    plot_delta_vs_reference(df_sample)

    print("Saved:")
    print(OUTPUT_DIR / f"05_delta_vs_{REFERENCE_VARIANT}_annual_heating.png")
    print(OUTPUT_DIR / f"annual_heating_delta_vs_{REFERENCE_VARIANT}.csv")


if __name__ == "__main__":
    main()