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
    import matplotlib.cbook as cbook

    labels = []
    seed_std_data = []
    q_rows = []

    for v in VARIANT_ORDER:
        rows_v_sample = df_sample.loc[df_sample["variant"] == v]
        rows_v_quant = df_variant.loc[df_variant["variant"] == v]

        seed_stds = (
            rows_v_sample[f"{KPI}_std"].dropna().values
            if not rows_v_sample.empty else np.array([])
        )

        if rows_v_quant.empty:
            continue

        labels.append(v)
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
    # Panel A: custom grayscale P5 / P50 / P95 plot
    # --------------------------------------------------------
    ax = axes[0]

    box_width = 0.55
    cap_width = 0.18

    y_max_top = max(float(row["p95"]) for row in q_rows)
    offset_top = 0.012 * y_max_top

    for xi, row in zip(x, q_rows):
        p05 = float(row["p05"])
        p50 = float(row["p50"])
        p95 = float(row["p95"])

        # box from P5 to P95
        rect = plt.Rectangle(
            (xi - box_width / 2, p05),
            box_width,
            p95 - p05,
            fill=False,
            edgecolor="black",
            linewidth=1.2
        )
        ax.add_patch(rect)

        # caps only
        ax.hlines(
            y=p05,
            xmin=xi - cap_width / 2,
            xmax=xi + cap_width / 2,
            colors="black",
            linewidth=1.2
        )
        ax.hlines(
            y=p95,
            xmin=xi - cap_width / 2,
            xmax=xi + cap_width / 2,
            colors="black",
            linewidth=1.2
        )

        # median line
        ax.hlines(
            y=p50,
            xmin=xi - box_width / 2,
            xmax=xi + box_width / 2,
            colors="black",
            linewidth=1.5
        )

        # median marker in black
        ax.plot(xi, p50, marker="o", color="black", markersize=3)

        # labels
        ax.text(xi, p05 + offset_top, f"{p05:,.0f}", ha="center", va="bottom", fontsize=8)
        ax.text(xi, p50 + offset_top, f"{p50:,.0f}", ha="center", va="bottom", fontsize=8)
        ax.text(xi, p95 + offset_top, f"{p95:,.0f}", ha="center", va="bottom", fontsize=8)

    ax.set_title("Annual heating demand: P5 / P50 / P95 by variant")
    ax.set_ylabel("Annual heating demand [kWh]")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    add_grid(ax)

    # --------------------------------------------------------
    # Panel B: seed-std boxplot with median labels only
    # --------------------------------------------------------
    ax = axes[1]

    data = [arr for arr in seed_std_data if len(arr) > 0]
    data_labels = [lab for lab, arr in zip(labels, seed_std_data) if len(arr) > 0]

    stats = cbook.boxplot_stats(data, whis=1.5, labels=data_labels)

    ax.bxp(
        stats,
        showfliers=False,
        boxprops=dict(color="black"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        medianprops=dict(color="black"),
    )

    y_max_bottom = max(s["whishi"] for s in stats)
    offset_bottom = 0.012 * y_max_bottom

    for i, s in enumerate(stats, start=1):
        med = float(s["med"])
        ax.text(i, med + offset_bottom, f"{med:,.0f}", ha="center", va="bottom", fontsize=8)

    ax.set_title("Annual heating demand: usage stochasticity across seeds")
    ax.set_xlabel("Variant")
    ax.set_ylabel("Seed standard deviation [kWh]")
    add_grid(ax)

    savefig(fig, OUTPUT_DIR / "04_combined_annual_heating_figure_clean.png")

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
