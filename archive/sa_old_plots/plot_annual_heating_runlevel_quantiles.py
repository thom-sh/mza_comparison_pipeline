#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================
INPUT_CSV = Path(r"../new_plots/sa_results/run_level_kpis.csv")
OUTPUT_DIR = Path(r"../new_plots/output/annual_heating_runlevel_quantiles")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

KPI = "annual_heating_kWh"
VARIANT_ORDER = ["V1", "V1_20HH", "V2", "V2_20HH", "V3", "V4", "V5", "V6", "V7", "V8"]

SHOW_PLOTS = True
DPI = 300
# ============================================================


def add_grid(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", alpha=0.30)
    ax.set_axisbelow(True)


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



def build_variant_quantiles_from_runs(df_runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []

    id_cols = [c for c in ["building_id", "weather_key", "sample_id", "seed"] if c in df_runs.columns]

    for variant, grp in df_runs.groupby("variant", observed=False):
        if grp.empty:
            continue

        s = pd.to_numeric(grp[KPI], errors="coerce").dropna()
        if s.empty:
            continue

        if id_cols:
            n_runs = int(grp[id_cols].drop_duplicates().shape[0])
        else:
            n_runs = int(s.shape[0])

        rows.append(
            {
                "variant": variant,
                "p05": p5(s),
                "p50": p50(s),
                "p95": p95(s),
                "mean": float(s.mean()),
                "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
                "n_runs": n_runs,
            }
        )

    return sort_variants(pd.DataFrame(rows))



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

    for xi, row in zip(x, df_variant.itertuples(index=False)):
        ax.text(xi, row.p50, f"{row.p50:,.0f}", ha="center", va="bottom", fontsize=9)
        ax.text(xi, row.p05, f"{row.p05:,.0f}", ha="center", va="top", fontsize=8)
        ax.text(xi, row.p95, f"{row.p95:,.0f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Annual heating demand: P5 / P50 / P95 across all runs")
    ax.set_xlabel("Variant")
    ax.set_ylabel("Annual heating demand [kWh]")
    add_grid(ax)

    savefig(fig, OUTPUT_DIR / "01_quantile_interval_annual_heating_runlevel.pdf")



def plot_boxplot_runs(df_runs: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    data = [
        pd.to_numeric(df_runs.loc[df_runs["variant"] == v, KPI], errors="coerce").dropna().values
        for v in VARIANT_ORDER
    ]
    labels = [v for v, arr in zip(VARIANT_ORDER, data) if len(arr) > 0]
    data = [arr for arr in data if len(arr) > 0]

    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_title("Annual heating demand: raw run distribution")
    ax.set_xlabel("Variant")
    ax.set_ylabel("Annual heating demand [kWh]")
    add_grid(ax)

    savefig(fig, OUTPUT_DIR / "02_boxplot_annual_heating_runs.pdf")



def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    df_runs = pd.read_csv(INPUT_CSV)

    required = {"variant", KPI}
    missing = required.difference(df_runs.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    df_runs = sort_variants(df_runs)

    # IMPORTANT:
    # Each row in run_level_kpis.csv is treated as one run.
    # Therefore sample_id * seed combinations remain separate and are NOT
    # aggregated to sample means before computing P5 / P50 / P95.
    df_variant = build_variant_quantiles_from_runs(df_runs)
    df_variant.to_csv(OUTPUT_DIR / "annual_heating_variant_quantiles_runlevel.csv", index=False)

    plot_quantile_interval(df_variant)
    plot_boxplot_runs(df_runs)

    print("Saved:")
    print(OUTPUT_DIR / "annual_heating_variant_quantiles_runlevel.csv")
    print(OUTPUT_DIR / "01_quantile_interval_annual_heating_runlevel.pdf")
    print(OUTPUT_DIR / "02_boxplot_annual_heating_runs.pdf")


if __name__ == "__main__":
    main()
