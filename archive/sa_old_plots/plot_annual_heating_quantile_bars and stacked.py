#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================
INPUT_CSV = Path(r"../new_plots/output/annual_heating_runlevel_quantiles/annual_heating_variant_quantiles_runlevel.csv")
OUTPUT_DIR = Path(r"../new_plots/output/annual_heating_runlevel_quantiles")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VARIANT_ORDER = ["V1", "V1_20HH", "V2", "V2_20HH", "V3", "V4", "V5", "V6", "V7", "V8"]
SHOW_PLOTS = True
DPI = 300

# Same quantile colors across all variants
COLOR_P05 = "#c9d6ea"
COLOR_P50 = "#7ea6d8"
COLOR_P95 = "#2f5f98"
LEGEND_EDGE_COLOR = "#bdc1c5"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 8,
    "axes.titleweight": "normal",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})
# ============================================================


def sort_variants(df: pd.DataFrame, col: str = "variant") -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.Categorical(out[col], categories=VARIANT_ORDER, ordered=True)
    return out.sort_values(col).reset_index(drop=True)


def add_grid(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", alpha=0.30)
    ax.set_axisbelow(True)


def savefig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def format_value_labels(ax: plt.Axes, bars, values: np.ndarray) -> None:
    ymax = ax.get_ylim()[1]
    offset = 0.012 * ymax
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            f"{value:,.0f}",
            ha="center",
            va="bottom",
            rotation=0,
            fontsize=8,
        )


def plot_quantile_bars(df: pd.DataFrame) -> None:
    labels = df["variant"].tolist()
    x = np.arange(len(labels), dtype=float)
    width = 0.24

    # p05 = df["p05"].to_numpy(dtype=float) / 1000  # Convert to MWh
    # p50 = df["p50"].to_numpy(dtype=float) / 1000  # Convert to MWh
    # p95 = df["p95"].to_numpy(dtype=float) / 1000  # Convert to MWh

    # for kWH values (without conversion):
    p05 = df["p05"].to_numpy(dtype=float) 
    p50 = df["p50"].to_numpy(dtype=float) 
    p95 = df["p95"].to_numpy(dtype=float) 


    fig, ax = plt.subplots(figsize=(10.5, 5.4))

    bars_p05 = ax.bar(x - width, p05, width=width, label="P5", color=COLOR_P05)
    bars_p50 = ax.bar(x,         p50, width=width, label="P50", color=COLOR_P50)
    bars_p95 = ax.bar(x + width, p95, width=width, label="P95", color=COLOR_P95)

    ax.set_ylabel("Annual heating demand [kWh]")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    add_grid(ax)

    legend = ax.legend(frameon=True)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor(LEGEND_EDGE_COLOR)
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_alpha(1.0)

    format_value_labels(ax, bars_p05, p05)
    format_value_labels(ax, bars_p50, p50)
    format_value_labels(ax, bars_p95, p95)

    savefig(fig, OUTPUT_DIR / "03_annual_heating_quantile_bars_kWh.pdf")

# for stacked bars:

def plot_quantile_bars_stacked(df: pd.DataFrame) -> None:
    labels = df["variant"].tolist()
    x = np.arange(len(labels), dtype=float)
    width = 0.55

    p05 = df["p05"].to_numpy(dtype=float) 
    p50 = df["p50"].to_numpy(dtype=float) 
    p95 = df["p95"].to_numpy(dtype=float) 

    # Stacked segments
    seg1 = p05
    seg2 = p50 - p05
    seg3 = p95 - p50

    fig, ax = plt.subplots(figsize=(10.5, 5.4))

    bars1 = ax.bar(x, seg1, width=width, label="P5", color=COLOR_P05)
    bars2 = ax.bar(x, seg2, width=width, bottom=seg1, label="P50 - P5", color=COLOR_P50)
    bars3 = ax.bar(x, seg3, width=width, bottom=seg1 + seg2, label="P95 - P50", color=COLOR_P95)

    ax.set_ylabel("Annual heating demand [kWh]")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    add_grid(ax)

    legend = ax.legend(frameon=True)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor(LEGEND_EDGE_COLOR)
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_alpha(1.0)

    # Optional: show P95 at top
    ymax = ax.get_ylim()[1]
    offset = 0.012 * ymax
    for xi, v05, v50, v95 in zip(x, p05, p50, p95):
        ax.text(xi, v05 + offset, f"{v05:,.1f}", ha="center", va="bottom", fontsize=8)
        ax.text(xi, v50 + offset, f"{v50:,.1f}", ha="center", va="bottom", fontsize=8)
        ax.text(xi, v95 + offset, f"{v95:,.1f}", ha="center", va="bottom", fontsize=8)

    savefig(fig, OUTPUT_DIR / "03_annual_heating_quantile_bars_stacked_kWh.pdf")


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    required = {"variant", "p05", "p50", "p95"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    df = sort_variants(df)
    df = df[df["variant"].notna()].copy()

    plot_quantile_bars(df)
    plot_quantile_bars_stacked(df)

    print("Saved:")
    print(OUTPUT_DIR / "03_annual_heating_quantile_bars.pdf")
    print(OUTPUT_DIR / "03_annual_heating_quantile_bars_stacked.pdf")


if __name__ == "__main__":
    main()
