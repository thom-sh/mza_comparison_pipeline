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
            rotation=90,
            fontsize=8,
        )


def plot_quantile_bars(df: pd.DataFrame) -> None:
    labels = df["variant"].tolist()
    x = np.arange(len(labels), dtype=float)
    width = 0.58

    # Convert to MWh
    p05 = df["p05"].to_numpy(dtype=float) / 1000.0
    p50 = df["p50"].to_numpy(dtype=float) / 1000.0
    p95 = df["p95"].to_numpy(dtype=float) / 1000.0

    yerr_low = p50 - p05
    yerr_high = p95 - p50

    fig, ax = plt.subplots(figsize=(10.5, 5.4))

    bars = ax.bar(
        x,
        p50,
        width=width,
        color=COLOR_P50,
        edgecolor="none",
        label="P50",
        zorder=3,
    )

    ax.errorbar(
        x,
        p50,
        yerr=[yerr_low, yerr_high],
        fmt="none",
        ecolor="black",
        elinewidth=1.2,
        capsize=5,
        capthick=1.2,
        zorder=4,
        label="P5–P95 range",
    )

    ax.set_title("Annual heating demand by variant (P50 with P5/P95 error bars)")
    ax.set_xlabel("Variant")
    ax.set_ylabel("Annual heating demand [MWh]")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    add_grid(ax)

    legend = ax.legend(frameon=True)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor(LEGEND_EDGE_COLOR)
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_alpha(1.0)

    # Value labels for P50 only
    ymax = ax.get_ylim()[1]
    offset = 0.012 * ymax
    for bar, value in zip(bars, p50):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            f"{value:,.1f}",
            ha="left",
            va="bottom",
            rotation=0,
            fontsize=8,
        )

    savefig(fig, OUTPUT_DIR / "03_annual_heating_p50_with_p5_p95_errorbars.pdf")

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

    print("Saved:")
    print(OUTPUT_DIR / "03_annual_heating_p50_with error_p5_p95.pdf")


if __name__ == "__main__":
    main()
