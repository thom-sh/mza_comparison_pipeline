# -*- coding: utf-8 -*-
"""
Plot thesis-style single-panel zoning-variant KPI figures from run_level_kpis.csv.

Creates separate full-text-width figures for each KPI and construction state:
  1) Annual heating demand [MWh]
  2) Peak heating load [kW]
  3) Overheating hours, any zone > 26 °C [h]
  4) Maximum interzonal temperature spread [K]

Figure logic:
  - one figure = one KPI + one construction state
  - x-axis: zoning variants labelled V1, V2, ..., V8
  - grouped bars: Napoli and Munich
  - bar height: P50 / median
  - error bars: P5--P95 range from all remaining propagated uncertainty

Place this script next to run_level_kpis.csv and run it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# ===============================================================
# CONFIG
# ===============================================================

PROJECT_DIR = Path(__file__).resolve().parent

INPUT_CSV = PROJECT_DIR / "sa_results" / "sa_main" / "run_level_kpis.csv"

OUTPUT_DIR = (
    PROJECT_DIR
    / "output"
    / "figures_kpi_variant_thesis_style"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Main thesis variants. Set True only if you want to include the additional 20HH variants.
INCLUDE_20HH = False

DPI = 300

# Full text-width figure. Height kept moderate for thesis layout.
FIGSIZE = (6.4, 2.9)

# If True, standard and retrofit plots for the same KPI use the same y-axis limit.
# This improves direct comparability. Set False if you want each figure to use its own y-scale.
SHARE_Y_LIMIT_ACROSS_STATES = True

# If True, each single-panel plot receives a small title: Standard or Retrofit.
# Set False if you prefer to describe this only in the LaTeX caption.
SHOW_STATE_TITLE = False

# Thesis colours: two teal shades + black text/axes/error bars.
COLORS = {
    "napoli": "#A9D1CE",   # light muted teal
    "munich": "#3F7D7A",   # dark muted teal
    "error": "#000000",    # black P5--P95 error bars
    "edge": "#000000",     # grey bar edges
    "axisedge": "#777D84", # black axes
    "grid": "#D9D9D9",     # light grey gridlines
    "text": "#000000",     # black text
}

WEATHER_LABELS = {
    "TRY_A": "Napoli",
    "TRY_B": "Munich",
    "Napoli": "Napoli",
    "Munich": "Munich",
}

WEATHER_COLORS = {
    "Napoli": COLORS["napoli"],
    "Munich": COLORS["munich"],
}

RETROFIT_ORDER = ["standard", "retrofit"]
RETROFIT_LABELS = {
    "standard": "Standard",
    "retrofit": "Retrofit",
}

VARIANT_ORDER_MAIN = ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8"]
VARIANT_ORDER_WITH_20HH = [
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8"
]

# Keep x-axis labels simple and consistent with the thesis variant table.
VARIANT_LABELS = {
    "V1": "V1",
    "V2": "V2",
    "V3": "V3",
    "V4": "V4",
    "V5": "V5",
    "V6": "V6",
    "V7": "V7",
    "V8": "V8",
    "V1_20HH": "V1$_{20H}$",
    "V2_20HH": "V2$_{20H}$",
}

KPI_CONFIGS = [
    {
        "name": "annual_heating_demand",
        "column": "annual_heating_kWh",
        "ylabel": "Annual heating demand [MWh]",
        "transform": lambda s: s / 1000.0,
        "filename": "kpi_annual_heating_demand_by_variant",
    },
    {
        "name": "peak_heating_load",
        "column": "peak_heating_kW",
        "ylabel": "Peak heating load [kW]",
        "transform": lambda s: s,
        "filename": "kpi_peak_heating_load_by_variant",
    },
    {
        "name": "overheating_hours",
        "column": "overheating_hours_any_zone_gt_26C",
        "ylabel": "Overheating hours [h]",
        "transform": lambda s: s,
        "filename": "kpi_overheating_hours_by_variant",
    },
    {
        "name": "max_interzonal_spread",
        "column": "max_interzone_spread_C",
        "ylabel": "Maximum interzonal spread [K]",
        "transform": lambda s: s,
        "filename": "kpi_max_interzonal_spread_by_variant",
    },
    {
        "name": "mean_interzonal_spread",
        "column": "mean_interzone_spread_C",
        "ylabel": "Mean interzonal spread [K]",
        "transform": lambda s: s,
        "filename": "kpi_mean_interzonal_spread_by_variant",
    },
    {
        "name": "maximum_air_temperature",
        "column": "max_tair_C",
        "ylabel": "Maximum air temperature [°C]",
        "transform": lambda s: s,
        "filename": "kpi_maximum_air_temperature_by_variant",
    },
]


# ===============================================================
# STYLE
# ===============================================================

def apply_thesis_style() -> None:
    """Apply a consistent thesis-style Matplotlib configuration."""
    plt.rcParams.update({
        "figure.dpi": DPI,
        "savefig.dpi": DPI,

        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,

        "text.color": COLORS["text"],
        "axes.labelcolor": COLORS["text"],
        "xtick.color": COLORS["text"],
        "ytick.color": COLORS["text"],

        "axes.edgecolor": COLORS["edge"],
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,

        "grid.color": COLORS["grid"],
        "grid.linewidth": 0.5,
        "grid.alpha": 1.0,

        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    })


# ===============================================================
# DATA PROCESSING
# ===============================================================

def load_and_prepare(path: Path) -> pd.DataFrame:
    """Load the run-level KPI CSV and standardise key labels."""
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find input CSV: {path.resolve()}\n"
            "Place run_level_kpis.csv next to this script or edit INPUT_CSV."
        )

    df = pd.read_csv(path)

    required = {
        "variant",
        "weather_key",
        "sa_retrofit_state",
        "annual_heating_kWh",
        "peak_heating_kW",
        "overheating_hours_any_zone_gt_26C",
        "max_interzone_spread_C",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    df = df.copy()
    df["weather_label"] = df["weather_key"].map(WEATHER_LABELS).fillna(df["weather_key"])
    df["sa_retrofit_state"] = df["sa_retrofit_state"].astype(str).str.lower()
    df["retrofit_label"] = df["sa_retrofit_state"].map(RETROFIT_LABELS)

    allowed_variants = VARIANT_ORDER_WITH_20HH if INCLUDE_20HH else VARIANT_ORDER_MAIN
    df = df[df["variant"].isin(allowed_variants)].copy()
    df = df[df["sa_retrofit_state"].isin(RETROFIT_ORDER)].copy()
    df = df[df["weather_label"].isin(["Napoli", "Munich"])].copy()

    if df.empty:
        raise ValueError("No rows remain after applying filters.")

    return df


def summarise_kpi(
    df: pd.DataFrame,
    column: str,
    transform: Callable[[pd.Series], pd.Series],
) -> pd.DataFrame:
    """Calculate P5, P50 and P95 for one KPI."""
    temp = df.copy()
    temp["_value"] = transform(pd.to_numeric(temp[column], errors="coerce"))
    temp = temp.dropna(subset=["_value"])

    summary = (
        temp.groupby(["variant", "sa_retrofit_state", "weather_label"], observed=True)["_value"]
        .quantile([0.05, 0.50, 0.95])
        .unstack()
        .reset_index()
        .rename(columns={0.05: "p05", 0.50: "p50", 0.95: "p95"})
    )

    return summary


# ===============================================================
# PLOTTING
# ===============================================================

def set_clean_axes(ax: plt.Axes) -> None:
    """Apply consistent axis styling."""
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.5)
    ax.grid(axis="x", visible=False)

    for spine in ax.spines.values():
        spine.set_color(COLORS["edge"])
        spine.set_linewidth(0.8)

    ax.tick_params(axis="both", colors=COLORS["text"], width=0.8, length=3)


def nice_ymax(value: float) -> float:
    """Return a clean upper y-limit."""
    if not np.isfinite(value) or value <= 0:
        return 1.0

    raw = value * 1.10
    exponent = np.floor(np.log10(raw))
    step = 10 ** exponent

    for mult in [1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 7.5, 10]:
        candidate = mult * step
        if candidate >= raw:
            return candidate

    return raw


def plot_kpi_single_state(
    summary: pd.DataFrame,
    config: dict,
    retrofit_state: str,
    y_max: float,
) -> Path:
    """Create one full-width single-panel KPI figure for one construction state."""
    variant_order = VARIANT_ORDER_WITH_20HH if INCLUDE_20HH else VARIANT_ORDER_MAIN
    variant_order = [v for v in variant_order if v in summary["variant"].unique()]

    x = np.arange(len(variant_order))
    width = 0.34
    offsets = {
        "Napoli": -width / 2,
        "Munich": width / 2,
    }

    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=False)

    state_data = summary[summary["sa_retrofit_state"] == retrofit_state].copy()

    for weather in ["Napoli", "Munich"]:
        weather_data = (
            state_data[state_data["weather_label"] == weather]
            .set_index("variant")
            .reindex(variant_order)
        )

        p50 = weather_data["p50"].to_numpy(dtype=float)
        p05 = weather_data["p05"].to_numpy(dtype=float)
        p95 = weather_data["p95"].to_numpy(dtype=float)

        lower = np.clip(p50 - p05, a_min=0, a_max=None)
        upper = np.clip(p95 - p50, a_min=0, a_max=None)

        ax.bar(
            x + offsets[weather],
            p50,
            width=width,
            color=WEATHER_COLORS[weather],
            edgecolor=COLORS["axisedge"],
            linewidth=0.55,
            label=weather,
            zorder=2,
        )

        ax.errorbar(
            x + offsets[weather],
            p50,
            yerr=np.vstack([lower, upper]),
            fmt="none",
            ecolor=COLORS["error"],
            elinewidth=1.00,
            capsize=2.4,
            capthick=1.00,
            zorder=3,
        )

    if SHOW_STATE_TITLE:
        ax.set_title(RETROFIT_LABELS[retrofit_state], pad=5)

    ax.set_ylabel(config["ylabel"])
    ax.set_xticks(x)
    ax.set_xticklabels([VARIANT_LABELS.get(v, v) for v in variant_order])
    # ax.set_xlabel("Zoning variant")
    ax.set_ylim(0, y_max * 1.04)
    set_clean_axes(ax)

    legend_handles = [
        Patch(facecolor=WEATHER_COLORS["Napoli"], edgecolor=None, linewidth=0.55, label="Napoli"),
        Patch(facecolor=WEATHER_COLORS["Munich"], edgecolor=None, linewidth=0.55, label="Munich"),
        Line2D([0], [0], color=COLORS["error"], linewidth=0.9, label="P5–P95 range"),
    ]

    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.74, 0.98 if SHOW_STATE_TITLE else 1.16),
        ncol=3,
        frameon=False,
        handlelength=1.4,
        columnspacing=1.0,
        handletextpad=0.4,
    )

    fig.subplots_adjust(
        left=0.095,
        right=0.995,
        bottom=0.18,
        top=0.78 if SHOW_STATE_TITLE else 0.84,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    suffix = retrofit_state.lower()
    pdf_path = OUTPUT_DIR / f"{config['filename']}_{suffix}.pdf"
    png_path = OUTPUT_DIR / f"{config['filename']}_{suffix}.png"

    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)

    return pdf_path


def plot_kpi(df: pd.DataFrame, config: dict) -> list[Path]:
    """Create one full-width single-panel figure per construction state for one KPI."""
    summary = summarise_kpi(df, config["column"], config["transform"])

    if SHARE_Y_LIMIT_ACROSS_STATES:
        y_max = nice_ymax(summary["p95"].max())
    else:
        y_max = np.nan

    paths = []
    for retrofit_state in RETROFIT_ORDER:
        state_summary = summary[summary["sa_retrofit_state"] == retrofit_state]
        state_y_max = y_max if SHARE_Y_LIMIT_ACROSS_STATES else nice_ymax(state_summary["p95"].max())
        paths.append(plot_kpi_single_state(summary, config, retrofit_state, state_y_max))

    return paths


def main() -> None:
    apply_thesis_style()

    df = load_and_prepare(INPUT_CSV)

    variant_order = VARIANT_ORDER_WITH_20HH if INCLUDE_20HH else VARIANT_ORDER_MAIN
    variants_available = [v for v in variant_order if v in df["variant"].unique()]

    print(f"Loaded {len(df):,} rows from {INPUT_CSV}")
    print(f"Variants plotted: {', '.join(variants_available)}")
    print(f"Output folder: {OUTPUT_DIR.resolve()}")

    output_paths: list[Path] = []
    for config in KPI_CONFIGS:
        output_paths.extend(plot_kpi(df, config))

    print("\nCreated figures:")
    for path in output_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
