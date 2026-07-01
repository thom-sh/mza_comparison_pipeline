# -*- coding: utf-8 -*-
"""
Plot thesis-style consolidated zoning-variant KPI figures from run_level_kpis.csv.

Creates one full-text-width figure per KPI:
  1) Annual heating demand [MWh]
  2) Peak heating load [kW]
  3) Overheating hours, any zone > 26 °C [h]
  4) Maximum interzonal temperature spread [K]
  5) Mean interzonal temperature spread [K]
  6) Maximum air temperature [°C]

Figure logic:
  - one figure = one KPI
  - x-axis: zoning variants labelled V1, V2, ..., V8
  - each variant has four bars:
        Napoli standard   = Napoli colour, solid fill
        Napoli retrofit   = Napoli colour, hatched fill
        Munich standard   = Munich colour, solid fill
        Munich retrofit   = Munich colour, hatched fill
  - bar height: P50 / median
  - error bars: P5--P95 range from all propagated uncertainty

Place this script next to the project folder containing:
  sa_results/sa_main/run_level_kpis.csv
or edit INPUT_CSV below.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    / "figures_kpi_variant_thesis_style_consolidated"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Main thesis variants. Set True only if you want to include the additional 20HH variants.
INCLUDE_20HH = False

DPI = 300

# Full text-width figure. Slightly taller because each KPI now contains four bars per variant.
FIGSIZE = (6.14, 3.65)

# If True, missing KPI columns are skipped with a warning.
# If False, the script stops when a KPI column is missing.
SKIP_MISSING_KPIS = True

# If True, the plot receives a small KPI title.
# Set False if you prefer to describe the KPI only in the LaTeX caption.
SHOW_KPI_TITLE = False

# Thesis colours: two teal shades + black text/axes/error bars.
COLORS = {
    "napoli": "#A9D1CE",    # light muted teal
    "munich": "#3F7D7A",    # dark muted teal
    "error": "#000000",     # black P5--P95 error bars
    "edge": "#000000",      # bar edges / hatch colour
    "axisedge": "#777D84",  # axes
    "grid": "#D9D9D9",      # light grey gridlines
    "text": "#000000",      # black text
}

HATCH_COLOR = "#2F2F2F"   # hatch line colour only
HATCH_LINEWIDTH = 0.5     # hatch line thickness

WEATHER_LABELS = {
    "TRY_A": "Napoli",
    "TRY_B": "Munich",
    "Napoli": "Napoli",
    "Munich": "Munich",
}

WEATHER_ORDER = ["Napoli", "Munich"]

WEATHER_COLORS = {
    "Napoli": COLORS["napoli"],
    "Munich": COLORS["munich"],
}

RETROFIT_ORDER = ["standard", "retrofit"]
RETROFIT_LABELS = {
    "standard": "Standard",
    "retrofit": "Retrofit",
}

# Empty string = solid fill. Retrofit receives hatch but keeps the same weather colour.
RETROFIT_HATCHES = {
    "standard": "",
    "retrofit": "////",
}

VARIANT_ORDER_MAIN = ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8"]
VARIANT_ORDER_WITH_20HH = [
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V1_20HH", "V2_20HH"
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
        "filename": "kpi_annual_heating_demand_by_variant_consolidated",
        "title": "Annual heating demand",
    },
    {
        "name": "peak_heating_load",
        "column": "peak_heating_kW",
        "ylabel": "Peak heating load [kW]",
        "transform": lambda s: s,
        "filename": "kpi_peak_heating_load_by_variant_consolidated",
        "title": "Peak heating load",
    },
    {
        "name": "overheating_hours",
        "column": "overheating_hours_any_zone_gt_26C",
        "ylabel": "Overheating hours [h]",
        "transform": lambda s: s,
        "filename": "kpi_overheating_hours_by_variant_consolidated",
        "title": "Overheating hours",
    },
    {
        "name": "max_interzonal_spread",
        "column": "max_interzone_spread_C",
        "ylabel": "Maximum interzonal spread [K]",
        "transform": lambda s: s,
        "filename": "kpi_max_interzonal_spread_by_variant_consolidated",
        "title": "Maximum interzonal spread",
    },
    {
        "name": "mean_interzonal_spread",
        "column": "mean_interzone_spread_C",
        "ylabel": "Mean interzonal spread [K]",
        "transform": lambda s: s,
        "filename": "kpi_mean_interzonal_spread_by_variant_consolidated",
        "title": "Mean interzonal spread",
    },
    {
        "name": "maximum_air_temperature",
        "column": "max_tair_C",
        "ylabel": "Maximum air temperature [°C]",
        "transform": lambda s: s,
        "filename": "kpi_maximum_air_temperature_by_variant_consolidated",
        "title": "Maximum air temperature",
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

        "axes.edgecolor": COLORS["axisedge"],
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
            "Place run_level_kpis.csv in sa_results/sa_main/ or edit INPUT_CSV."
        )

    df = pd.read_csv(path)

    required = {
        "variant",
        "weather_key",
        "sa_retrofit_state",
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
    df = df[df["weather_label"].isin(WEATHER_ORDER)].copy()

    if df.empty:
        raise ValueError("No rows remain after applying filters.")

    return df


def available_kpi_configs(df: pd.DataFrame) -> list[dict]:
    """Return KPI configs whose columns exist in the dataframe."""
    configs: list[dict] = []
    for config in KPI_CONFIGS:
        if config["column"] in df.columns:
            configs.append(config)
        elif SKIP_MISSING_KPIS:
            print(f"[WARN] Skipping KPI because column is missing: {config['column']}")
        else:
            raise ValueError(f"Missing KPI column in CSV: {config['column']}")

    if not configs:
        raise ValueError("None of the configured KPI columns are available in the CSV.")

    return configs


def summarise_kpi(
    df: pd.DataFrame,
    column: str,
    transform: Callable[[pd.Series], pd.Series],
) -> pd.DataFrame:
    """Calculate P5, P50 and P95 for one KPI by variant, retrofit state and weather."""
    temp = df.copy()
    temp["_value"] = transform(pd.to_numeric(temp[column], errors="coerce"))
    temp = temp.dropna(subset=["_value"])

    if temp.empty:
        raise ValueError(f"No valid numeric values found for KPI column: {column}")

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


def plot_kpi_consolidated(summary: pd.DataFrame, config: dict) -> Path:
    variant_order = VARIANT_ORDER_WITH_20HH if INCLUDE_20HH else VARIANT_ORDER_MAIN
    variant_order = [v for v in variant_order if v in summary["variant"].unique()]

    if not variant_order:
        raise ValueError(f"No variants found for KPI: {config['name']}")

    x = np.arange(len(variant_order))

    bar_width = 0.17

    # NEW ORDER:
    # standard bars together first, then retrofit bars
    combo_order = [
        ("Napoli", "standard"),
        ("Munich", "standard"),
        ("Napoli", "retrofit"),
        ("Munich", "retrofit"),
    ]

    offsets = {
        ("Napoli", "standard"): -1.5 * bar_width,
        ("Munich", "standard"): -0.5 * bar_width,
        ("Napoli", "retrofit"):  0.5 * bar_width,
        ("Munich", "retrofit"):  1.5 * bar_width,
    }

    y_max = nice_ymax(summary["p95"].max())

    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=False)

    for weather, retrofit_state in combo_order:
        data = (
            summary[
                (summary["weather_label"] == weather)
                & (summary["sa_retrofit_state"] == retrofit_state)
            ]
            .set_index("variant")
            .reindex(variant_order)
        )

        p50 = data["p50"].to_numpy(dtype=float)
        p05 = data["p05"].to_numpy(dtype=float)
        p95 = data["p95"].to_numpy(dtype=float)

        lower = np.clip(p50 - p05, a_min=0, a_max=None)
        upper = np.clip(p95 - p50, a_min=0, a_max=None)

        xpos = x + offsets[(weather, retrofit_state)]

        ax.bar(
            xpos,
            p50,
            width=bar_width,
            color=WEATHER_COLORS[weather],
            edgecolor=COLORS["edge"],
            linewidth=0.55,
            # hatch=RETROFIT_HATCHES[retrofit_state],
            zorder=2,
        )

# --------------------------------------------------
# Overlay hatch only for retrofit bars
# --------------------------------------------------
        if retrofit_state == "retrofit":
            with plt.rc_context({
                "hatch.color": HATCH_COLOR,
                "hatch.linewidth": HATCH_LINEWIDTH,
            }):
                ax.bar(
                    xpos,
                    p50,
                    width=bar_width,
                    color=(0, 0, 0, 0),      # transparent fill
                    edgecolor=HATCH_COLOR,    # used for hatch colour
                    linewidth=0.0,            # avoid changing visible border
                    hatch=RETROFIT_HATCHES[retrofit_state],
                    zorder=2.1,
                )

        ax.errorbar(
            xpos,
            p50,
            yerr=np.vstack([lower, upper]),
            fmt="none",
            ecolor=COLORS["error"],
            elinewidth=0.85,
            capsize=1.9,
            capthick=0.85,
            zorder=3,
        )

    if SHOW_KPI_TITLE:
        ax.set_title(config["title"], pad=5)

    ax.set_ylabel(config["ylabel"])

    ax.set_xticks(x)
    ax.set_xticklabels([VARIANT_LABELS.get(v, v) for v in variant_order])

    ax.set_ylim(0, y_max * 1.04)

    set_clean_axes(ax)

    legend_handles = [
        Patch(
            facecolor=WEATHER_COLORS["Napoli"],
            edgecolor=COLORS["edge"],
            linewidth=0.55,
            label="Napoli",
        ),
        Patch(
            facecolor=WEATHER_COLORS["Munich"],
            edgecolor=COLORS["edge"],
            linewidth=0.55,
            label="Munich",
        ),
        Patch(
            facecolor="white",
            edgecolor=COLORS["edge"],
            linewidth=0.65,
            label="Standard",
        ),
        Patch(
            facecolor="white",
            edgecolor=HATCH_COLOR,
            linewidth=HATCH_LINEWIDTH,
            hatch=RETROFIT_HATCHES["retrofit"],
            label="Retrofit",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["error"],
            linewidth=0.9,
            label="P5–P95 range",
        ),
    ]

    ax.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(1.0, 0.975 if not SHOW_KPI_TITLE else 1.18),
        ncol=5,
        frameon=False,
        handlelength=1.35,
        columnspacing=0.85,
        handletextpad=0.35,
        borderaxespad=0.0,
    )

    fig.subplots_adjust(
        left=0.095,
        right=0.995,
        bottom=0.17,
        top=0.76 if not SHOW_KPI_TITLE else 0.74,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_path = OUTPUT_DIR / f"{config['filename']}.pdf"
    png_path = OUTPUT_DIR / f"{config['filename']}.png"
    csv_path = OUTPUT_DIR / f"{config['filename']}_summary.csv"

    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.show()
    plt.close(fig)

    summary.to_csv(csv_path, index=False)

    return pdf_path

def plot_kpi(df: pd.DataFrame, config: dict) -> Path:
    """Create one consolidated full-width figure for one KPI."""
    summary = summarise_kpi(df, config["column"], config["transform"])
    return plot_kpi_consolidated(summary, config)


def main() -> None:
    apply_thesis_style()

    df = load_and_prepare(INPUT_CSV)
    configs = available_kpi_configs(df)

    variant_order = VARIANT_ORDER_WITH_20HH if INCLUDE_20HH else VARIANT_ORDER_MAIN
    variants_available = [v for v in variant_order if v in df["variant"].unique()]

    print(f"Loaded {len(df):,} rows from {INPUT_CSV}")
    print(f"Variants plotted: {', '.join(variants_available)}")
    print("Bar encoding: weather = colour, retrofit state = hatch")
    print(f"Output folder: {OUTPUT_DIR.resolve()}")

    output_paths: list[Path] = []
    for config in configs:
        output_paths.append(plot_kpi(df, config))

    print("\nCreated consolidated figures:")
    for path in output_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
