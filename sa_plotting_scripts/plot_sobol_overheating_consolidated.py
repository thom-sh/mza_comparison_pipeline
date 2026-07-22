# -*- coding: utf-8 -*-
"""
Plot consolidated Sobol sensitivity indices for overheating hours.

This script reads the generated:
    sobol_indices_overheating_hours.csv

It does NOT recalculate overheating hours from timeseries.csv.
It only plots the already-computed Sobol indices.

Visual encoding:
    - colour = weather case
        Napoli = light Sobol orange
        Munich = darker Sobol orange
    - hatch = Sobol index type
        S1 = solid bar
        ST = hatched bar

Legend handles:
    Napoli S1
    Napoli ST
    Munich S1
    Munich ST
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ===============================================================
# CONFIG
# ===============================================================

# Use this if running as a normal .py file
try:
    PROJECT_DIR = Path(__file__).resolve().parent
except NameError:
    # Use this fallback if running in Jupyter
    PROJECT_DIR = Path.cwd()

# ---------------------------------------------------------------
# Option A: relative thesis project path
# ---------------------------------------------------------------
INPUT_CSV = (
    PROJECT_DIR
    / "sa_results"
    / "sa_step4_sobol"
    / "sobol_indices_overheating_hours.csv"
)

# ---------------------------------------------------------------
# Option B: uncomment and use this if running from another folder
# ---------------------------------------------------------------
# PROJECT_DIR = Path(r"C:\WF\Thomas Sharon\Floorplan_Dataset\Sensitivity Analysis")
# INPUT_CSV = (
#     PROJECT_DIR
#     / "output"
#     / "figures_sobol_thesis_style_consolidated"
#     / "sobol_indices_overheating_hours.csv"
# )

OUTPUT_DIR = PROJECT_DIR / "output" / "figures_sobol_thesis_style_consolidated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300
FIGSIZE = (6.4, 3.2)

VARIANT_ORDER = ["V3"]

WEATHER_ORDER = ["TRY_A", "TRY_B"]

WEATHER_LABELS = {
    "TRY_A": "Napoli",
    "TRY_B": "Munich",
}

TOP_N_PARAMS = None
FIXED_XMAX = 1.0

SHOW_TITLES = False
SHOW_VALUE_LABELS = False
SHOW_FIGURE = True


# ===============================================================
# THESIS STYLE
# ===============================================================

COLORS = {
    # Sobol orange palette
    "napoli": "#F1D7B5",
    "munich": "#D9A36A",

    # General thesis style
    "edge": "#000000",
    "grid": "#D9D9D9",
    "text": "#000000",
    "legend_edge": "#BDC1C5",
}

WEATHER_COLORS = {
    "TRY_A": COLORS["napoli"],
    "TRY_B": COLORS["munich"],
}

INDEX_ORDER = ["S1", "ST"]

INDEX_HATCHES = {
    "S1": "",
    "ST": "////",
}

HATCH_COLOR = "#2F2F2F"
HATCH_LINEWIDTH = 0.5

PARAM_LABELS = {
    "baseACH": "Base ACH",
    "yoc_shift": "YOC shift",
    "tset_mean_C": "Mean setpoint",
    "shadingFactor": "Shading factor",
    "gWin": "Window g-value",
    "gains_scale": "Gains scale",
    "wwr_factor": "WWR factor",
    "tset_spread_K": "Setpoint spread",
    "UWin": "Window U-value",
    "hConWin": "Window h-conv.",
}


# ===============================================================
# STYLE FUNCTIONS
# ===============================================================

def apply_thesis_style() -> None:
    plt.rcParams.update({
        "figure.dpi": DPI,
        "savefig.dpi": DPI,

        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,

        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",

        "text.color": COLORS["text"],
        "axes.labelcolor": COLORS["text"],
        "xtick.color": COLORS["text"],
        "ytick.color": COLORS["text"],

        "axes.edgecolor": "black",
        "axes.linewidth": 0.8,

        "grid.color": COLORS["grid"],
        "grid.linewidth": 0.5,
        "grid.alpha": 1.0,

        "hatch.linewidth": HATCH_LINEWIDTH,

        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    })


def clean_axes(ax: plt.Axes, grid_axis: str = "x") -> None:
    ax.set_axisbelow(True)

    if grid_axis == "x":
        ax.grid(axis="x", zorder=0)
        ax.grid(axis="y", visible=False)
    elif grid_axis == "y":
        ax.grid(axis="y", zorder=0)
        ax.grid(axis="x", visible=False)
    else:
        ax.grid(True, zorder=0)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("black")

    ax.tick_params(
        axis="both",
        which="both",
        color="black",
        labelcolor="black",
        width=0.8,
        length=3,
    )


def nice_max(value: float) -> float:
    if not np.isfinite(value) or value <= 0:
        return 1.0

    raw = value * 1.12
    exponent = np.floor(np.log10(raw))
    step = 10 ** exponent

    for mult in [1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 7.5, 10]:
        candidate = mult * step
        if candidate >= raw:
            return candidate

    return raw


# ===============================================================
# DATA FUNCTIONS
# ===============================================================

def load_overheating_sobol_indices(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find input CSV:\n{path.resolve()}\n\n"
            "Check INPUT_CSV. It should point to sobol_indices_overheating_hours.csv."
        )

    df = pd.read_csv(path)

    required = {
        "variant",
        "weather_key",
        "param",
        "S1",
        "ST",
    }

    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"Missing required columns in {path.name}: {missing}"
        )

    df = df.copy()

    df = df[df["variant"].isin(VARIANT_ORDER)].copy()
    df = df[df["weather_key"].isin(WEATHER_ORDER)].copy()

    if df.empty:
        raise ValueError(
            "No rows remain after filtering variant/weather. "
            "Check VARIANT_ORDER and WEATHER_ORDER."
        )

    df["S1"] = pd.to_numeric(df["S1"], errors="coerce")
    df["ST"] = pd.to_numeric(df["ST"], errors="coerce")
    df["param_label"] = df["param"].map(PARAM_LABELS).fillna(df["param"])
    df["weather_label"] = df["weather_key"].map(WEATHER_LABELS).fillna(df["weather_key"])

    return df


def prepare_combined_plot_data(df: pd.DataFrame) -> pd.DataFrame:
    part = df.copy()

    part = (
        part.groupby(["weather_key", "param", "param_label"], as_index=False)
        .agg(
            S1=("S1", "mean"),
            ST=("ST", "mean"),
        )
    )

    available_weather = [w for w in WEATHER_ORDER if w in part["weather_key"].unique()]
    if len(available_weather) < 1:
        return pd.DataFrame()

    order_df = (
        part.groupby("param_label", as_index=False)["ST"]
        .max()
        .sort_values("ST", ascending=False)
    )

    if TOP_N_PARAMS is not None:
        order_df = order_df.head(TOP_N_PARAMS)

    param_order = order_df["param_label"].tolist()

    part = part[part["param_label"].isin(param_order)].copy()
    part["param_label"] = pd.Categorical(
        part["param_label"],
        categories=param_order,
        ordered=True,
    )

    part = part.sort_values("param_label")

    return part


# ===============================================================
# BAR AND LEGEND HELPERS
# ===============================================================

def draw_barh_with_optional_hatch(
    ax: plt.Axes,
    y_pos: np.ndarray,
    values: np.ndarray,
    height: float,
    facecolor: str,
    hatch: str = "",
) -> None:
    """
    Draw a horizontal bar with an optional hatch overlay.

    The first layer draws the filled bar.
    The second layer draws only hatch lines, so hatch colour is independent.
    """

    ax.barh(
        y_pos,
        values,
        height=height,
        color=facecolor,
        edgecolor=COLORS["edge"],
        linewidth=0.65,
        zorder=2,
    )

    if hatch:
        ax.barh(
            y_pos,
            values,
            height=height,
            color="none",
            edgecolor=HATCH_COLOR,
            linewidth=0.0,
            hatch=hatch,
            zorder=3,
        )


def add_sobol_legend(ax: plt.Axes) -> None:
    """
    Combined legend handles:
        Napoli S1
        Napoli ST
        Munich S1
        Munich ST
    """

    handles = [
        Patch(
            facecolor=WEATHER_COLORS["TRY_A"],
            edgecolor=COLORS["edge"],
            linewidth=0.65,
            hatch=INDEX_HATCHES["S1"],
            label=r"Napoli $S_1$",
        ),
        Patch(
            facecolor=WEATHER_COLORS["TRY_A"],
            edgecolor=HATCH_COLOR,
            linewidth=0.65,
            hatch=INDEX_HATCHES["ST"],
            label=r"Napoli $S_T$",
        ),
        Patch(
            facecolor=WEATHER_COLORS["TRY_B"],
            edgecolor=COLORS["edge"],
            linewidth=0.65,
            hatch=INDEX_HATCHES["S1"],
            label=r"Munich $S_1$",
        ),
        Patch(
            facecolor=WEATHER_COLORS["TRY_B"],
            edgecolor=HATCH_COLOR,
            linewidth=0.65,
            hatch=INDEX_HATCHES["ST"],
            label=r"Munich $S_T$",
        ),
    ]

    legend = ax.legend(
        handles=handles,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        facecolor="white",
        edgecolor=COLORS["legend_edge"],
        loc="lower right",
        ncol=1,
        borderpad=0.25,
        handlelength=1.4,
        handleheight=0.7,
        handletextpad=0.35,
        labelspacing=0.20,
        fontsize=8,
    )

    legend.get_frame().set_linewidth(0.7)


# ===============================================================
# PLOT
# ===============================================================

def plot_overheating_combined_weather_s1_st(df: pd.DataFrame) -> Path | None:
    part = prepare_combined_plot_data(df)

    if part.empty:
        print("Skipping plot because no data are available.")
        return None

    available_weather = [w for w in WEATHER_ORDER if w in part["weather_key"].unique()]
    if len(available_weather) < 1:
        print("Skipping plot because no weather cases are available.")
        return None

    param_order = list(part["param_label"].cat.categories)
    y = np.arange(len(param_order))

    bar_height = 0.16
    gap_between_index_groups = 0.08

    offsets = {
        ("S1", available_weather[0]): -(1.5 * bar_height + gap_between_index_groups / 2),
        ("S1", available_weather[1]): -(0.5 * bar_height + gap_between_index_groups / 2),
        ("ST", available_weather[0]): +(0.5 * bar_height + gap_between_index_groups / 2),
        ("ST", available_weather[1]): +(1.5 * bar_height + gap_between_index_groups / 2),
    }

    fig_height = max(3.0, 0.40 * len(param_order) + 1.05)
    fig, ax = plt.subplots(figsize=(FIGSIZE[0], fig_height))
    ax.set_axisbelow(True)

    max_value = 0.0

    for index_type in INDEX_ORDER:
        for weather_key in available_weather[:2]:
            values_df = (
                part[part["weather_key"] == weather_key]
                .set_index("param_label")
                .reindex(param_order)
            )

            values = values_df[index_type].to_numpy(dtype=float)

            if np.isfinite(values).any():
                max_value = max(max_value, float(np.nanmax(values)))

            y_pos = y + offsets[(index_type, weather_key)]

            draw_barh_with_optional_hatch(
                ax=ax,
                y_pos=y_pos,
                values=values,
                height=bar_height,
                facecolor=WEATHER_COLORS.get(weather_key, COLORS["napoli"]),
                hatch=INDEX_HATCHES[index_type],
            )

            if SHOW_VALUE_LABELS:
                for yy, value in zip(y_pos, values):
                    if np.isfinite(value) and value > 0:
                        ax.text(
                            value + 0.01,
                            yy,
                            f"{value:.2f}",
                            ha="left",
                            va="center",
                            fontsize=8,
                            color=COLORS["text"],
                        )

    ax.set_yticks(y)
    ax.set_yticklabels(param_order)
    ax.invert_yaxis()

    ax.set_xlabel("Sobol sensitivity index")

    if FIXED_XMAX is not None:
        ax.set_xlim(0, FIXED_XMAX)
    else:
        ax.set_xlim(0, nice_max(max_value))

    if SHOW_TITLES:
        ax.set_title("Overheating hours", pad=6)

    clean_axes(ax, grid_axis="x")
    add_sobol_legend(ax)

    fig.subplots_adjust(
        left=0.27,
        right=0.995,
        bottom=0.22,
        top=0.98,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_path = OUTPUT_DIR / "sobol_v3_overheating_hours_combined_weather_S1_ST_from_csv.pdf"
    png_path = OUTPUT_DIR / "sobol_v3_overheating_hours_combined_weather_S1_ST_from_csv.png"

    fig.savefig(pdf_path)
    fig.savefig(png_path)

    if SHOW_FIGURE:
        plt.show()

    plt.close(fig)

    return pdf_path


# ===============================================================
# MAIN
# ===============================================================

def main() -> None:
    apply_thesis_style()

    df = load_overheating_sobol_indices(INPUT_CSV)

    print(f"Loaded {len(df):,} overheating Sobol-index rows from:")
    print(INPUT_CSV)

    print("\nVariants used:")
    print(", ".join(sorted(df["variant"].unique())))

    print("\nWeather cases available:")
    print(", ".join(sorted(df["weather_key"].unique())))

    print("\nParameters available:")
    print(", ".join(sorted(df["param"].unique())))

    path = plot_overheating_combined_weather_s1_st(df)

    if path is not None:
        print("\nCreated figure:")
        print(path)


if __name__ == "__main__":
    main()