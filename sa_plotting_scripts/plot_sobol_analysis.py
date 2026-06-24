# -*- coding: utf-8 -*-
"""
Thesis-style Sobol sensitivity plots from sobol_indices.csv.

Creates:
1) Horizontal grouped bar plots for Sobol S1 and ST indices
   - one plot per KPI and weather case
2) Optional weather-comparison plot for total-order index ST

Input expected:
    sobol_indices.csv

Required columns:
    variant, weather_key, kpi, param, S1, ST
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ===============================================================
# CONFIG
# ===============================================================

PROJECT_DIR = Path(__file__).resolve().parent

INPUT_CSV = (
    PROJECT_DIR
    / "sa_results"
    / "sa_step4_sobol"
    / "sobol_indices.csv"
)

OUTPUT_DIR = PROJECT_DIR / "output" / "figures_sobol_thesis_style"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300

# Only V3
VARIANT_ORDER = ["V3"]

# Set to None to use all available weather cases
WEATHER_ORDER = ["TRY_A", "TRY_B"]

WEATHER_LABELS = {
    "TRY_A": "Napoli",
    "TRY_B": "Munich",
}

# Main plots: S1 and ST for each KPI/weather
CREATE_S1_ST_BAR_PLOTS = True

# Additional plot: compare ST between weather cases
CREATE_ST_WEATHER_COMPARISON = True

# If None, all parameters are shown
TOP_N_PARAMS = None

# Sobol indices are dimensionless and usually interpreted from 0 to 1.
# Use 1.0 for direct comparability; set to None for automatic tighter limits.
FIXED_XMAX = 1.0

SHOW_TITLES = False
SHOW_VALUE_LABELS = False


# ===============================================================
# THESIS STYLE
# ===============================================================

COLORS = {
    # Orange sensitivity palette
    "S1": "#F1D7B5",        # light warm orange
    "ST": "#D9A36A",        # muted orange
    "ST_dark": "#B86B4B",   # dark burnt orange

    # Weather comparison shades
    "TRY_A": "#F1D7B5",
    "TRY_B": "#D9A36A",

    # General thesis style
    "grid": "#D9D9D9",
    "text": "#000000",
    "edge": "#777D84",
    "legend_edge": "#BDC1C5",
}

PARAM_LABELS = {
    "baseACH": "Base ACH",
    "wwr_factor": "WWR factor",
    "gains_scale": "Gains scale",
    "tset_mean_C": "Mean setpoint",
    "tset_spread_K": "Setpoint spread",
    "yoc_shift": "YOC shift",
    "shadingFactor": "Shading factor",
    "gWin": "Window g-value",
    "UWin": "Window U-value",
    "hConWin": "Window h-conv.",
}

KPI_CONFIGS = [
    {
        "kpi": "heat_demand_kWh",
        "filename": "sobol_v3_annual_heating_demand",
        "title": "Annual heating demand",
        "xlabel": "Sobol sensitivity index [-]",
    },
    {
        "kpi": "peak_heat_kW",
        "filename": "sobol_v3_peak_heating_load",
        "title": "Peak heating load",
        "xlabel": "Sobol sensitivity index [-]",
    },
]


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
    elif grid_axis == "both":
        ax.grid(True, zorder=0)
    else:
        ax.grid(False)

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

def load_sobol_indices(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find input CSV:\n{path.resolve()}\n\n"
            "Check that INPUT_CSV ends with sobol_indices.csv."
        )

    if path.is_dir():
        raise IsADirectoryError(
            f"INPUT_CSV points to a folder, not a CSV file:\n{path.resolve()}\n\n"
            "Change INPUT_CSV so it ends with sobol_indices.csv."
        )

    df = pd.read_csv(path)

    required = {
        "variant",
        "weather_key",
        "kpi",
        "param",
        "S1",
        "ST",
    }

    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in sobol_indices.csv: {missing}")

    df = df.copy()

    df = df[df["variant"].isin(VARIANT_ORDER)].copy()

    if WEATHER_ORDER is not None:
        df = df[df["weather_key"].isin(WEATHER_ORDER)].copy()

    if df.empty:
        raise ValueError("No rows remain after filtering variant/weather.")

    df["S1"] = pd.to_numeric(df["S1"], errors="coerce")
    df["ST"] = pd.to_numeric(df["ST"], errors="coerce")
    df["param_label"] = df["param"].map(PARAM_LABELS).fillna(df["param"])
    df["weather_label"] = df["weather_key"].map(WEATHER_LABELS).fillna(df["weather_key"])

    return df


def prepare_kpi_weather_data(
    df: pd.DataFrame,
    kpi: str,
    weather_key: str,
) -> pd.DataFrame:
    part = df[
        (df["kpi"] == kpi)
        & (df["weather_key"] == weather_key)
    ].copy()

    if part.empty:
        return part

    # Average in case there are multiple seeds/buildings.
    part = (
        part.groupby(["param", "param_label"], as_index=False)
        .agg(
            S1=("S1", "mean"),
            ST=("ST", "mean"),
        )
    )

    return part


def get_param_order(part: pd.DataFrame) -> list[str]:
    order = (
        part.sort_values("ST", ascending=False)["param_label"]
        .tolist()
    )

    if TOP_N_PARAMS is not None:
        order = order[:TOP_N_PARAMS]

    return order


# ===============================================================
# PLOT 1: S1 AND ST BAR PLOT
# ===============================================================

def plot_sobol_s1_st_bar(
    df: pd.DataFrame,
    config: dict,
    weather_key: str,
) -> Path | None:
    part = prepare_kpi_weather_data(df, config["kpi"], weather_key)

    if part.empty:
        print(f"Skipping missing combination: {config['kpi']} / {weather_key}")
        return None

    param_order = get_param_order(part)
    part = part[part["param_label"].isin(param_order)].copy()

    part = (
        part.set_index("param_label")
        .reindex(param_order)
        .reset_index()
    )

    y = np.arange(len(param_order))
    bar_height = 0.24

    fig_height = max(2.8, 0.34 * len(param_order) + 0.75)
    fig, ax = plt.subplots(figsize=(6.4, fig_height))
    ax.set_axisbelow(True)

    s1_values = part["S1"].to_numpy(dtype=float)
    st_values = part["ST"].to_numpy(dtype=float)

    max_value = np.nanmax([np.nanmax(s1_values), np.nanmax(st_values)])

    ax.barh(
        y - bar_height / 2,
        s1_values,
        height=bar_height,
        label=r"$S_1$",
        color=COLORS["S1"],
        edgecolor=COLORS["edge"],
        linewidth=0.7,
        zorder=2,
    )

    ax.barh(
        y + bar_height / 2,
        st_values,
        height=bar_height,
        label=r"$S_T$",
        color=COLORS["ST"],
        edgecolor=COLORS["edge"],
        linewidth=0.7,
        zorder=2,
    )

    if SHOW_VALUE_LABELS:
        for yy, value in zip(y - bar_height / 2, s1_values):
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

        for yy, value in zip(y + bar_height / 2, st_values):
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

    ax.set_xlabel(config["xlabel"])

    if FIXED_XMAX is not None:
        ax.set_xlim(0, FIXED_XMAX)
    else:
        ax.set_xlim(0, nice_max(max_value))

    if SHOW_TITLES:
        ax.set_title(f"{config['title']} - {WEATHER_LABELS.get(weather_key, weather_key)}", pad=6)

    clean_axes(ax, grid_axis="x")

    legend = ax.legend(
        frameon=True,
        fancybox=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor=COLORS["legend_edge"],
        loc="lower right",
        ncol=1,
        borderpad=0.4,
        handlelength=1.8,
        handletextpad=0.6,
        labelspacing=0.4,
    )
    legend.get_frame().set_linewidth(0.8)

    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    weather_suffix = WEATHER_LABELS.get(weather_key, weather_key).lower()
    weather_suffix = weather_suffix.replace(" ", "_")

    pdf_path = OUTPUT_DIR / f"{config['filename']}_{weather_suffix}_S1_ST.pdf"
    png_path = OUTPUT_DIR / f"{config['filename']}_{weather_suffix}_S1_ST.png"

    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)

    return pdf_path


# ===============================================================
# PLOT 2: WEATHER COMPARISON OF ST
# ===============================================================

def plot_sobol_st_weather_comparison(
    df: pd.DataFrame,
    config: dict,
) -> Path | None:
    part = df[df["kpi"] == config["kpi"]].copy()

    if part.empty:
        print(f"Skipping weather comparison for missing KPI: {config['kpi']}")
        return None

    part = (
        part.groupby(["weather_key", "param", "param_label"], as_index=False)
        .agg(ST=("ST", "mean"))
    )

    pivot = (
        part.pivot_table(
            index="param_label",
            columns="weather_key",
            values="ST",
            aggfunc="mean",
        )
    )

    available_weather = [w for w in WEATHER_ORDER if w in pivot.columns]

    if len(available_weather) < 2:
        print(f"Skipping weather comparison for {config['kpi']} because fewer than two weather cases are available.")
        return None

    pivot["_max_ST"] = pivot[available_weather].max(axis=1)
    pivot = pivot.sort_values("_max_ST", ascending=False)

    if TOP_N_PARAMS is not None:
        pivot = pivot.head(TOP_N_PARAMS)

    param_order = pivot.index.tolist()
    y = np.arange(len(param_order))
    bar_height = 0.24

    fig_height = max(2.8, 0.34 * len(param_order) + 0.75)
    fig, ax = plt.subplots(figsize=(6.4, fig_height))
    ax.set_axisbelow(True)

    offsets = {
        available_weather[0]: -bar_height / 2,
        available_weather[1]: bar_height / 2,
    }

    max_value = 0.0

    for weather_key in available_weather[:2]:
        values = pivot[weather_key].to_numpy(dtype=float)
        max_value = max(max_value, np.nanmax(values))

        ax.barh(
            y + offsets[weather_key],
            values,
            height=bar_height,
            label=WEATHER_LABELS.get(weather_key, weather_key),
            color=COLORS.get(weather_key, COLORS["ST"]),
            edgecolor=COLORS["edge"],
            linewidth=0.7,
            zorder=2,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(param_order)
    ax.invert_yaxis()

    ax.set_xlabel(r"Sobol total-order index $S_T$ [-]")

    if FIXED_XMAX is not None:
        ax.set_xlim(0, FIXED_XMAX)
    else:
        ax.set_xlim(0, nice_max(max_value))

    if SHOW_TITLES:
        ax.set_title(config["title"], pad=6)

    clean_axes(ax, grid_axis="x")

    legend = ax.legend(
        frameon=True,
        fancybox=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor=COLORS["legend_edge"],
        loc="lower right",
        ncol=1,
        borderpad=0.4,
        handlelength=1.8,
        handletextpad=0.6,
        labelspacing=0.4,
    )
    legend.get_frame().set_linewidth(0.8)

    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_path = OUTPUT_DIR / f"{config['filename']}_weather_comparison_ST.pdf"
    png_path = OUTPUT_DIR / f"{config['filename']}_weather_comparison_ST.png"

    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)

    return pdf_path


# ===============================================================
# MAIN
# ===============================================================

def main() -> None:
    apply_thesis_style()

    df = load_sobol_indices(INPUT_CSV)

    print(f"Loaded {len(df):,} Sobol-index rows from:")
    print(INPUT_CSV)

    print("\nVariants used:")
    print(", ".join(sorted(df["variant"].unique())))

    print("\nWeather cases available:")
    print(", ".join(sorted(df["weather_key"].unique())))

    print("\nKPIs available:")
    print(", ".join(sorted(df["kpi"].unique())))

    created_paths: list[Path] = []

    for config in KPI_CONFIGS:
        if CREATE_S1_ST_BAR_PLOTS:
            for weather_key in WEATHER_ORDER:
                path = plot_sobol_s1_st_bar(df, config, weather_key)
                if path is not None:
                    created_paths.append(path)

        if CREATE_ST_WEATHER_COMPARISON:
            path = plot_sobol_st_weather_comparison(df, config)
            if path is not None:
                created_paths.append(path)

    print("\nCreated figures:")
    for path in created_paths:
        print(path)


if __name__ == "__main__":
    main()