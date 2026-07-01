# -*- coding: utf-8 -*-
"""
Thesis-style Morris sensitivity plots from morris_indices.csv.

Creates for V3 only:
1) Ranked Morris mu-star horizontal bar plots
2) Morris mu-star vs sigma scatter plots with parameter labels

Input expected:
    morris_indices.csv

Required columns:
    variant, kpi, param, mu, mu_star, sigma

Optional columns:
    building_id, weather_key
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
    / "sa_step3_morris"
    / "morris_indices.csv"
)

OUTPUT_DIR = PROJECT_DIR / "output" / "figures_morris_thesis_style"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300

SCATTER_LABEL_TOP_N = 4

# Only V3
VARIANT_ORDER = ["V3"]

# Optional filters
# Set to None to use all available rows
FILTER_WEATHER = None
FILTER_BUILDING_ID = None

# If None, all parameters are shown.
# If you want only top 8 parameters, set TOP_N_PARAMS = 8
TOP_N_PARAMS = None

# Main thesis figures
CREATE_MU_STAR_BAR = True

# Appendix / optional diagnostic figures
CREATE_MU_STAR_SIGMA_SCATTER = True

# No title inside thesis plot; use LaTeX captions
SHOW_TITLES = False

# Add value labels at bar end
SHOW_BAR_VALUE_LABELS = False


# ===============================================================
# THESIS STYLE
# ===============================================================

COLORS = {
    # Sensitivity-analysis warm palette
    "sensitivity": "#D9A36A",
    "sensitivity_dark": "#B86B4B",
    "sensitivity_light": "#F1D7B5",

    # General thesis style
    "grid": "#D9D9D9",
    "text": "#000000",
    "edge": "#000000",
    "legend_edge": "#BDC1C5",
}

VARIANT_COLORS = {
    "V3": COLORS["sensitivity"],
}

VARIANT_LABELS = {
    "V3": "V3",
}

VARIANT_MARKERS = {
    "V3": "s",
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
        "filename": "morris_v3_mu_star_annual_heating_demand",
        "xlabel": r"Morris $\mu^\ast$ [MWh]",
        "scatter_xlabel": r"Morris $\mu^\ast$ [MWh]",
        "scatter_ylabel": r"Morris $\sigma$ [MWh]",
        "transform": lambda s: s / 1000.0,
        "title": "Annual heating demand",
    },
    {
        "kpi": "peak_heat_kW",
        "filename": "morris_v3_mu_star_peak_heating_load",
        "xlabel": r"Morris $\mu^\ast$ [kW]",
        "scatter_xlabel": r"Morris $\mu^\ast$ [kW]",
        "scatter_ylabel": r"Morris $\sigma$ [kW]",
        "transform": lambda s: s,
        "title": "Peak heating load",
    },

    # Keep this block only if your morris_indices.csv later contains this KPI.
    # Otherwise, the script will automatically skip it.
    {
        "kpi": "overheating_hours",
        "filename": "morris_v3_mu_star_overheating_hours",
        "xlabel": r"Morris $\mu^\ast$ [h]",
        "scatter_xlabel": r"Morris $\mu^\ast$ [h]",
        "scatter_ylabel": r"Morris $\sigma$ [h]",
        "transform": lambda s: s,
        "title": "Overheating hours",
    },
]


# ===============================================================
# STYLE
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
    """Return clean upper axis limit."""
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
# DATA
# ===============================================================

def load_morris_indices(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find input CSV:\n{path.resolve()}\n\n"
            "Check that INPUT_CSV ends with morris_indices.csv."
        )

    if path.is_dir():
        raise IsADirectoryError(
            f"INPUT_CSV points to a folder, not a CSV file:\n{path.resolve()}\n\n"
            "Change INPUT_CSV so it ends with morris_indices.csv."
        )

    df = pd.read_csv(path)

    required = {
        "variant",
        "kpi",
        "param",
        "mu",
        "mu_star",
        "sigma",
    }

    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in morris_indices.csv: {missing}")

    df = df.copy()

    df = df[df["variant"].isin(VARIANT_ORDER)].copy()

    if FILTER_WEATHER is not None and "weather_key" in df.columns:
        df = df[df["weather_key"] == FILTER_WEATHER].copy()

    if FILTER_BUILDING_ID is not None and "building_id" in df.columns:
        df = df[df["building_id"] == FILTER_BUILDING_ID].copy()

    if df.empty:
        raise ValueError("No rows remain after filtering.")

    return df


def prepare_kpi_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    part = df[df["kpi"] == config["kpi"]].copy()

    if part.empty:
        return part

    transform = config["transform"]

    part["mu_plot"] = transform(pd.to_numeric(part["mu"], errors="coerce"))
    part["mu_star_plot"] = transform(pd.to_numeric(part["mu_star"], errors="coerce"))
    part["sigma_plot"] = transform(pd.to_numeric(part["sigma"], errors="coerce"))

    part["param_label"] = part["param"].map(PARAM_LABELS).fillna(part["param"])

    # If there are multiple rows per parameter, average them.
    # This keeps the plot stable if results exist for multiple buildings/weather cases.
    part = (
        part.groupby(["variant", "param", "param_label"], as_index=False)
        .agg(
            mu_plot=("mu_plot", "mean"),
            mu_star_plot=("mu_star_plot", "mean"),
            sigma_plot=("sigma_plot", "mean"),
        )
    )

    return part


def get_ranked_param_order(part: pd.DataFrame) -> list[str]:
    order = (
        part.groupby("param_label")["mu_star_plot"]
        .max()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    if TOP_N_PARAMS is not None:
        order = order[:TOP_N_PARAMS]

    return order


# ===============================================================
# PLOT 1: MORRIS MU-STAR BAR PLOT
# ===============================================================

def plot_morris_mu_star_bar(df: pd.DataFrame, config: dict) -> Path | None:
    part = prepare_kpi_data(df, config)

    if part.empty:
        print(f"Skipping KPI not found in CSV: {config['kpi']}")
        return None

    param_order = get_ranked_param_order(part)
    part = part[part["param_label"].isin(param_order)].copy()

    y = np.arange(len(param_order))
    bar_height = 0.34

    fig_height = max(2.8, 0.34 * len(param_order) + 0.15)
    fig, ax = plt.subplots(figsize=(6.4, fig_height))
    ax.set_axisbelow(True)

    variant = "V3"

    variant_data = (
        part[part["variant"] == variant]
        .set_index("param_label")
        .reindex(param_order)
    )

    values = variant_data["mu_star_plot"].to_numpy(dtype=float)
    values = np.nan_to_num(values, nan=0.0)

    max_value = float(np.nanmax(values)) if len(values) else 0.0
    x_max = nice_max(max_value)

    bars = ax.barh(
        y,
        values,
        height=bar_height,
        color=VARIANT_COLORS[variant],
        edgecolor=COLORS["edge"],
        linewidth=0.7,
        zorder=2,
    )

    if SHOW_BAR_VALUE_LABELS:
        for bar in bars:
            value = bar.get_width()
            if value > 0:
                ax.text(
                    value + x_max * 0.012,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.1f}",
                    ha="left",
                    va="center",
                    fontsize=8,
                    color=COLORS["text"],
                )

    ax.set_yticks(y)
    ax.set_yticklabels(param_order)
    ax.invert_yaxis()

    ax.set_xlabel(config["xlabel"])
    ax.set_xlim(0, x_max)

    if SHOW_TITLES:
        ax.set_title(config["title"], pad=6)

    clean_axes(ax, grid_axis="x")

    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_path = OUTPUT_DIR / f"{config['filename']}_bar.pdf"
    png_path = OUTPUT_DIR / f"{config['filename']}_bar.png"

    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)

    return pdf_path


# ===============================================================
# PLOT 2: MORRIS MU-STAR VS SIGMA SCATTER
# ===============================================================

def plot_morris_mu_star_sigma(df: pd.DataFrame, config: dict) -> Path | None:
    part = prepare_kpi_data(df, config)

    if part.empty:
        print(f"Skipping KPI not found in CSV: {config['kpi']}")
        return None

    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    ax.set_axisbelow(True)

    variant = "V3"
    variant_data = part[part["variant"] == variant].copy()

    x = variant_data["mu_star_plot"].to_numpy(dtype=float)
    y = variant_data["sigma_plot"].to_numpy(dtype=float)

    max_x = float(np.nanmax(x)) if len(x) else 0.0
    max_y = float(np.nanmax(y)) if len(y) else 0.0

    x_max = nice_max(max_x)
    y_max = nice_max(max_y)

    ax.scatter(
        x,
        y,
        color=VARIANT_COLORS[variant],
        marker=VARIANT_MARKERS.get(variant, "s"),
        s=52,
        alpha=0.92,
        edgecolors=COLORS["edge"],
        linewidths=0.6,
        zorder=2,
    )

# -----------------------------------------------------------
# Label only the most important / most separated parameters
# to avoid clutter near the origin
# -----------------------------------------------------------
# -----------------------------------------------------------
# Label only the most important parameters and manually offset
# labels to avoid overlap
# -----------------------------------------------------------
    label_data = variant_data.copy()

    label_data["_label_score"] = (
        label_data["mu_star_plot"] / max_x
        + label_data["sigma_plot"] / max_y
    )

    label_data = (
        label_data
        .sort_values("_label_score", ascending=False)
        .head(SCATTER_LABEL_TOP_N)
    )

    # Manual label offsets in points: (x_offset, y_offset, horizontal_alignment)
    label_offsets = {
        "Base ACH": (10, 6, "left"),
        "YOC shift": (-8, 10, "right"),
        "Mean setpoint": (-8, 18, "right"),
        "Shading factor": (-10, 28, "right"),
        "WWR factor": (8, 8, "left"),
    }

    for _, row in label_data.iterrows():
        x_val = row["mu_star_plot"]
        y_val = row["sigma_plot"]
        label = row["param_label"]

        if not np.isfinite(x_val) or not np.isfinite(y_val):
            continue

        dx, dy, ha = label_offsets.get(label, (8, 8, "left"))

        ax.annotate(
            label,
            xy=(x_val, y_val),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va="bottom",
            fontsize=8,
            color=COLORS["text"],
            zorder=3,
        )

    ax.set_xlabel(config["scatter_xlabel"])
    ax.set_ylabel(config["scatter_ylabel"])

    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)

    if SHOW_TITLES:
        ax.set_title(config["title"], pad=6)

    clean_axes(ax, grid_axis="both")

    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_path = OUTPUT_DIR / f"{config['filename']}_mu_star_sigma.pdf"
    png_path = OUTPUT_DIR / f"{config['filename']}_mu_star_sigma.png"

    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)

    return pdf_path


# ===============================================================
# MAIN
# ===============================================================

def main() -> None:
    apply_thesis_style()

    df = load_morris_indices(INPUT_CSV)

    print(f"Loaded {len(df):,} Morris-index rows from:")
    print(INPUT_CSV)

    print("\nVariants used:")
    print(", ".join(sorted(df["variant"].unique())))

    print("\nKPIs available in CSV:")
    print(", ".join(sorted(df["kpi"].unique())))

    created_paths: list[Path] = []

    for config in KPI_CONFIGS:
        if CREATE_MU_STAR_BAR:
            path = plot_morris_mu_star_bar(df, config)
            if path is not None:
                created_paths.append(path)

        if CREATE_MU_STAR_SIGMA_SCATTER:
            path = plot_morris_mu_star_sigma(df, config)
            if path is not None:
                created_paths.append(path)

    print("\nCreated figures:")
    for path in created_paths:
        print(path)


if __name__ == "__main__":
    main()