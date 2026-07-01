# -*- coding: utf-8 -*-
"""
Create and plot Sobol indices for overheating hours.

This script:
1) Reads sobol_runs_collected.csv.
2) Adds an overheating-hours KPI by reading each run's timeseries.csv.
3) Computes Sobol S1 and ST indices using the existing A, B, and AB block structure.
4) Creates thesis-style orange Sobol plots.

Overheating definition:
    any-zone overheating hours = hours where at least one TAir_* column exceeds 26 °C.

Required columns in sobol_runs_collected.csv:
    variant, building_id, weather_key, seed, block, row_id, mix_col

Optional column:
    overall_json

Expected run folder:
    row_0000/
        overall.json
        timeseries.csv
        run_config.json
        zone_map.json
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

SOBOL_ROOT = (
    PROJECT_DIR
    / "sa_results"
    / "sa_step4_sobol"
)

RUNS_CSV = SOBOL_ROOT / "sobol_runs_collected_1.csv"

OUTPUT_DIR = PROJECT_DIR / "output" / "figures_sobol_thesis_style"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_RUNS_WITH_OVERHEATING_CSV = (
    SOBOL_ROOT / "sobol_runs_collected_with_overheating_hours.csv"
)

OUT_INDICES_CSV = (
    SOBOL_ROOT / "sobol_indices_overheating_hours.csv"
)

OUT_WEATHER_COMPARISON_CSV = (
    SOBOL_ROOT / "sobol_weather_comparison_ST_overheating_hours.csv"
)

DPI = 300

# Only V3
VARIANT_ORDER = ["V3"]

WEATHER_ORDER = ["TRY_A", "TRY_B"]

WEATHER_LABELS = {
    "TRY_A": "Napoli",
    "TRY_B": "Munich",
}

# Final KPI name used in the output CSV
OVERHEATING_KPI = "overheating_hours"

# Overheating definition
OVERHEATING_THRESHOLD_C = 26.0
OVERHEATING_MODE = "any_zone"  # "any_zone" or "zone_hours"

# Sobol parameter order used in your existing Sobol setup
PARAM_ORDER = [
    "baseACH",
    "yoc_shift",
    "tset_mean_C",
    "shadingFactor",
    "gWin",
    "gains_scale",
]

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

# Plot settings
FIXED_XMAX = 1.0
TOP_N_PARAMS = None
SHOW_TITLES = False
SHOW_VALUE_LABELS = False


# ===============================================================
# THESIS STYLE
# ===============================================================

COLORS = {
    "S1": "#F1D7B5",
    "ST": "#D9A36A",
    "TRY_A": "#F1D7B5",
    "TRY_B": "#D9A36A",
    "edge": "#777D84",
    "grid": "#D9D9D9",
    "text": "#000000",
    "legend_edge": "#BDC1C5",
}


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
# PATH HANDLING
# ===============================================================

def rebuild_run_folder(row: pd.Series) -> Path:
    """Rebuild the Sobol run folder from metadata."""
    variant = str(row["variant"])
    building_id = str(row["building_id"])
    weather_key = str(row["weather_key"])
    seed = int(row["seed"])
    block = str(row["block"])
    row_id = int(row["row_id"])

    row_folder = f"row_{row_id:04d}"

    if block in ["A", "B"]:
        return (
            SOBOL_ROOT
            / variant
            / building_id
            / weather_key
            / f"seed_{seed}"
            / block
            / "base"
            / row_folder
        )

    if block == "AB":
        mix_col = int(row["mix_col"])
        return (
            SOBOL_ROOT
            / variant
            / building_id
            / weather_key
            / f"seed_{seed}"
            / block
            / f"col_{mix_col:02d}"
            / row_folder
        )

    raise ValueError(f"Unknown block type: {block}")


def resolve_run_folder(row: pd.Series) -> Path:
    """
    Resolve run folder.

    If overall_json exists in the CSV, use its parent folder.
    Otherwise rebuild the folder path from Sobol metadata.
    """
    if "overall_json" in row.index:
        raw_path = Path(str(row["overall_json"]))
        if raw_path.exists():
            return raw_path.parent

    rebuilt = rebuild_run_folder(row)
    if rebuilt.exists():
        return rebuilt

    raise FileNotFoundError(
        "Could not find Sobol run folder.\n"
        f"Rebuilt path tried:\n{rebuilt}\n\n"
        "Check SOBOL_ROOT and folder structure."
    )


# ===============================================================
# OVERHEATING CALCULATION FROM TIMESERIES
# ===============================================================

def calculate_overheating_hours_from_timeseries(timeseries_csv: Path) -> float:
    """
    Calculate overheating hours from timeseries.csv.

    The script automatically detects whether TAir_* columns are in Kelvin or Celsius.

    Definition:
        any_zone   = hours where at least one zone exceeds 26 °C
        zone_hours = summed overheating hours across all zones
    """

    if not timeseries_csv.exists():
        raise FileNotFoundError(f"Cannot find timeseries file:\n{timeseries_csv}")

    df = pd.read_csv(timeseries_csv)

    tair_cols = [
        col for col in df.columns
        if col.startswith("TAir_")
    ]

    if not tair_cols:
        raise ValueError(
            f"No zone air-temperature columns found in:\n{timeseries_csv}\n"
            "Expected columns like TAir_1, TAir_2, ..."
        )

    tair = df[tair_cols].apply(pd.to_numeric, errors="coerce")

    # -----------------------------------------------------------
    # Detect unit
    # Modelica outputs are often in Kelvin.
    # If typical values are > 100, assume Kelvin and convert to °C.
    # -----------------------------------------------------------
    median_temp = np.nanmedian(tair.to_numpy())

    if median_temp > 100:
        tair_c = tair - 273.15
    else:
        tair_c = tair

    # -----------------------------------------------------------
    # Determine timestep in hours
    # -----------------------------------------------------------
    if "time_s" in df.columns:
        time_s = pd.to_numeric(df["time_s"], errors="coerce")

        valid_time = time_s.dropna()

        if len(valid_time) > 1:
            dt_hours = float(valid_time.diff().dropna().median() / 3600.0)

            # If the time series includes both start and end point,
            # use intervals instead of counting the final endpoint.
            expected_intervals = int(round((valid_time.iloc[-1] - valid_time.iloc[0]) / (dt_hours * 3600.0)))

            if len(df) == expected_intervals + 1:
                tair_c = tair_c.iloc[:-1]
        else:
            dt_hours = 1.0
    else:
        dt_hours = 1.0

    # -----------------------------------------------------------
    # Overheating calculation
    # -----------------------------------------------------------
    exceedance = tair_c > OVERHEATING_THRESHOLD_C

    if OVERHEATING_MODE == "any_zone":
        overheating_hours = exceedance.any(axis=1).sum() * dt_hours

    elif OVERHEATING_MODE == "zone_hours":
        overheating_hours = exceedance.sum(axis=1).sum() * dt_hours

    else:
        raise ValueError(
            "Invalid OVERHEATING_MODE. Use 'any_zone' or 'zone_hours'."
        )

    return float(overheating_hours)


def add_overheating_hours(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add overheating-hours column by reading timeseries.csv in each Sobol run folder.
    """

    df = df.copy()

    if OVERHEATING_KPI in df.columns:
        print(f"Using existing column: {OVERHEATING_KPI}")
        df[OVERHEATING_KPI] = pd.to_numeric(df[OVERHEATING_KPI], errors="coerce")
        return df

    values = []

    print("Reading overheating hours from timeseries.csv files...")

    for i, row in df.iterrows():
        run_folder = resolve_run_folder(row)
        timeseries_csv = run_folder / "timeseries.csv"

        value = calculate_overheating_hours_from_timeseries(timeseries_csv)
        values.append(value)

        if len(values) % 250 == 0:
            print(f"  processed {len(values)} / {len(df)}")

    df[OVERHEATING_KPI] = values

    return df


# ===============================================================
# SOBOL COMPUTATION
# ===============================================================

def compute_sobol_indices(df: pd.DataFrame, kpi_col: str) -> pd.DataFrame:
    rows = []

    group_cols = ["variant", "building_id", "weather_key", "seed"]

    skipped_incomplete = 0
    skipped_zero_variance = 0
    skipped_missing_ab = 0

    for group_key, group in df.groupby(group_cols):
        variant, building_id, weather_key, seed = group_key

        A = (
            group[group["block"] == "A"]
            [["row_id", kpi_col]]
            .rename(columns={kpi_col: "A"})
        )

        B = (
            group[group["block"] == "B"]
            [["row_id", kpi_col]]
            .rename(columns={kpi_col: "B"})
        )

        AB_all = group[group["block"] == "AB"].copy()

        if A.empty or B.empty or AB_all.empty:
            skipped_incomplete += 1
            print(f"Skipping incomplete group: {group_key}")
            print(f"  A rows: {len(A)}, B rows: {len(B)}, AB rows: {len(AB_all)}")
            continue

        a_values = A["A"].to_numpy(dtype=float)
        b_values = B["B"].to_numpy(dtype=float)

        variance = np.var(
            pd.concat([A["A"], B["B"]], ignore_index=True).to_numpy(dtype=float),
            ddof=1,
        )

        if not np.isfinite(variance) or variance <= 0:
            skipped_zero_variance += 1
            print(f"Skipping zero-variance group: {group_key}")
            print(f"  A min/max: {np.nanmin(a_values)} / {np.nanmax(a_values)}")
            print(f"  B min/max: {np.nanmin(b_values)} / {np.nanmax(b_values)}")
            continue

        for param_index, param in enumerate(PARAM_ORDER):
            AB = (
                AB_all[AB_all["mix_col"] == param_index]
                [["row_id", kpi_col]]
                .rename(columns={kpi_col: "AB"})
            )

            merged = (
                A.merge(B, on="row_id", how="inner")
                .merge(AB, on="row_id", how="inner")
                .dropna(subset=["A", "B", "AB"])
            )

            if merged.empty:
                skipped_missing_ab += 1
                print(f"Skipping missing AB merge for group: {group_key}")
                print(f"  param_index: {param_index}, param: {param}")
                print(f"  A row_id sample: {A['row_id'].head().tolist()}")
                print(f"  B row_id sample: {B['row_id'].head().tolist()}")
                print(f"  AB row_id sample: {AB['row_id'].head().tolist()}")
                continue

            y_a = merged["A"].to_numpy(dtype=float)
            y_b = merged["B"].to_numpy(dtype=float)
            y_ab = merged["AB"].to_numpy(dtype=float)

            S1 = np.mean(y_b * (y_ab - y_a)) / variance
            ST = 0.5 * np.mean((y_a - y_ab) ** 2) / variance

            rows.append({
                "variant": variant,
                "building_id": building_id,
                "weather_key": weather_key,
                "seed": seed,
                "kpi": kpi_col,
                "param": param,
                "param_index": param_index,
                "S1": S1,
                "ST": ST,
                "interaction_gap": ST - S1,
                "n_rows": len(merged),
                "variance": variance,
            })

    indices = pd.DataFrame(rows)

    print("\nSobol computation summary:")
    print(f"  computed rows: {len(indices)}")
    print(f"  skipped incomplete groups: {skipped_incomplete}")
    print(f"  skipped zero-variance groups: {skipped_zero_variance}")
    print(f"  skipped missing AB merges: {skipped_missing_ab}")

    if indices.empty:
        raise ValueError(
            "No Sobol indices were computed. "
            "Check the printed summary above. Most likely the overheating KPI has zero variance."
        )

    indices["rank_S1"] = (
        indices.groupby(["variant", "building_id", "weather_key", "seed", "kpi"])["S1"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    indices["rank_ST"] = (
        indices.groupby(["variant", "building_id", "weather_key", "seed", "kpi"])["ST"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    return indices

# ===============================================================
# PLOTS
# ===============================================================

def prepare_plot_data(indices: pd.DataFrame, weather_key: str) -> pd.DataFrame:
    part = indices[indices["weather_key"] == weather_key].copy()

    if part.empty:
        return part

    part["S1"] = pd.to_numeric(part["S1"], errors="coerce")
    part["ST"] = pd.to_numeric(part["ST"], errors="coerce")
    part["param_label"] = part["param"].map(PARAM_LABELS).fillna(part["param"])

    part = (
        part.groupby(["param", "param_label"], as_index=False)
        .agg(S1=("S1", "mean"), ST=("ST", "mean"))
    )

    part = part.sort_values("ST", ascending=False)

    if TOP_N_PARAMS is not None:
        part = part.head(TOP_N_PARAMS)

    return part


def plot_sobol_s1_st(indices: pd.DataFrame, weather_key: str) -> Path | None:
    part = prepare_plot_data(indices, weather_key)

    if part.empty:
        print(f"No data for weather case: {weather_key}")
        return None

    param_order = part["param_label"].tolist()
    y = np.arange(len(param_order))
    bar_height = 0.24

    fig_height = max(2.8, 0.34 * len(param_order) + 0.75)
    fig, ax = plt.subplots(figsize=(6.4, fig_height))
    ax.set_axisbelow(True)

    s1_values = part["S1"].to_numpy(dtype=float)
    st_values = part["ST"].to_numpy(dtype=float)

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

    ax.set_xlabel("Sobol sensitivity index [-]")

    if FIXED_XMAX is not None:
        ax.set_xlim(0, FIXED_XMAX)
    else:
        max_value = np.nanmax([np.nanmax(s1_values), np.nanmax(st_values)])
        ax.set_xlim(0, nice_max(max_value))

    if SHOW_TITLES:
        ax.set_title(f"Overheating hours - {WEATHER_LABELS.get(weather_key, weather_key)}")

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

    weather_label = WEATHER_LABELS.get(weather_key, weather_key).lower().replace(" ", "_")

    pdf_path = OUTPUT_DIR / f"sobol_v3_overheating_hours_{weather_label}_S1_ST.pdf"
    png_path = OUTPUT_DIR / f"sobol_v3_overheating_hours_{weather_label}_S1_ST.png"

    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)

    return pdf_path


def plot_weather_comparison_ST(indices: pd.DataFrame) -> Path | None:
    part = indices.copy()
    part["param_label"] = part["param"].map(PARAM_LABELS).fillna(part["param"])

    part = (
        part.groupby(["weather_key", "param", "param_label"], as_index=False)
        .agg(ST=("ST", "mean"))
    )

    pivot = part.pivot_table(
        index="param_label",
        columns="weather_key",
        values="ST",
        aggfunc="mean",
    )

    available_weather = [w for w in WEATHER_ORDER if w in pivot.columns]

    if len(available_weather) < 2:
        print("Skipping ST weather comparison because fewer than two weather cases are available.")
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

    for weather_key in available_weather[:2]:
        values = pivot[weather_key].to_numpy(dtype=float)

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
        max_value = pivot[available_weather].to_numpy(dtype=float).max()
        ax.set_xlim(0, nice_max(max_value))

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

    pdf_path = OUTPUT_DIR / "sobol_v3_overheating_hours_weather_comparison_ST.pdf"
    png_path = OUTPUT_DIR / "sobol_v3_overheating_hours_weather_comparison_ST.png"

    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)

    comparison = pivot[available_weather].reset_index()
    comparison.to_csv(OUT_WEATHER_COMPARISON_CSV, index=False)

    return pdf_path


# ===============================================================
# MAIN
# ===============================================================

def main() -> None:
    apply_thesis_style()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not RUNS_CSV.exists():
        raise FileNotFoundError(f"Cannot find RUNS_CSV:\n{RUNS_CSV}")

    df = pd.read_csv(RUNS_CSV)

    required = {
        "variant",
        "building_id",
        "weather_key",
        "seed",
        "block",
        "row_id",
        "mix_col",
    }

    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in RUNS_CSV: {missing}")

    df = df[df["variant"].isin(VARIANT_ORDER)].copy()
    df = df[df["weather_key"].isin(WEATHER_ORDER)].copy()

    print(f"Loaded {len(df):,} Sobol run rows.")

    df = add_overheating_hours(df)

    df.to_csv(OUT_RUNS_WITH_OVERHEATING_CSV, index=False)

    print("\nOverheating-hour summary:")
    print(
        df.groupby(["weather_key", "block"])[OVERHEATING_KPI]
        .agg(["count", "min", "max", "mean", "std"])
    )

    print("\nBlock counts:")
    print(
        df.groupby(["weather_key", "seed", "block"])
        .size()
        .reset_index(name="count")
    )

    print("\nAB mix_col counts:")
    print(
        df[df["block"] == "AB"]
        .groupby(["weather_key", "seed", "mix_col"])
        .size()
        .reset_index(name="count")
    )

    indices = compute_sobol_indices(df, OVERHEATING_KPI)
    indices.to_csv(OUT_INDICES_CSV, index=False)

    print("\nSobol indices for overheating hours:")
    print(indices[["weather_key", "param", "S1", "ST", "rank_ST"]].to_string(index=False))

    created_paths = []

    for weather_key in WEATHER_ORDER:
        path = plot_sobol_s1_st(indices, weather_key)
        if path is not None:
            created_paths.append(path)

    path = plot_weather_comparison_ST(indices)
    if path is not None:
        created_paths.append(path)

    print("\nSaved CSV files:")
    print(OUT_RUNS_WITH_OVERHEATING_CSV)
    print(OUT_INDICES_CSV)
    print(OUT_WEATHER_COMPARISON_CSV)

    print("\nCreated figures:")
    for path in created_paths:
        print(path)


if __name__ == "__main__":
    main()