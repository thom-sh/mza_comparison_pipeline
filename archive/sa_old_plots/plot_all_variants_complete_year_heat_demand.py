#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================
# SCRIPT_DIR = Path(__file__).resolve().parent

# # Current pipeline folders
# RESULTS_ROOT = (SCRIPT_DIR / "../new_plots/sa_results").resolve()
# OUTPUT_DIR = (SCRIPT_DIR / "../new_plots/output/variant_profiles").resolve()
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_ROOT = Path(r"C:\WF\Thomas Sharon\Floorplan_Dataset\Sensitivity Analysis\results\sa_results")
OUTPUT_DIR = Path(r"C:\Sharon\mza_sensitivity_analysis\plotting_scripts\new_plots\output\variant_profiles")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESAMPLE_RULE = "3D"   # "12H", "1D", "2D", "3D", "7D"

VARIANT_ORDER = ["V1", "V1_20HH", "V2", "V2_20HH", "V3", "V4", "V5", "V6", "V7", "V8"]
SELECTED_VARIANTS = ["V1", "V1_20HH", "V2", "V2_20HH", "V3", "V4", "V5", "V6", "V7", "V8"]

# Optional filters
BUILDING_ID_CONTAINS = None      # e.g. "DESHPDHK0000lce9"
WEATHER_FILTER = None            # e.g. "TRY_A" or "TRY_B"

# If True: average seeds first for each sample, then take median across samples.
# If False: each seed/run is treated as an individual profile.
AGGREGATE_SEEDS_FIRST = False

DPI = 300
SHOW_PLOTS = True
FIGSIZE = (15, 5.5)

OUTPUT_FILENAME = f"all_variants_complete_year_median_heat_demand_{RESAMPLE_RULE}.pdf"
OUTPUT_CSV_PREFIX = f"complete_year_{RESAMPLE_RULE}_heat_demand_profile"

VARIANT_STYLES: dict[str, dict[str, object]] = {
    "V1": {"color": "#4c78a8", "linestyle": "--", "linewidth": 1.1, "label": "V1"},
    "V2": {"color": "#f58518", "linestyle": "-.", "linewidth": 1.1, "label": "V2"},
    "V3": {"color": "#54a24b", "linestyle": "-", "linewidth": 1.2, "label": "V3"},
    "V4": {"color": "#e45756", "linestyle": (0, (7, 1)), "linewidth": 1.1, "label": "V4"},
    "V5": {"color": "#72b7b2", "linestyle": ":", "linewidth": 1.3, "label": "V5"},
    "V6": {"color": "#b279a2", "linestyle": (0, (5, 2)), "linewidth": 1.1, "label": "V6"},
    "V7": {"color": "#ff9da6", "linestyle": (0, (3, 1, 1, 1)), "linewidth": 1.1, "label": "V7"},
    "V8": {"color": "#9d755d", "linestyle": (0, (7, 2, 1, 2)), "linewidth": 1.1, "label": "V8"},
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 9,
    "legend.fontsize": 7,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})
# ============================================================


class RunFiles:
    def __init__(self, run_dir: Path, timeseries_csv: Path, overall_json: Optional[Path]):
        self.run_dir = run_dir
        self.timeseries_csv = timeseries_csv
        self.overall_json = overall_json


def load_json(path: Optional[Path]) -> dict:
    if path is None or not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_datetime_index(df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    year = int(metadata.get("year", 2021))
    base = pd.Timestamp(f"{year}-01-01 00:00:00")
    if "time_s" not in df.columns:
        raise KeyError("Column 'time_s' is missing in timeseries.csv")

    out = df.copy()
    out["datetime"] = base + pd.to_timedelta(out["time_s"], unit="s")
    return out


def parse_run_from_path(ts_csv: Path) -> dict:
    parts = list(ts_csv.parts)
    sample_idx = None
    seed_idx = None

    for i, part in enumerate(parts):
        if re.fullmatch(r"sample_\d+", part):
            sample_idx = i
        if re.fullmatch(r"seed_\d+", part):
            seed_idx = i

    if sample_idx is None or seed_idx is None:
        raise ValueError(f"Could not parse sample/seed from path: {ts_csv}")

    if seed_idx != sample_idx + 1 or sample_idx < 3:
        raise ValueError(f"Unexpected results folder structure: {ts_csv}")

    variant = parts[sample_idx - 3]
    building_id = parts[sample_idx - 2]
    weather_key = parts[sample_idx - 1]
    sample_id = int(parts[sample_idx].split("_")[-1])
    seed = int(parts[seed_idx].split("_")[-1])

    return {
        "variant": variant,
        "building_id": building_id,
        "weather_key": weather_key,
        "sample_id": sample_id,
        "seed": seed,
    }


def base_building_group_key(building_id: str) -> str:
    s = str(building_id).strip()
    m = re.match(r"^(.*?)(?:[_-]?)(\d+)$", s)
    if not m:
        return s
    prefix = str(m.group(1)).strip()
    return prefix or s


def safe_series(df: pd.DataFrame, column: str) -> Optional[pd.Series]:
    if column not in df.columns:
        return None
    s = pd.to_numeric(df[column], errors="coerce")
    if s.isna().all():
        return None
    return s.astype(float)


def available_zone_numbers(df: pd.DataFrame) -> list[int]:
    nums: list[int] = []
    for col in df.columns:
        m = re.fullmatch(r"HeatDemand_(\d+)", col)
        if m:
            nums.append(int(m.group(1)))
    return sorted(set(nums))


def total_heat_kW(df: pd.DataFrame) -> pd.Series:
    """Return total heating demand in kW.

    Preference:
    1. Sum all HeatDemand_<zone> columns.
    2. Fall back to HeatDemand_SUM.
    3. Fall back to HeatDemand.

    Values are assumed to be W and are converted to kW.
    """
    zone_numbers = available_zone_numbers(df)

    if zone_numbers:
        total = pd.Series(0.0, index=df.index, dtype=float)
        found = False
        for z in zone_numbers:
            s = safe_series(df, f"HeatDemand_{z}")
            if s is not None:
                total = total.add(s, fill_value=0.0)
                found = True
        if found:
            return total / 1000.0

    for fallback_col in ["HeatDemand_SUM", "HeatDemand"]:
        s = safe_series(df, fallback_col)
        if s is not None:
            return s / 1000.0

    raise RuntimeError(
        "No heating demand column found. Expected HeatDemand_<zone>, HeatDemand_SUM, or HeatDemand."
    )


def discover_variant_runs(
    results_root: Path,
    variant: str,
    building_id_contains: Optional[str] = None,
    weather_filter: Optional[str] = None,
) -> list[RunFiles]:
    variant_dir = results_root / variant
    if not variant_dir.exists():
        return []

    runs: list[RunFiles] = []
    for ts_csv in sorted(variant_dir.rglob("timeseries.csv")):
        parts = ts_csv.parts

        if building_id_contains is not None and building_id_contains not in parts:
            continue
        if weather_filter is not None and weather_filter not in parts:
            continue

        run_dir = ts_csv.parent
        runs.append(
            RunFiles(
                run_dir=run_dir,
                timeseries_csv=ts_csv,
                overall_json=run_dir / "overall.json" if (run_dir / "overall.json").exists() else None,
            )
        )

    return runs


def load_run_profile(run_files: RunFiles) -> pd.DataFrame:
    meta_from_path = parse_run_from_path(run_files.timeseries_csv)
    overall = load_json(run_files.overall_json)

    df = pd.read_csv(run_files.timeseries_csv)
    df = build_datetime_index(df, overall)

    out = pd.DataFrame({
        "datetime": df["datetime"],
        "heat_kW": total_heat_kW(df),
    })

    out["variant"] = meta_from_path["variant"]
    out["building_id"] = meta_from_path["building_id"]
    out["base_building_group"] = base_building_group_key(meta_from_path["building_id"])
    out["weather_key"] = meta_from_path["weather_key"]
    out["sample_id"] = meta_from_path["sample_id"]
    out["seed"] = meta_from_path["seed"]
    return out


def build_profiles_table(runs: list[RunFiles]) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []

    for i, rf in enumerate(runs, start=1):
        pieces.append(load_run_profile(rf))

        if i % 50 == 0 or i == len(runs):
            print(f"Loaded {i}/{len(runs)} runs")

    if not pieces:
        return pd.DataFrame()

    return pd.concat(pieces, ignore_index=True)


def aggregate_across_seeds_first(df_profiles: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["variant", "base_building_group", "weather_key", "sample_id", "datetime"]
    return (
        df_profiles
        .groupby(group_cols, dropna=False)["heat_kW"]
        .mean()
        .reset_index(name="heat_kW")
    )


def compute_variant_median_profile(df_profiles: pd.DataFrame) -> pd.DataFrame:
    if AGGREGATE_SEEDS_FIRST:
        df_work = aggregate_across_seeds_first(df_profiles)
    else:
        df_work = df_profiles[["datetime", "heat_kW"]].copy()

    med = (
        df_work
        .groupby("datetime", dropna=False)["heat_kW"]
        .median()
        .reset_index(name="median_heat_kW")
        .sort_values("datetime")
    )

    if RESAMPLE_RULE is not None:
        med = (
            med
            .set_index("datetime")
            .resample(RESAMPLE_RULE)["median_heat_kW"]
            .mean()
            .reset_index()
        )

    return med


def savefig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)


def format_year_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonthday=15))


def plot_complete_year_profiles(median_by_variant: dict[str, pd.DataFrame]) -> Path:
    fig, ax = plt.subplots(figsize=FIGSIZE)

    plotted_any = False
    start_dates: list[pd.Timestamp] = []
    end_dates: list[pd.Timestamp] = []

    for variant in SELECTED_VARIANTS:
        med_df = median_by_variant.get(variant)

        if med_df is None or med_df.empty:
            continue

        style = VARIANT_STYLES.get(variant, {})
        ax.plot(
            med_df["datetime"],
            med_df["median_heat_kW"],
            color=style.get("color", None),
            linestyle=style.get("linestyle", "-"),
            linewidth=float(style.get("linewidth", 1.1)),
            label=str(style.get("label", variant)),
        )

        start_dates.append(pd.to_datetime(med_df["datetime"].min()))
        end_dates.append(pd.to_datetime(med_df["datetime"].max()))
        plotted_any = True

    if not plotted_any:
        raise RuntimeError("No profiles available for plotting.")

    ax.set_ylabel("Heating demand [kW]")

    ax.set_xlim(min(start_dates), max(end_dates))
    style_axis(ax)
    format_year_axis(ax)

    ax.legend(
        loc="upper right",
        frameon=True,
        ncol=2,
    )

    out_path = OUTPUT_DIR / OUTPUT_FILENAME
    savefig(fig, out_path)
    return out_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    median_by_variant: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict[str, object]] = []

    for variant in VARIANT_ORDER:
        if variant not in SELECTED_VARIANTS:
            continue

        runs = discover_variant_runs(
            results_root=RESULTS_ROOT,
            variant=variant,
            building_id_contains=BUILDING_ID_CONTAINS,
            weather_filter=WEATHER_FILTER,
        )

        if not runs:
            print(f"[INFO] No runs found for {variant}")
            continue

        print(f"[INFO] Processing {variant}: {len(runs)} runs found")
        df_profiles = build_profiles_table(runs)

        if df_profiles.empty:
            print(f"[INFO] Empty profile table for {variant}")
            continue

        med_df = compute_variant_median_profile(df_profiles)
        median_by_variant[variant] = med_df

        med_csv = OUTPUT_DIR / f"{OUTPUT_CSV_PREFIX}_{variant}_median_heat_demand_profile.csv"
        med_df.to_csv(med_csv, index=False)

        n_runs = int(
            df_profiles[["base_building_group", "weather_key", "sample_id", "seed"]]
            .drop_duplicates()
            .shape[0]
        )

        if AGGREGATE_SEEDS_FIRST:
            n_profiles_used = int(
                df_profiles[["base_building_group", "weather_key", "sample_id"]]
                .drop_duplicates()
                .shape[0]
            )
        else:
            n_profiles_used = n_runs

        summary_rows.append({
            "variant": variant,
            "n_runs": n_runs,
            "n_profiles_used": n_profiles_used,
            "n_timesteps": int(len(med_df)),
            "median_profile_csv": str(med_csv),
        })

    if not median_by_variant:
        raise RuntimeError("No variant median profiles could be built.")

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = OUTPUT_DIR / f"{OUTPUT_CSV_PREFIX}_all_variants_heat_demand_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    fig_path = plot_complete_year_profiles(median_by_variant)

    print("Saved:")
    print(summary_csv)
    for row in summary_rows:
        print(row["median_profile_csv"])
    print(fig_path)


if __name__ == "__main__":
    main()
