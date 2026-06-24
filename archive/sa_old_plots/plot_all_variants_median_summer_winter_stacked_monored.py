#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = (SCRIPT_DIR / "../results/sa_results").resolve()

VARIANT_ORDER = ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8"]
SELECTED_VARIANTS = ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8"]
BUILDING_ID_CONTAINS = None
WEATHER_FILTER = None
AGGREGATE_SEEDS_FIRST = True

DPI = 150
SHOW_PLOTS = True
FIGSIZE = (14, 4.8)

OUTPUT_DIR = RESULTS_ROOT / "_analysis_plots_fixed" / "variant_profiles"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# EDIT THESE TIME RANGES
# ------------------------------------------------------------
SEASON_SPECS: list[tuple[str, str, str]] = [
    ("winter", "2021-01-01 00:00:00", "2021-01-02 23:00:00"),
    ("summer", "2021-06-01 00:00:00", "2021-08-31 23:00:00"),
]
# ------------------------------------------------------------

# Single-hue stacked palette inspired by the reference screenshot.
# Bottom band darkest, upper bands progressively lighter.
STACK_FILL_COLORS_LOW_TO_HIGH = [
    "#A63A3A",
    "#B24A4A",
    "#C85F5F",
    "#DB7E7E",
    "#F0A4A4",
    "#D7D7D7",
    "#E5E5E5",
    "#F0F0F0",
]

COMMON_LINE_COLOR = "#7A2323"
COMMON_LINE_WIDTH = 0.01
COMMON_FILL_ALPHA = 0.88

VARIANT_STYLES: dict[str, dict[str, object]] = {
    "V1": {"linestyle": "--", "linewidth": COMMON_LINE_WIDTH, "label": "1 Zone"},
    "V2": {"linestyle": "-.", "linewidth": COMMON_LINE_WIDTH, "label": "1/Storey + SC"},
    "V3": {"linestyle": "-", "linewidth": COMMON_LINE_WIDTH, "label": "2/Fl. A + SC"},
    "V4": {"linestyle": (0, (7, 1)), "linewidth": COMMON_LINE_WIDTH, "label": "2/Fl. B + SC"},
    "V5": {"linestyle": ":", "linewidth": COMMON_LINE_WIDTH, "label": "4/Fl. A + SC"},
    "V6": {"linestyle": (0, (5, 2)), "linewidth": COMMON_LINE_WIDTH, "label": "4/Fl. B + SC"},
    "V7": {"linestyle": (0, (3, 1, 1, 1)), "linewidth": COMMON_LINE_WIDTH, "label": "2/Fl. A no SC"},
    "V8": {"linestyle": (0, (7, 2, 1, 2)), "linewidth": COMMON_LINE_WIDTH, "label": "2/Fl. A heated SC"},
}


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
    zone_numbers = available_zone_numbers(df)
    if not zone_numbers:
        raise RuntimeError("No HeatDemand_<zone> columns found.")

    total = pd.Series(0.0, index=df.index, dtype=float)
    found = False
    for z in zone_numbers:
        s = safe_series(df, f"HeatDemand_{z}")
        if s is not None:
            total = total.add(s, fill_value=0.0)
            found = True

    if not found:
        raise RuntimeError("Could not sum any HeatDemand_<zone> columns.")

    return total / 1000.0


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
    return pd.concat(pieces, ignore_index=True)


def aggregate_across_seeds_first(df_profiles: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["base_building_group", "weather_key", "sample_id", "datetime"]
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
    ax.tick_params(axis="x", rotation=0)


def format_time_axis(ax: plt.Axes, start: pd.Timestamp, end: pd.Timestamp) -> None:
    duration_days = (end - start).total_seconds() / 86400.0

    if duration_days <= 7:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    elif duration_days <= 35:
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    else:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))


def _color_for_band(band_idx: int, n_bands: int) -> str:
    if n_bands <= 1:
        return STACK_FILL_COLORS_LOW_TO_HIGH[0]
    palette_idx = int(round(band_idx * (len(STACK_FILL_COLORS_LOW_TO_HIGH) - 1) / (n_bands - 1)))
    return STACK_FILL_COLORS_LOW_TO_HIGH[palette_idx]


def plot_season_overlay(
    median_by_variant: dict[str, pd.DataFrame],
    label: str,
    start_str: str,
    end_str: str,
) -> Path:
    fig, ax = plt.subplots(figsize=FIGSIZE)

    start = pd.Timestamp(start_str)
    end = pd.Timestamp(end_str)

    window_frames: list[dict[str, object]] = []
    for variant in SELECTED_VARIANTS:
        med_df = median_by_variant.get(variant)
        if med_df is None or med_df.empty:
            continue

        plot_df = med_df.loc[(med_df["datetime"] >= start) & (med_df["datetime"] <= end), ["datetime", "median_heat_kW"]].copy()
        if plot_df.empty:
            continue

        plot_df = plot_df.sort_values("datetime").reset_index(drop=True)
        window_frames.append({
            "variant": variant,
            "df": plot_df,
            "mean_kW": float(plot_df["median_heat_kW"].mean()),
        })

    if not window_frames:
        raise RuntimeError(f"No data found for plot window: {label}")

    # Rank curves from bottom to top based on mean load in this window.
    ranked = sorted(window_frames, key=lambda item: float(item["mean_kW"]))

    # Use the lowest curve's datetime as the common x-axis. Profiles are expected
    # to share the same timestamps after median aggregation.
    x = ranked[0]["df"]["datetime"]
    y_series = [item["df"]["median_heat_kW"].to_numpy() for item in ranked]

    # Stacked visual bands: 0->lowest, then between adjacent ranked curves.
    n_bands = len(y_series)
    lower = 0.0
    for band_idx, upper in enumerate(y_series):
        fill_color = _color_for_band(band_idx, n_bands)
        ax.fill_between(
            x,
            lower,
            upper,
            color=fill_color,
            alpha=COMMON_FILL_ALPHA,
            linewidth=0.0,
            zorder=1 + band_idx,
        )
        lower = upper

    # Plot all variant curves on top with same color, different linestyles.
    for rank, item in enumerate(ranked):
        variant = str(item["variant"])
        plot_df = item["df"]
        style = VARIANT_STYLES.get(variant, {})
        ax.plot(
            plot_df["datetime"],
            plot_df["median_heat_kW"],
            label=str(style.get("label", variant)),
            color=COMMON_LINE_COLOR,
            linestyle=style.get("linestyle", "-"),
            linewidth=float(style.get("linewidth", COMMON_LINE_WIDTH)),
            zorder=20 + rank,
        )

    ax.set_title(f"Median heating demand across variants – {label}")
    ax.set_xlabel("")
    ax.set_ylabel("Heating Demand [kW]")
    ax.set_xlim(start, end)
    style_axis(ax)
    format_time_axis(ax, start, end)
    ax.legend(loc="upper right", frameon=True, ncol=1)

    out_path = OUTPUT_DIR / f"all_variants_median_overlay_{label}_stacked_monored.png"
    savefig(fig, out_path)
    return out_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    median_by_variant: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict[str, object]] = []

    for variant in VARIANT_ORDER:
        runs = discover_variant_runs(
            results_root=RESULTS_ROOT,
            variant=variant,
            building_id_contains=BUILDING_ID_CONTAINS,
            weather_filter=WEATHER_FILTER,
        )
        if not runs:
            print(f"[INFO] No runs found for {variant}")
            continue

        df_profiles = build_profiles_table(runs)
        med_df = compute_variant_median_profile(df_profiles)
        median_by_variant[variant] = med_df
        med_df.to_csv(OUTPUT_DIR / f"{variant}_median_profile.csv", index=False)

        if AGGREGATE_SEEDS_FIRST:
            df_count = aggregate_across_seeds_first(df_profiles)
            n_profiles = int(df_count[["base_building_group", "weather_key", "sample_id"]].drop_duplicates().shape[0])
        else:
            n_profiles = int(df_profiles[["base_building_group", "weather_key", "sample_id", "seed"]].drop_duplicates().shape[0])

        summary_rows.append({
            "variant": variant,
            "n_runs": int(df_profiles[["base_building_group", "weather_key", "sample_id", "seed"]].drop_duplicates().shape[0]),
            "n_profiles_used": n_profiles,
            "n_timesteps": int(len(med_df)),
        })

    if not median_by_variant:
        raise RuntimeError("No variant median profiles could be built.")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_DIR / "all_variants_median_overlay_summary.csv", index=False)

    saved = []
    for label, start_str, end_str in SEASON_SPECS:
        saved.append(plot_season_overlay(median_by_variant, label, start_str, end_str))

    print("Saved median CSVs per variant and figures:")
    for variant in median_by_variant:
        print(OUTPUT_DIR / f"{variant}_median_profile.csv")
    print(OUTPUT_DIR / "all_variants_median_overlay_summary.csv")
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()
