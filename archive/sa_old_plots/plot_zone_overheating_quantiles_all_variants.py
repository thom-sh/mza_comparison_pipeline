#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = (SCRIPT_DIR / "../results/sa_results").resolve()

OUTPUT_DIR = RESULTS_ROOT / "_analysis_plots_fixed" / "zone_overheating_profiles_all_variants"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VARIANT_ORDER = ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8"]
BUILDING_ID_CONTAINS = None          # e.g. "DESHPDHK0000lce9"
WEATHER_FILTER = None                # e.g. "TRY_A" or "TRY_B"
AGGREGATE_SEEDS_FIRST = True         # average seeds first, then compute P5/P50/P95 across samples

OVERHEATING_THRESHOLD_C = 26.0
SHOW_PLOTS = True
DPI = 150
N_COLS = 2
FIG_WIDTH = 18
FIG_HEIGHT_PER_ROW = 4.2
SHARE_Y = False

# Note:
# - "median" and "p50" are numerically the same.
# - Both are exported because you explicitly asked for both before.
STAT_TO_COLUMN = {
    "median": "q50",
    "p05": "q05",
    "p50": "q50",
    "p95": "q95",
}

# Time windows to export.
# Winter is split into Jan-Feb and Dec of the same simulation year.
WINDOW_SPECS: dict[str, dict] = {
    "1_day": {"kind": "duration_days", "days": 1},
    "3_days": {"kind": "duration_days", "days": 3},
    "1_month": {"kind": "date_range", "start": "01-01 00:00:00", "end": "01-31 23:00:00"},
    "3_months": {"kind": "date_range", "start": "01-01 00:00:00", "end": "03-31 23:00:00"},
    "6_months": {"kind": "date_range", "start": "01-01 00:00:00", "end": "06-30 23:00:00"},
    "full_year": {"kind": "all"},
    "summer": {"kind": "date_range", "start": "06-01 00:00:00", "end": "08-31 23:00:00"},
    "winter": {
        "kind": "multi_range",
        "ranges": [
            ("01-01 00:00:00", "02-28 23:00:00"),
            ("12-01 00:00:00", "12-31 23:00:00"),
        ],
    },
}
# ============================================================


class RunFiles:
    def __init__(self, run_dir: Path, timeseries_csv: Path, zone_map_json: Optional[Path], overall_json: Optional[Path]):
        self.run_dir = run_dir
        self.timeseries_csv = timeseries_csv
        self.zone_map_json = zone_map_json
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


def parse_zone_map(zone_map_path: Optional[Path]) -> dict[int, str]:
    if zone_map_path is None or not zone_map_path.exists():
        return {}

    data = load_json(zone_map_path)
    mapping: dict[int, str] = {}

    if isinstance(data, dict):
        if "zone_map" in data and isinstance(data["zone_map"], dict):
            for k, v in data["zone_map"].items():
                try:
                    mapping[int(k)] = str(v)
                except Exception:
                    pass
            return mapping

        zones = data.get("zones")
        if isinstance(zones, list):
            for i, item in enumerate(zones, start=1):
                if isinstance(item, dict):
                    zno = item.get("index", item.get("zone_no", i))
                    zname = item.get("name", item.get("zone_name", f"Zone_{zno}"))
                    try:
                        mapping[int(zno)] = str(zname)
                    except Exception:
                        pass
                else:
                    mapping[i] = str(item)
        elif isinstance(zones, dict):
            for k, v in zones.items():
                try:
                    mapping[int(k)] = str(v)
                except Exception:
                    pass
    return mapping


def safe_series(df: pd.DataFrame, column: str) -> Optional[pd.Series]:
    if column not in df.columns:
        return None
    s = pd.to_numeric(df[column], errors="coerce")
    if s.isna().all():
        return None
    return s.astype(float)


def maybe_kelvin_to_celsius(series: Optional[pd.Series]) -> Optional[pd.Series]:
    if series is None:
        return None
    finite = series.dropna()
    if finite.empty:
        return series
    if finite.median() > 150:
        return series - 273.15
    return series


def infer_dt_hours(df: pd.DataFrame) -> float:
    if "time_s" not in df.columns or len(df) < 2:
        return 0.0
    dt_s = pd.to_numeric(df["time_s"], errors="coerce").diff().dropna().median()
    if pd.isna(dt_s):
        return 0.0
    return float(dt_s) / 3600.0


def available_zone_numbers(df: pd.DataFrame) -> list[int]:
    nums: list[int] = []
    for col in df.columns:
        m = re.fullmatch(r"TAir_(\d+)", col)
        if m:
            nums.append(int(m.group(1)))
    return sorted(set(nums))


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
                zone_map_json=run_dir / "zone_map.json" if (run_dir / "zone_map.json").exists() else None,
                overall_json=run_dir / "overall.json" if (run_dir / "overall.json").exists() else None,
            )
        )
    return runs


def load_run_zone_temperatures(run_files: RunFiles) -> pd.DataFrame:
    meta_from_path = parse_run_from_path(run_files.timeseries_csv)
    overall = load_json(run_files.overall_json)
    zone_map = parse_zone_map(run_files.zone_map_json)

    df = pd.read_csv(run_files.timeseries_csv)
    df = build_datetime_index(df, overall)
    dt_h = infer_dt_hours(df)

    zone_numbers = available_zone_numbers(df)
    if not zone_numbers:
        raise RuntimeError(f"No TAir_<zone> columns found in {run_files.timeseries_csv}")

    pieces: list[pd.DataFrame] = []
    for z in zone_numbers:
        tair = maybe_kelvin_to_celsius(safe_series(df, f"TAir_{z}"))
        if tair is None:
            continue
        out = pd.DataFrame(
            {
                "datetime": df["datetime"],
                "zone_no": z,
                "zone_name": str(zone_map.get(z, f"Zone_{z}")),
                "tair_C": tair,
            }
        )
        pieces.append(out)

    if not pieces:
        raise RuntimeError(f"Could not load any zonal temperature profiles from {run_files.timeseries_csv}")

    out = pd.concat(pieces, ignore_index=True)
    out["variant"] = meta_from_path["variant"]
    out["building_id"] = meta_from_path["building_id"]
    out["base_building_group"] = base_building_group_key(meta_from_path["building_id"])
    out["weather_key"] = meta_from_path["weather_key"]
    out["sample_id"] = meta_from_path["sample_id"]
    out["seed"] = meta_from_path["seed"]
    out["dt_hours"] = dt_h
    return out


def build_zone_temperature_table(runs: list[RunFiles]) -> pd.DataFrame:
    pieces = []
    for i, rf in enumerate(runs, start=1):
        pieces.append(load_run_zone_temperatures(rf))
        if i % 50 == 0 or i == len(runs):
            print(f"Loaded {i}/{len(runs)} runs")
    return pd.concat(pieces, ignore_index=True)


def aggregate_across_seeds_first(df_profiles: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "base_building_group",
        "weather_key",
        "sample_id",
        "datetime",
        "zone_no",
        "zone_name",
    ]
    out = (
        df_profiles
        .groupby(group_cols, dropna=False)
        .agg(tair_C=("tair_C", "mean"), dt_hours=("dt_hours", "median"))
        .reset_index()
    )
    return out


def profile_id_columns(df: pd.DataFrame, aggregated: bool) -> list[str]:
    cols = ["base_building_group", "weather_key", "sample_id"]
    if not aggregated:
        cols.append("seed")
    return cols


def select_window(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    kind = str(spec.get("kind", "all"))
    year = int(pd.Timestamp(df["datetime"].iloc[0]).year)

    if kind == "all":
        return df.copy()

    if kind == "duration_days":
        n_days = float(spec["days"])
        start = pd.Timestamp(df["datetime"].min())
        end = start + pd.Timedelta(days=n_days) - pd.Timedelta(hours=1)
        return df[(df["datetime"] >= start) & (df["datetime"] <= end)].copy()

    if kind == "date_range":
        start = pd.Timestamp(f"{year}-{spec['start']}")
        end = pd.Timestamp(f"{year}-{spec['end']}")
        return df[(df["datetime"] >= start) & (df["datetime"] <= end)].copy()

    if kind == "multi_range":
        pieces = []
        for start_s, end_s in spec.get("ranges", []):
            start = pd.Timestamp(f"{year}-{start_s}")
            end = pd.Timestamp(f"{year}-{end_s}")
            pieces.append(df[(df["datetime"] >= start) & (df["datetime"] <= end)].copy())
        if not pieces:
            return df.iloc[0:0].copy()
        return pd.concat(pieces, ignore_index=True).sort_values(["zone_no", "datetime"])

    raise ValueError(f"Unknown window kind: {kind}")


def build_cumulative_overheating_profiles(df_profiles: pd.DataFrame, aggregated: bool) -> pd.DataFrame:
    if df_profiles.empty:
        return df_profiles.copy()

    id_cols = profile_id_columns(df_profiles, aggregated)
    work = df_profiles.copy().sort_values(id_cols + ["zone_no", "datetime"])
    work["overheat_h_step"] = np.where(
        work["tair_C"] > float(OVERHEATING_THRESHOLD_C),
        work["dt_hours"],
        0.0,
    )
    work["cum_overheating_h"] = (
        work.groupby(id_cols + ["zone_no", "zone_name"], dropna=False)["overheat_h_step"]
        .cumsum()
    )
    return work


def compute_window_quantiles(df_profiles: pd.DataFrame) -> pd.DataFrame:
    aggregated = AGGREGATE_SEEDS_FIRST
    df_work = aggregate_across_seeds_first(df_profiles) if aggregated else df_profiles.copy()
    df_work = build_cumulative_overheating_profiles(df_work, aggregated=aggregated)

    q = (
        df_work
        .groupby(["datetime", "zone_no", "zone_name"], dropna=False)["cum_overheating_h"]
        .quantile([0.05, 0.5, 0.95])
        .unstack()
        .reset_index()
        .rename(columns={0.05: "q05", 0.5: "q50", 0.95: "q95"})
        .sort_values(["zone_no", "datetime"])
    )
    return q


def build_window_final_totals(df_profiles: pd.DataFrame) -> pd.DataFrame:
    aggregated = AGGREGATE_SEEDS_FIRST
    df_work = aggregate_across_seeds_first(df_profiles) if aggregated else df_profiles.copy()
    id_cols = profile_id_columns(df_work, aggregated)
    df_work = build_cumulative_overheating_profiles(df_work, aggregated=aggregated)

    totals = (
        df_work
        .groupby(id_cols + ["zone_no", "zone_name"], dropna=False)["cum_overheating_h"]
        .max()
        .reset_index(name="overheating_hours")
    )

    summary = (
        totals
        .groupby(["zone_no", "zone_name"], dropna=False)["overheating_hours"]
        .quantile([0.05, 0.5, 0.95])
        .unstack()
        .reset_index()
        .rename(columns={0.05: "p05_overheating_h", 0.5: "p50_overheating_h", 0.95: "p95_overheating_h"})
        .sort_values("zone_no")
    )
    return summary


def infer_gap_threshold_hours(x: pd.Series) -> float:
    if len(x) < 2:
        return float("inf")
    diffs_h = x.diff().dropna().dt.total_seconds() / 3600.0
    if diffs_h.empty:
        return float("inf")
    dt_h = float(diffs_h.median())
    return max(2.5 * dt_h, 3.0)


def segmented_xy(x: pd.Series, y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    if len(x) == 0:
        return np.array([]), np.array([])

    x_list: list[object] = [x.iloc[0]]
    y_list: list[float] = [float(y.iloc[0])]
    gap_h = infer_gap_threshold_hours(x)

    for i in range(1, len(x)):
        diff_h = (x.iloc[i] - x.iloc[i - 1]).total_seconds() / 3600.0
        if diff_h > gap_h:
            x_list.append(x.iloc[i - 1] + pd.Timedelta(hours=1))
            y_list.append(np.nan)
        x_list.append(x.iloc[i])
        y_list.append(float(y.iloc[i]))

    return np.asarray(x_list, dtype=object), np.asarray(y_list, dtype=float)


def add_grid(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.3)


def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def plot_window_stat_small_multiples(
    quantiles_by_variant: dict[str, pd.DataFrame],
    n_profiles_by_variant: dict[str, int],
    window_name: str,
    stat_name: str,
) -> Path:
    variants_to_plot = [v for v in VARIANT_ORDER if v in quantiles_by_variant]
    if not variants_to_plot:
        raise RuntimeError("No variant quantile profiles available for plotting.")

    n_variants = len(variants_to_plot)
    n_rows = math.ceil(n_variants / N_COLS)

    fig, axes = plt.subplots(
        n_rows,
        N_COLS,
        figsize=(FIG_WIDTH, FIG_HEIGHT_PER_ROW * n_rows),
        sharex=False,
        sharey=SHARE_Y,
        constrained_layout=True,
    )
    axes_flat = list(axes.flat) if hasattr(axes, "flat") else [axes]

    value_col = STAT_TO_COLUMN[stat_name]

    for ax_idx, variant in enumerate(variants_to_plot):
        ax = axes_flat[ax_idx]
        q_df = quantiles_by_variant[variant]

        if q_df.empty:
            ax.set_visible(False)
            continue

        zone_labels = []
        for zone_no, grp in q_df.groupby("zone_no", sort=True):
            grp = grp.sort_values("datetime")
            zone_name = str(grp["zone_name"].iloc[0])
            label = f"Z{int(zone_no)}"
            if zone_name and zone_name != f"Zone_{int(zone_no)}":
                label = f"Z{int(zone_no)}: {zone_name}"
            zone_labels.append(label)

            x_seg, y_seg = segmented_xy(grp["datetime"], grp[value_col])
            ax.plot(x_seg, y_seg, linewidth=1.3, label=label)

        ax.set_title(f"{variant} (n_pr={n_profiles_by_variant[variant]})")
        add_grid(ax)
        if ax_idx % N_COLS == 0:
            ax.set_ylabel("Cumulative overheating hours [h]")
        if ax_idx >= (n_rows - 1) * N_COLS:
            ax.set_xlabel("Time")

        legend_cols = 1 if len(zone_labels) <= 8 else (2 if len(zone_labels) <= 14 else 3)
        ax.legend(fontsize=6, ncol=legend_cols, loc="upper left", framealpha=0.9)

    for ax in axes_flat[n_variants:]:
        ax.set_visible(False)

    fig.suptitle(
        f"Zone overheating (> {OVERHEATING_THRESHOLD_C:.1f}°C) across variants – {stat_name.upper()} – {window_name.replace('_', ' ')}",
        fontsize=14,
    )

    out_path = OUTPUT_DIR / f"{stat_name}_{window_name}_all_variants_zone_overheating.png"
    savefig(fig, out_path)
    return out_path


def export_window_csvs(
    quantiles_by_variant: dict[str, pd.DataFrame],
    totals_by_variant: dict[str, pd.DataFrame],
    window_name: str,
) -> None:
    for variant, q_df in quantiles_by_variant.items():
        if not q_df.empty:
            out_csv = OUTPUT_DIR / "csv" / window_name / f"{variant}_zone_overheating_quantiles_{window_name}.csv"
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            q_df.to_csv(out_csv, index=False)

    for variant, totals_df in totals_by_variant.items():
        if not totals_df.empty:
            out_csv = OUTPUT_DIR / "csv" / window_name / f"{variant}_zone_overheating_totals_{window_name}.csv"
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            totals_df.to_csv(out_csv, index=False)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_profiles_by_variant: dict[str, pd.DataFrame] = {}
    n_profiles_by_variant: dict[str, int] = {}

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

        print(f"[INFO] Building zonal temperature profiles for {variant} ...")
        df_profiles = build_zone_temperature_table(runs)
        raw_out = OUTPUT_DIR / "raw" / f"{variant}_raw_zone_temperature_profiles.csv"
        raw_out.parent.mkdir(parents=True, exist_ok=True)
        df_profiles.to_csv(raw_out, index=False)

        if AGGREGATE_SEEDS_FIRST:
            df_sample_profiles = aggregate_across_seeds_first(df_profiles)
            n_profiles = int(
                df_sample_profiles[["base_building_group", "weather_key", "sample_id"]]
                .drop_duplicates()
                .shape[0]
            )
            sample_out = OUTPUT_DIR / "sample_mean_profiles" / f"{variant}_sample_mean_zone_temperature_profiles.csv"
            sample_out.parent.mkdir(parents=True, exist_ok=True)
            df_sample_profiles.to_csv(sample_out, index=False)
        else:
            n_profiles = int(
                df_profiles[["base_building_group", "weather_key", "sample_id", "seed"]]
                .drop_duplicates()
                .shape[0]
            )

        raw_profiles_by_variant[variant] = df_profiles
        n_profiles_by_variant[variant] = n_profiles

    if not raw_profiles_by_variant:
        raise RuntimeError("No variant profiles could be built. Check RESULTS_ROOT and filters.")

    saved_figures: list[Path] = []
    for window_name, window_spec in WINDOW_SPECS.items():
        quantiles_by_variant: dict[str, pd.DataFrame] = {}
        totals_by_variant: dict[str, pd.DataFrame] = {}

        for variant, df_profiles in raw_profiles_by_variant.items():
            df_window = select_window(df_profiles, window_spec)
            if df_window.empty:
                print(f"[INFO] Empty window for {variant}: {window_name}")
                continue

            q_df = compute_window_quantiles(df_window)
            totals_df = build_window_final_totals(df_window)
            quantiles_by_variant[variant] = q_df
            totals_by_variant[variant] = totals_df

        if not quantiles_by_variant:
            print(f"[INFO] No quantiles available for window: {window_name}")
            continue

        export_window_csvs(quantiles_by_variant, totals_by_variant, window_name)

        for stat_name in STAT_TO_COLUMN:
            out_path = plot_window_stat_small_multiples(
                quantiles_by_variant=quantiles_by_variant,
                n_profiles_by_variant=n_profiles_by_variant,
                window_name=window_name,
                stat_name=stat_name,
            )
            saved_figures.append(out_path)
            print(f"Saved: {out_path}")

    summary_lines = [
        f"RESULTS_ROOT={RESULTS_ROOT}",
        f"OUTPUT_DIR={OUTPUT_DIR}",
        f"AGGREGATE_SEEDS_FIRST={AGGREGATE_SEEDS_FIRST}",
        f"OVERHEATING_THRESHOLD_C={OVERHEATING_THRESHOLD_C}",
        f"BUILDING_ID_CONTAINS={BUILDING_ID_CONTAINS}",
        f"WEATHER_FILTER={WEATHER_FILTER}",
        f"n_saved_figures={len(saved_figures)}",
        "figures:",
    ]
    summary_lines.extend(str(p) for p in saved_figures)
    (OUTPUT_DIR / "run_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
