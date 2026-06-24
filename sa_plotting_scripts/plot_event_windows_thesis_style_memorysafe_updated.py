#!/usr/bin/env python3
from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


# ============================================================
# CONFIGURATION
# ============================================================
# Edit these paths for your local machine.
PROJECT_DIR = Path(__file__).resolve().parent

# Repository root:
# .../mza_sensitivity_analysis
REPO_DIR = PROJECT_DIR.parent

RESULTS_ROOT = PROJECT_DIR / "sa_results" / "sa_main"

BUILDING_DATA_PKL = REPO_DIR / "data" / "sa_building_data" / "building_data_merged.pkl"
BUILDING_DATA_CSV = REPO_DIR / "data" / "sa_building_data" / "building_data_merged.csv"

OUTPUT_DIR = PROJECT_DIR / "output" / "event_window_thesis_style"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Main thesis variants only.
VARIANT_ORDER = ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8"]

# Used for the peak-heating and weighted-mean overheating plots.
SELECTED_VARIANTS = ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8"]

# Used for the zone-level overheating plot.
# Keep this limited; otherwise the zone-level figure becomes unreadable.
ZONE_PLOT_VARIANTS = ["V2", "V3", "V5"]

BUILDING_ID_CONTAINS = None      # e.g. "DESHPDHK0000lce9"
WEATHER_FILTER = None            # e.g. "TRY_A" or "TRY_B"

# False = each seed/run is treated separately when computing the median profile.
# True  = average seeds first per sample, then compute the median profile.
AGGREGATE_SEEDS_FIRST = False

EVENT_WINDOW_DAYS = 3
OVERHEATING_THRESHOLD_C = 26.0
SKIP_INITIAL_DAYS_FOR_PEAK = 1

DPI = 300
SHOW_PLOTS = False
SAVE_PNG = True
SHOW_PLOT_TITLES = False

# Zone-temperature plot option.
# False = show only individual zone temperatures.
# True  = also overlay the area-weighted mean temperature as a thicker line.
INCLUDE_MEAN_IN_ZONE_TEMPERATURE_PLOT = False

# Thesis figure sizes, full text width.
FIGSIZE_SINGLE = (6.4, 2.85)
FIGSIZE_PEAK = (6.4, 3.65)
FIGSIZE_ZONE = (6.4, 3.15)
FIGSIZE_OVERHEAT = (6.4, 3.65)

# Thesis style colours.
THESIS_COLORS = {
    "grid": "#D9D9D9",
    "text": "#000000",
    "axis": "#000000",
    "threshold": "#000000",
    "peak_marker": "#666666",
}

# Muted technical line palette. Kept calm and compatible with the teal thesis style.
VARIANT_STYLES: dict[str, dict[str, object]] = {
    "V1": {"color": "#000000", "linestyle": "-", "label": "V1"},
    "V2": {"color": "#8FBFBC", "linestyle": "-", "label": "V2"},
    "V3": {"color": "#6FA9A6", "linestyle": "-", "label": "V3"},
    "V4": {"color": "#3F7D7A", "linestyle": "-", "label": "V4"},
    "V5": {"color": "#D9A36A", "linestyle": "-", "label": "V5"},
    "V6": {"color": "#B86B4B", "linestyle": "-", "label": "V6"},
    "V7": {"color": "#7A7A7A", "linestyle": "-", "label": "V7"},
    "V8": {"color": "#6F5F90", "linestyle": "-", "label": "V8"},
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "text.color": THESIS_COLORS["text"],
    "axes.labelcolor": THESIS_COLORS["text"],
    "xtick.color": THESIS_COLORS["text"],
    "ytick.color": THESIS_COLORS["text"],
    "axes.edgecolor": THESIS_COLORS["axis"],
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
# ============================================================


class RunFiles:
    def __init__(
        self,
        run_dir: Path,
        timeseries_csv: Path,
        zone_map_json: Optional[Path],
        overall_json: Optional[Path],
    ):
        self.run_dir = run_dir
        self.timeseries_csv = timeseries_csv
        self.zone_map_json = zone_map_json
        self.overall_json = overall_json


def load_json(path: Optional[Path]) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_datetime_index(df: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
    year = int(metadata.get("year", 2021))
    base = pd.Timestamp(f"{year}-01-01 00:00:00")
    if "time_s" not in df.columns:
        raise KeyError("Column 'time_s' is missing in timeseries.csv")

    out = df.copy()
    out["datetime"] = base + pd.to_timedelta(out["time_s"], unit="s")
    return out


def parse_run_from_path(ts_csv: Path) -> dict[str, Any]:
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

    return {
        "variant": parts[sample_idx - 3],
        "building_id": parts[sample_idx - 2],
        "weather_key": parts[sample_idx - 1],
        "sample_id": int(parts[sample_idx].split("_")[-1]),
        "seed": int(parts[seed_idx].split("_")[-1]),
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


def maybe_kelvin_to_celsius(series: Optional[pd.Series]) -> Optional[pd.Series]:
    if series is None:
        return None
    finite = series.dropna()
    if finite.empty:
        return series
    if finite.median() > 150:
        return series - 273.15
    return series


def available_zone_numbers(df: pd.DataFrame, base_name: str) -> list[int]:
    nums: list[int] = []
    pattern = rf"{re.escape(base_name)}_(\d+)"
    for col in df.columns:
        m = re.fullmatch(pattern, col)
        if m:
            nums.append(int(m.group(1)))
    return sorted(set(nums))


def parse_zone_map(zone_map_path: Optional[Path]) -> dict[int, str]:
    """Read zone_map.json and return {zone_number: zone_name}."""
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


def load_building_payload_by_id(
    building_id: str,
    pkl_path: Path,
    csv_path: Path,
) -> Optional[dict[str, Any]]:
    building_id = str(building_id).strip()

    if pkl_path.is_file():
        with open(pkl_path, "rb") as fh:
            data = pickle.load(fh)

        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                bid = item.get("building_id") or item.get("Building ID") or item.get("id")
                if str(bid).strip() == building_id:
                    return item

    if csv_path.is_file():
        df = pd.read_csv(csv_path)
        id_col = None
        for candidate in ["building_id", "Building ID", "id"]:
            if candidate in df.columns:
                id_col = candidate
                break

        if id_col is not None:
            subset = df[df[id_col].astype(str).str.strip() == building_id]
            if not subset.empty:
                return {"building_id": building_id, "rows": subset.to_dict(orient="records")}

    return None


def zone_floor_area_from_payload_zone(zone: dict[str, Any]) -> float:
    floors = zone.get("floors") or []
    try:
        return float(floors[0][1][0])
    except Exception:
        return 0.0


def collect_zone_areas_from_payload(building_payload: Optional[dict[str, Any]]) -> dict[str, float]:
    """Return {zone_name: floor_area_m2} from the building payload."""
    if not isinstance(building_payload, dict):
        return {}

    storeys = ((building_payload.get("polygons", {}) or {}).get("storeys", []) or [])

    zone_area_map: dict[str, float] = {}
    for storey in storeys:
        for zone in (storey.get("zones", []) or []):
            zname = str(zone.get("name", "")).strip()
            if not zname:
                continue
            area = zone_floor_area_from_payload_zone(zone)
            zone_area_map[zname] = zone_area_map.get(zname, 0.0) + area

    return zone_area_map


def _lookup_zone_area(zone_name: str, zone_area_map: dict[str, float]) -> float:
    """Try exact zone-name match first, then suffix match as a fallback."""
    zone_name = str(zone_name).strip()

    if zone_name in zone_area_map:
        return float(zone_area_map[zone_name])

    matches = [
        area for name, area in zone_area_map.items()
        if str(name).strip().endswith(zone_name) or zone_name.endswith(str(name).strip())
    ]
    if len(matches) == 1:
        return float(matches[0])

    return 0.0


def resolve_zone_weights(
    zone_numbers: list[int],
    zone_map: dict[int, str],
    building_payload: Optional[dict[str, Any]],
) -> np.ndarray:
    """Area weights for zone temperatures. Equal weights are used if areas cannot be resolved."""
    if not zone_numbers:
        return np.array([], dtype=float)

    zone_area_map = collect_zone_areas_from_payload(building_payload)

    weights = np.zeros(len(zone_numbers), dtype=float)
    for i, zone_no in enumerate(zone_numbers):
        zone_name = str(zone_map.get(zone_no, f"Zone_{zone_no}")).strip()
        weights[i] = _lookup_zone_area(zone_name, zone_area_map)

    if np.sum(weights) > 0:
        return weights / np.sum(weights)

    return np.full(len(zone_numbers), 1.0 / len(zone_numbers), dtype=float)


def weighted_mean_series(series_list: list[pd.Series], weights: np.ndarray) -> pd.Series:
    total = pd.Series(0.0, index=series_list[0].index, dtype=float)
    for s, w in zip(series_list, weights):
        total = total.add(s * float(w), fill_value=0.0)
    return total


def total_heat_kW(df: pd.DataFrame) -> pd.Series:
    zone_numbers = available_zone_numbers(df, "HeatDemand")

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

    raise RuntimeError("No heating demand column found. Expected HeatDemand_<zone>, HeatDemand_SUM, or HeatDemand.")

def total_internal_gains_kW(df: pd.DataFrame) -> pd.Series:
    """
    Total internal gains in kW.

    The simulation output stores internal gains separately as:
    GainsLights_<zone>, GainsMachines_<zone>, GainsHumans_<zone>.

    These are assumed to be in W and converted to kW.
    """

    gain_base_names = [
        "GainsLights",
        "GainsMachines",
        "GainsHumans",
    ]

    total = pd.Series(0.0, index=df.index, dtype=float)
    found = False

    for base_name in gain_base_names:
        zone_numbers = available_zone_numbers(df, base_name)

        for z in zone_numbers:
            s = safe_series(df, f"{base_name}_{z}")
            if s is not None:
                total = total.add(s, fill_value=0.0)
                found = True

    if found:
        return total / 1000.0

    candidates = [
        c for c in df.columns
        if "gain" in c.lower()
    ]

    raise RuntimeError(
        "No internal gains column found. Check the timeseries.csv column names. "
        f"Possible related columns are: {candidates[:40]}"
    )

def tair_area_weighted_mean_and_max_C(
    df: pd.DataFrame,
    zone_map: dict[int, str],
    building_payload: Optional[dict[str, Any]],
) -> tuple[pd.Series, pd.Series, list[int], list[pd.Series]]:
    """Return area-weighted mean Tair, hottest-zone Tair, valid zone numbers, and Tair series list."""
    zone_numbers = available_zone_numbers(df, "TAir")

    if not zone_numbers:
        s = maybe_kelvin_to_celsius(safe_series(df, "TAir"))
        if s is not None:
            return s, s, [1], [s]
        raise RuntimeError("No TAir_<zone> or TAir column found.")

    series_list: list[pd.Series] = []
    valid_zone_numbers: list[int] = []

    for z in zone_numbers:
        s = maybe_kelvin_to_celsius(safe_series(df, f"TAir_{z}"))
        if s is not None:
            series_list.append(s)
            valid_zone_numbers.append(z)

    if not series_list:
        raise RuntimeError("Could not read any TAir_<zone> columns.")

    tair_df = pd.concat(series_list, axis=1)
    max_tair_C = tair_df.max(axis=1)

    if len(series_list) == 1:
        return series_list[0], max_tair_C, valid_zone_numbers, series_list

    weights = resolve_zone_weights(valid_zone_numbers, zone_map, building_payload)

    if len(weights) == len(series_list) and np.sum(weights) > 0:
        area_weighted_tair_C = weighted_mean_series(series_list, weights)
    else:
        area_weighted_tair_C = tair_df.mean(axis=1)

    return area_weighted_tair_C, max_tair_C, valid_zone_numbers, series_list


def is_stairwell_like(zone_name: str) -> bool:
    s = str(zone_name).strip().lower()
    tokens = re.split(r"[^a-zA-Z0-9]+", s)
    return (
        "stair" in s
        or "stairwell" in s
        or "treppe" in s
        or "treppen" in s
        or "core" in s
        or "corridor" in s
        or "flur" in s
        or "sw" in tokens
    )


def infer_dt_hours(df: pd.DataFrame) -> float:
    if "datetime" not in df.columns or len(df) < 2:
        return 1.0

    dt = pd.to_datetime(df["datetime"]).sort_values().diff().dropna().median()
    if pd.isna(dt):
        return 1.0

    dt_h = dt.total_seconds() / 3600.0
    return float(dt_h) if dt_h > 0 else 1.0


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


def load_run_profile(
    run_files: RunFiles,
    include_building_profile: bool = True,
    include_zone_profile: bool = False,
    zone_window: Optional[tuple[pd.Timestamp, pd.Timestamp]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = parse_run_from_path(run_files.timeseries_csv)
    overall = load_json(run_files.overall_json)
    zone_map = parse_zone_map(run_files.zone_map_json)

    df = pd.read_csv(run_files.timeseries_csv)
    df = build_datetime_index(df, overall)

    building_payload = load_building_payload_by_id(
        building_id=meta["building_id"],
        pkl_path=BUILDING_DATA_PKL,
        csv_path=BUILDING_DATA_CSV,
    )

    area_weighted_tair_C, max_tair_C, valid_zone_numbers, tair_series_list = tair_area_weighted_mean_and_max_C(
        df=df,
        zone_map=zone_map,
        building_payload=building_payload,
    )

    if include_building_profile:
        profile = pd.DataFrame({
            "datetime": df["datetime"],
            "heat_kW": total_heat_kW(df),
            "internal_gains_kW": total_internal_gains_kW(df),
            "area_weighted_tair_C": area_weighted_tair_C,
            "max_tair_C": max_tair_C,
        })

        # Building-level any-zone overheating:
        profile["any_zone_overheat_flag"] = (profile["max_tair_C"] > OVERHEATING_THRESHOLD_C).astype(float)
        profile["any_zone_overheat_excess_C"] = (profile["max_tair_C"] - OVERHEATING_THRESHOLD_C).clip(lower=0.0)

        for key, value in meta.items():
            profile[key] = value
        profile["base_building_group"] = base_building_group_key(meta["building_id"])
    else:
        profile = pd.DataFrame()

    # Zone-level profiles are very large if loaded for all runs and the full year.
    # They are therefore loaded only in a second pass after the overheating
    # event window has been identified.
    if not include_zone_profile:
        return profile, pd.DataFrame()

    zone_rows: list[pd.DataFrame] = []
    for zone_no, tair_s in zip(valid_zone_numbers, tair_series_list):
        zone_name = str(zone_map.get(zone_no, f"Zone_{zone_no}")).strip()
        zone_df = pd.DataFrame({
            "datetime": df["datetime"],
            "zone_tair_C": tair_s,
            "zone_no": int(zone_no),
            "zone_name": zone_name,
            "is_stairwell": is_stairwell_like(zone_name),
        })
        if zone_window is not None:
            window_start, window_end = zone_window
            zone_df = zone_df.loc[
                (zone_df["datetime"] >= window_start)
                & (zone_df["datetime"] <= window_end)
            ].copy()
            if zone_df.empty:
                continue

        for key, value in meta.items():
            zone_df[key] = value
        zone_df["base_building_group"] = base_building_group_key(meta["building_id"])
        zone_rows.append(zone_df)

    zone_profile = pd.concat(zone_rows, ignore_index=True) if zone_rows else pd.DataFrame()
    return profile, zone_profile


def build_profiles_tables(
    runs: list[RunFiles],
    include_building_profile: bool = True,
    include_zone_profile: bool = False,
    zone_window: Optional[tuple[pd.Timestamp, pd.Timestamp]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    building_pieces: list[pd.DataFrame] = []
    zone_pieces: list[pd.DataFrame] = []

    for i, rf in enumerate(runs, start=1):
        building_profile, zone_profile = load_run_profile(
            rf,
            include_building_profile=include_building_profile,
            include_zone_profile=include_zone_profile,
            zone_window=zone_window,
        )
        if not building_profile.empty:
            building_pieces.append(building_profile)
        if not zone_profile.empty:
            zone_pieces.append(zone_profile)

        if i % 50 == 0 or i == len(runs):
            print(f"Loaded {i}/{len(runs)} runs")

    building_all = pd.concat(building_pieces, ignore_index=True) if building_pieces else pd.DataFrame()
    zone_all = pd.concat(zone_pieces, ignore_index=True) if zone_pieces else pd.DataFrame()
    return building_all, zone_all


def aggregate_across_seeds_first(df_profiles: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["variant", "base_building_group", "weather_key", "sample_id", "datetime"]
    value_cols = [
        "heat_kW",
        "internal_gains_kW",
        "area_weighted_tair_C",
        "max_tair_C",
        "any_zone_overheat_flag",
        "any_zone_overheat_excess_C",
    ]

    return (
        df_profiles
        .groupby(group_cols, dropna=False)[value_cols]
        .mean()
        .reset_index()
    )


def compute_median_profiles(df_profiles: pd.DataFrame) -> pd.DataFrame:
    """Median building-level profiles per variant and timestamp."""
    value_cols = [
        "heat_kW",
        "internal_gains_kW",
        "area_weighted_tair_C",
        "max_tair_C",
        "any_zone_overheat_flag",
        "any_zone_overheat_excess_C",
    ]

    if AGGREGATE_SEEDS_FIRST:
        df_work = aggregate_across_seeds_first(df_profiles)
    else:
        df_work = df_profiles[["variant", "datetime", *value_cols]].copy()

    return (
        df_work
        .groupby(["variant", "datetime"], dropna=False)[value_cols]
        .median()
        .reset_index()
        .sort_values(["variant", "datetime"])
    )


def compute_median_zone_profiles(df_zone_profiles: pd.DataFrame) -> pd.DataFrame:
    """Median zone-level Tair profiles per variant, zone and timestamp."""
    if df_zone_profiles.empty:
        return pd.DataFrame()

    if AGGREGATE_SEEDS_FIRST:
        group_seed = [
            "variant", "base_building_group", "weather_key", "sample_id",
            "zone_no", "zone_name", "is_stairwell", "datetime"
        ]
        df_work = (
            df_zone_profiles
            .groupby(group_seed, dropna=False)["zone_tair_C"]
            .mean()
            .reset_index()
        )
    else:
        keep_cols = ["variant", "zone_no", "zone_name", "is_stairwell", "datetime", "zone_tair_C"]
        df_work = df_zone_profiles[keep_cols].copy()

    return (
        df_work
        .groupby(["variant", "zone_no", "zone_name", "is_stairwell", "datetime"], dropna=False)["zone_tair_C"]
        .median()
        .reset_index()
        .sort_values(["variant", "zone_no", "datetime"])
    )


def find_peak_heating_window(median_profiles: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """Find the peak heating window after skipping the initial warm-up day."""

    first_time = pd.Timestamp(median_profiles["datetime"].min())
    last_time = pd.Timestamp(median_profiles["datetime"].max())

    peak_search_start = first_time + pd.Timedelta(days=SKIP_INITIAL_DAYS_FOR_PEAK)

    df_peak = median_profiles.loc[median_profiles["datetime"] >= peak_search_start].copy()

    if df_peak.empty:
        raise RuntimeError("No data left after skipping initial days for peak heating search.")

    idx = df_peak["heat_kW"].idxmax()
    peak_time = pd.Timestamp(df_peak.loc[idx, "datetime"])

    window = pd.Timedelta(days=EVENT_WINDOW_DAYS)
    half_window = pd.Timedelta(days=EVENT_WINDOW_DAYS / 2.0)

    start = peak_time - half_window
    end = peak_time + half_window

    if start < peak_search_start:
        start = peak_search_start
        end = start + window

    if end > last_time:
        end = last_time
        start = end - window

    return start, end, peak_time


def find_worst_any_zone_overheating_window(median_profiles: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Find the 3-day window with the highest any-zone overheating hours.

    The event signal is the sum of median any-zone overheating indicators across variants.
    """
    event_signal = (
        median_profiles
        .groupby("datetime", dropna=False)["any_zone_overheat_flag"]
        .sum()
        .sort_index()
    )

    dt = event_signal.index.to_series().diff().dropna().median()
    dt_h = dt.total_seconds() / 3600.0 if not pd.isna(dt) else 1.0
    if dt_h <= 0:
        dt_h = 1.0

    window_points = max(1, int(round(EVENT_WINDOW_DAYS * 24.0 / dt_h)))
    rolling_sum = event_signal.rolling(window=window_points, min_periods=window_points).sum()

    if rolling_sum.dropna().empty:
        rolling_sum = event_signal.rolling(window=window_points, min_periods=1).sum()

    end = pd.Timestamp(rolling_sum.idxmax())
    start = end - pd.Timedelta(days=EVENT_WINDOW_DAYS)

    first = pd.Timestamp(event_signal.index.min())
    last = pd.Timestamp(event_signal.index.max())
    if start < first:
        start = first
        end = start + pd.Timedelta(days=EVENT_WINDOW_DAYS)
    if end > last:
        end = last
        start = end - pd.Timedelta(days=EVENT_WINDOW_DAYS)

    return start, end


def summarize_overheating_hours_in_window(
    median_profiles: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Summarize median any-zone overheating hours per variant inside the selected window."""
    rows = []

    for variant in SELECTED_VARIANTS:
        df_v = median_profiles.loc[
            (median_profiles["variant"] == variant)
            & (median_profiles["datetime"] >= start)
            & (median_profiles["datetime"] <= end)
        ].copy()

        if df_v.empty:
            continue

        dt_h = infer_dt_hours(df_v)
        overheat_hours = float(df_v["any_zone_overheat_flag"].sum() * dt_h)
        degree_hours = float(df_v["any_zone_overheat_excess_C"].sum() * dt_h)

        rows.append({
            "variant": variant,
            "window_start": start,
            "window_end": end,
            "dt_h": dt_h,
            "median_any_zone_overheating_hours": overheat_hours,
            "median_any_zone_degree_hours_C_h": degree_hours,
            "max_hottest_zone_temperature_C": float(df_v["max_tair_C"].max()),
            "max_area_weighted_temperature_C": float(df_v["area_weighted_tair_C"].max()),
        })

    return pd.DataFrame(rows)


def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    if SAVE_PNG and path.suffix.lower() == ".pdf":
        fig.savefig(path.with_suffix(".png"), dpi=DPI, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, axis="both", linestyle="-", linewidth=0.5, color=THESIS_COLORS["grid"])
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(THESIS_COLORS["axis"])
        spine.set_linewidth(0.8)
    ax.tick_params(axis="both", width=0.8, length=3, colors=THESIS_COLORS["text"])


def format_event_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m\n%H:%M"))


def subset_window(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, variant: str) -> pd.DataFrame:
    return df.loc[
        (df["variant"] == variant)
        & (df["datetime"] >= start)
        & (df["datetime"] <= end)
    ].copy()


def plot_peak_heating_with_temperature(
    median_profiles: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    peak_time: pd.Timestamp,
    output_name: str,
) -> Path:
    """Two-panel peak-heating event plot.

    Top panel: total heating load [kW].
    Bottom panel: area-weighted mean indoor air temperature [°C].

    The event window is selected automatically by searching for the maximum
    heating load after skipping the initial warm-up day.
    """
    fig, (ax_heat, ax_temp) = plt.subplots(
        2,
        1,
        figsize=FIGSIZE_PEAK,
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.12},
    )

    for variant in SELECTED_VARIANTS:
        df_v = subset_window(median_profiles, start, end, variant)
        if df_v.empty:
            continue

        style = VARIANT_STYLES.get(variant, {})
        color = style.get("color", None)
        linestyle = style.get("linestyle", "-")
        label = str(style.get("label", variant))

        # Heating load is plotted first because this figure is selected from
        # the peak-heating demand event.
        ax_heat.plot(
            df_v["datetime"],
            df_v["heat_kW"],
            color=color,
            linestyle=linestyle,
            linewidth=1.1,
            label=label,
        )

        ax_temp.plot(
            df_v["datetime"],
            df_v["area_weighted_tair_C"],
            color=color,
            linestyle=linestyle,
            linewidth=1.1,
            label=label,
        )

    for ax in (ax_heat, ax_temp):
        ax.axvline(peak_time, color=THESIS_COLORS["peak_marker"], linestyle=":", linewidth=0.9)
        ax.set_xlim(start, end)
        style_axis(ax)
        format_event_axis(ax)

    if SHOW_PLOT_TITLES:
        ax_heat.set_title(f"{EVENT_WINDOW_DAYS}-day peak heating event window", pad=4)

    ax_heat.set_ylabel("Heating load [kW]")
    ax_temp.set_ylabel("Weighted mean\nair temperature [°C]")
    ax_temp.set_xlabel("Time")

    # One shared legend above the figure.
    handles, labels = ax_heat.get_legend_handles_labels()
    peak_handle = Line2D([0], [0], color=THESIS_COLORS["peak_marker"], linestyle=":", linewidth=0.9, label="Peak time")
    fig.legend(
        handles + [peak_handle],
        labels + ["Peak time"],
        loc="upper center",
        bbox_to_anchor=(0.71, 1.02),
        ncol=5,
        frameon=False,
        handlelength=1.8,
        columnspacing=0.9,
    )

    fig.subplots_adjust(left=0.105, right=0.995, bottom=0.15, top=0.88)

    out_path = OUTPUT_DIR / output_name
    savefig(fig, out_path)
    return out_path

def plot_overheating_weighted_mean_temperature(
    median_profiles: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    output_name: str,
) -> Path:
    """Two-panel overheating event plot.

    Top panel: total internal gains [kW].
    Bottom panel: area-weighted mean indoor air temperature [°C].
    """

    fig, (ax_temp, ax_gains) = plt.subplots(
        2,
        1,
        figsize=FIGSIZE_OVERHEAT,
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.12},
    )

    for variant in SELECTED_VARIANTS:
        df_v = subset_window(median_profiles, start, end, variant)
        if df_v.empty:
            continue

        style = VARIANT_STYLES.get(variant, {})
        color = style.get("color", None)
        linestyle = style.get("linestyle", "-")
        label = str(style.get("label", variant))

        ax_gains.plot(
            df_v["datetime"],
            df_v["internal_gains_kW"],
            color=color,
            linestyle=linestyle,
            linewidth=1.1,
            label=label,
        )

        ax_temp.plot(
            df_v["datetime"],
            df_v["area_weighted_tair_C"],
            color=color,
            linestyle=linestyle,
            linewidth=1.1,
            label=label,
        )

    ax_temp.axhline(
        OVERHEATING_THRESHOLD_C,
        color=THESIS_COLORS["threshold"],
        linestyle="--",
        linewidth=0.9,
        label=f"{OVERHEATING_THRESHOLD_C:g} °C threshold",
    )

    for ax in (ax_gains, ax_temp):
        ax.set_xlim(start, end)
        style_axis(ax)
        format_event_axis(ax)

    if SHOW_PLOT_TITLES:
        ax_gains.set_title(
            f"{EVENT_WINDOW_DAYS}-day overheating event window",
            pad=4,
        )

    ax_gains.set_ylabel("Internal gains [kW]")
    ax_temp.set_ylabel("Weighted mean\nair temperature [°C]")
    ax_temp.set_xlabel("Time")

    handles, labels = ax_gains.get_legend_handles_labels()
    threshold_handle = Line2D(
        [0],
        [0],
        color=THESIS_COLORS["threshold"],
        linestyle="--",
        linewidth=0.9,
        label=f"{OVERHEATING_THRESHOLD_C:g} °C threshold",
    )

    fig.legend(
        handles + [threshold_handle],
        labels + [f"{OVERHEATING_THRESHOLD_C:g} °C threshold"],
        loc="upper center",
        bbox_to_anchor=(0.70, 1.02),
        ncol=5,
        frameon=False,
        handlelength=1.8,
        columnspacing=0.9,
    )

    fig.subplots_adjust(left=0.105, right=0.995, bottom=0.15, top=0.88)

    out_path = OUTPUT_DIR / output_name
    savefig(fig, out_path)
    return out_path

def plot_overheating_zone_temperatures(
    median_profiles: pd.DataFrame,
    median_zone_profiles: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    output_name: str,
) -> Path:
    """Single-panel zone-level overheating event plot for selected variants.

    Thin lines show individual zone temperatures. Stairwell/core-like zones are
    shown dashed when they can be identified from the zone names.

    If INCLUDE_MEAN_IN_ZONE_TEMPERATURE_PLOT is True, the area-weighted mean
    temperature of the same variant is overlaid as a thicker line. The default
    is False because the purpose of this figure is to show zone-level variation.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_ZONE)

    legend_handles: list[Line2D] = []
    legend_labels: list[str] = []

    for variant in ZONE_PLOT_VARIANTS:
        style = VARIANT_STYLES.get(variant, {})
        color = str(style.get("color", "#333333"))
        label = str(style.get("label", variant))

        df_zone = median_zone_profiles.loc[
            (median_zone_profiles["variant"] == variant)
            & (median_zone_profiles["datetime"] >= start)
            & (median_zone_profiles["datetime"] <= end)
        ].copy()

        if df_zone.empty:
            continue

        for (_, zone_name, is_stairwell), df_z in df_zone.groupby(["zone_no", "zone_name", "is_stairwell"], dropna=False):
            ax.plot(
                df_z["datetime"],
                df_z["zone_tair_C"],
                color=color,
                linestyle="--" if bool(is_stairwell) else "-",
                linewidth=0.65,
                alpha=0.55,
                zorder=1,
            )

        if INCLUDE_MEAN_IN_ZONE_TEMPERATURE_PLOT:
            # Optional overlay of the weighted mean of the same variant.
            df_mean = subset_window(median_profiles, start, end, variant)
            if not df_mean.empty:
                ax.plot(
                    df_mean["datetime"],
                    df_mean["area_weighted_tair_C"],
                    color=color,
                    linestyle="-",
                    linewidth=1.5,
                    alpha=1.0,
                    zorder=3,
                )
            legend_handles.append(Line2D([0], [0], color=color, linewidth=1.5))
            legend_labels.append(f"{label} mean")
        else:
            # Legend entry represents the individual zone-temperature lines.
            legend_handles.append(Line2D([0], [0], color=color, linewidth=0.9, alpha=0.75))
            legend_labels.append(f"{label} zones")

    threshold_handle = Line2D([0], [0], color=THESIS_COLORS["threshold"], linestyle="--", linewidth=0.9)
    legend_handles.append(threshold_handle)
    legend_labels.append(f"{OVERHEATING_THRESHOLD_C:g} °C threshold")

    ax.axhline(
        OVERHEATING_THRESHOLD_C,
        color=THESIS_COLORS["threshold"],
        linestyle="--",
        linewidth=0.9,
        zorder=2,
    )

    if SHOW_PLOT_TITLES:
        ax.set_title(f"{EVENT_WINDOW_DAYS}-day overheating event: zone temperatures", pad=4)

    ax.set_ylabel("Zone air\ntemperature [°C]")
    ax.set_xlabel("Time")
    ax.set_xlim(start, end)
    style_axis(ax)
    format_event_axis(ax)
    ax.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.61, 1.24),
        ncol=4,
        frameon=False,
        handlelength=1.8,
        columnspacing=0.9,
    )
    fig.subplots_adjust(left=0.105, right=0.995, bottom=0.20, top=0.80)

    out_path = OUTPUT_DIR / output_name
    savefig(fig, out_path)
    return out_path

def main() -> None:
    if not RESULTS_ROOT.exists():
        raise FileNotFoundError(f"RESULTS_ROOT does not exist: {RESULTS_ROOT}")

    all_building_profiles: list[pd.DataFrame] = []
    run_summary_rows: list[dict[str, Any]] = []

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
        # First pass: load only building-level profiles.
        # Zone-level profiles are loaded later only for the selected overheating window.
        df_profiles, _ = build_profiles_tables(
            runs,
            include_building_profile=True,
            include_zone_profile=False,
        )

        if df_profiles.empty:
            continue

        all_building_profiles.append(df_profiles)

        n_runs = int(
            df_profiles[["base_building_group", "weather_key", "sample_id", "seed"]]
            .drop_duplicates()
            .shape[0]
        )
        run_summary_rows.append({"variant": variant, "n_runs": n_runs})

    if not all_building_profiles:
        raise RuntimeError("No profiles could be built.")

    df_all = pd.concat(all_building_profiles, ignore_index=True)

    median_profiles = compute_median_profiles(df_all)

    median_profiles.to_csv(OUTPUT_DIR / "event_window_median_profiles.csv", index=False)
    pd.DataFrame(run_summary_rows).to_csv(OUTPUT_DIR / "event_window_run_summary.csv", index=False)

    peak_start, peak_end, peak_time = find_peak_heating_window(median_profiles)
    overheat_start, overheat_end = find_worst_any_zone_overheating_window(median_profiles)

    selected_windows = pd.DataFrame([
        {
            "event": "peak_heating",
            "window_start": peak_start,
            "window_end": peak_end,
            "event_time": peak_time,
            "selection_rule": f"maximum median heating load after skipping the first {SKIP_INITIAL_DAYS_FOR_PEAK} day(s)",
        },
        {
            "event": "overheating",
            "window_start": overheat_start,
            "window_end": overheat_end,
            "event_time": pd.NaT,
            "selection_rule": f"maximum rolling {EVENT_WINDOW_DAYS}-day sum of median any-zone overheating indicators above {OVERHEATING_THRESHOLD_C:g} °C",
        },
    ])
    selected_windows.to_csv(OUTPUT_DIR / "selected_event_windows.csv", index=False)

    overheat_summary = summarize_overheating_hours_in_window(
        median_profiles=median_profiles,
        start=overheat_start,
        end=overheat_end,
    )
    overheat_summary.to_csv(OUTPUT_DIR / "event_window_any_zone_overheating_hours_summary.csv", index=False)

    # Second pass: now that the overheating event window is known, load only
    # zone-level data for the selected variants and only for this 3-day window.
    # This avoids holding full-year zone-level profiles for all runs in memory.
    all_zone_profiles: list[pd.DataFrame] = []
    zone_window = (overheat_start, overheat_end)

    for variant in ZONE_PLOT_VARIANTS:
        runs = discover_variant_runs(
            results_root=RESULTS_ROOT,
            variant=variant,
            building_id_contains=BUILDING_ID_CONTAINS,
            weather_filter=WEATHER_FILTER,
        )

        if not runs:
            print(f"[INFO] No zone-level runs found for {variant}")
            continue

        print(f"[INFO] Loading zone-level event-window profiles for {variant}: {len(runs)} runs found")
        _, df_zone_profiles = build_profiles_tables(
            runs,
            include_building_profile=False,
            include_zone_profile=True,
            zone_window=zone_window,
        )

        if not df_zone_profiles.empty:
            all_zone_profiles.append(df_zone_profiles)

    df_zone_all = pd.concat(all_zone_profiles, ignore_index=True) if all_zone_profiles else pd.DataFrame()
    median_zone_profiles = compute_median_zone_profiles(df_zone_all)

    if not median_zone_profiles.empty:
        median_zone_profiles.to_csv(OUTPUT_DIR / "event_window_median_zone_profiles.csv", index=False)

    peak_fig = plot_peak_heating_with_temperature(
        median_profiles=median_profiles,
        start=peak_start,
        end=peak_end,
        peak_time=peak_time,
        output_name="event_window_peak_heating_with_weighted_mean_temperature_3days.pdf",
    )

    weighted_mean_fig = plot_overheating_weighted_mean_temperature(
        median_profiles=median_profiles,
        start=overheat_start,
        end=overheat_end,
        output_name="event_window_overheating_weighted_mean_temperature_3days_1.pdf",
    )

    if median_zone_profiles.empty:
        zone_fig = None
        print("[WARNING] No zone-level temperature profiles could be built. Zone-temperature plot was skipped.")
    else:
        zone_fig = plot_overheating_zone_temperatures(
            median_profiles=median_profiles,
            median_zone_profiles=median_zone_profiles,
            start=overheat_start,
            end=overheat_end,
            output_name="event_window_overheating_zone_temperatures_selected_variants_3days.pdf",
        )

    print("Saved:")
    print(OUTPUT_DIR / "event_window_median_profiles.csv")
    print(OUTPUT_DIR / "event_window_run_summary.csv")
    print(OUTPUT_DIR / "selected_event_windows.csv")
    print(OUTPUT_DIR / "event_window_any_zone_overheating_hours_summary.csv")
    print(peak_fig)
    print(weighted_mean_fig)
    if zone_fig is not None:
        print(zone_fig)


if __name__ == "__main__":
    main()
