#!/usr/bin/env python3
from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent

# Repository root:
# .../mza_sensitivity_analysis
REPO_DIR = PROJECT_DIR.parent

RESULTS_ROOT = PROJECT_DIR / "sa_results"

BUILDING_DATA_PKL = REPO_DIR / "data" / "sa_building_data" / "building_data_merged.pkl"
BUILDING_DATA_CSV = REPO_DIR / "data" / "sa_building_data" / "building_data_merged.csv"

USE_AREA_WEIGHTED_TAIR = True
USE_AREA_WEIGHTED_TSET = True
SHOW_PLOTS = False
DPI = 150

KPIS_TO_PLOT = [
    "annual_heating_kWh",
    "peak_heating_kW",
    "overheating_hours_meanTair_gt_26C",
    "mean_interzone_spread_C",
]

REFERENCE_VARIANT = "V1"
VARIANT_ORDER = ["V1", "V1_20HH", "V2", "V2_20HH", "V3", "V4", "V5", "V6", "V7", "V8"]

OUTPUT_DIRNAME = "_analysis_plots_fixed"


class RunFiles:
    def __init__(self, run_dir: Path, timeseries_csv: Path, zone_map_json: Optional[Path], overall_json: Optional[Path]):
        self.run_dir = run_dir
        self.timeseries_csv = timeseries_csv
        self.zone_map_json = zone_map_json
        self.overall_json = overall_json


def add_grid(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.3)


def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def sort_variants(df: pd.DataFrame, col: str = "variant") -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.Categorical(out[col], categories=VARIANT_ORDER, ordered=True)
    return out.sort_values(col)


def p5(x: pd.Series) -> float:
    return float(np.nanpercentile(x, 5))


def p50(x: pd.Series) -> float:
    return float(np.nanpercentile(x, 50))


def p95(x: pd.Series) -> float:
    return float(np.nanpercentile(x, 95))


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


def base_building_group_key(building_id: str) -> str:
    s = str(building_id).strip()
    m = re.match(r"^(.*?)(?:[_-]?)(\d+)$", s)
    if not m:
        return s
    prefix = str(m.group(1)).strip()
    return prefix or s


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

    variant = parts[sample_idx - 3]
    building_id = parts[sample_idx - 2]
    weather_key = parts[sample_idx - 1]
    sample_id = int(parts[sample_idx].split("_")[-1])
    seed = int(parts[seed_idx].split("_")[-1])

    return {
        "variant": variant,
        "building_id": building_id,
        "base_building_group": base_building_group_key(building_id),
        "weather_key": weather_key,
        "sample_id": sample_id,
        "seed": seed,
        "run_dir": str(ts_csv.parent),
    }


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


def available_zone_numbers(df: pd.DataFrame) -> list[int]:
    nums: list[int] = []
    for col in df.columns:
        m = re.fullmatch(r"TAir_(\d+)", col)
        if m:
            nums.append(int(m.group(1)))
    return sorted(set(nums))


def load_building_payload_by_id(building_id: str, pkl_path: Path, csv_path: Path) -> Optional[dict[str, Any]]:
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


def resolve_zone_weights(zone_numbers: list[int], zone_map: dict[int, str], building_payload: Optional[dict[str, Any]]) -> np.ndarray:
    if not zone_numbers:
        return np.array([], dtype=float)
    zone_area_map = collect_zone_areas_from_payload(building_payload)
    weights = np.zeros(len(zone_numbers), dtype=float)
    for i, zone_no in enumerate(zone_numbers):
        zone_name = str(zone_map.get(zone_no, f"Zone_{zone_no}")).strip()
        weights[i] = float(zone_area_map.get(zone_name, 0.0))
    if np.sum(weights) > 0:
        return weights / np.sum(weights)
    return np.full(len(zone_numbers), 1.0 / len(zone_numbers), dtype=float)


def weighted_mean_series(series_list: list[pd.Series], weights: np.ndarray) -> pd.Series:
    total = pd.Series(0.0, index=series_list[0].index, dtype=float)
    for s, w in zip(series_list, weights):
        total = total.add(s * float(w), fill_value=0.0)
    return total


def sum_zone_series(df: pd.DataFrame, base_name: str, zone_numbers: list[int]) -> pd.Series:
    total = pd.Series(0.0, index=df.index, dtype=float)
    found = False
    for zone_no in zone_numbers:
        s = safe_series(df, f"{base_name}_{zone_no}")
        if s is not None:
            total = total.add(s, fill_value=0.0)
            found = True
    if not found:
        raise KeyError(f"No columns found for '{base_name}_<zone>'")
    return total


def mean_zone_series(df: pd.DataFrame, base_name: str, zone_numbers: list[int], zone_map: dict[int, str], building_payload: Optional[dict[str, Any]], use_area_weighted: bool) -> pd.Series:
    series_list: list[pd.Series] = []
    valid_zone_numbers: list[int] = []
    for zone_no in zone_numbers:
        s = safe_series(df, f"{base_name}_{zone_no}")
        if s is not None:
            series_list.append(s)
            valid_zone_numbers.append(zone_no)
    if not series_list:
        raise KeyError(f"No columns found for '{base_name}_<zone>'")
    if len(series_list) == 1:
        return series_list[0]
    if use_area_weighted:
        weights = resolve_zone_weights(valid_zone_numbers, zone_map, building_payload)
        if len(weights) == len(series_list) and np.sum(weights) > 0:
            return weighted_mean_series(series_list, weights)
    total = pd.Series(0.0, index=series_list[0].index, dtype=float)
    for s in series_list:
        total = total.add(s, fill_value=0.0)
    return total / float(len(series_list))


def infer_dt_hours(df: pd.DataFrame) -> float:
    if "time_s" not in df.columns or len(df) < 2:
        return 0.0
    dt_s = pd.to_numeric(df["time_s"], errors="coerce").diff().dropna().median()
    if pd.isna(dt_s):
        return 0.0
    return float(dt_s) / 3600.0


def compute_run_kpis(run_files: RunFiles) -> dict[str, Any]:
    path_meta = parse_run_from_path(run_files.timeseries_csv)
    overall = load_json(run_files.overall_json)
    zone_map = parse_zone_map(run_files.zone_map_json)

    df = pd.read_csv(run_files.timeseries_csv)
    df = build_datetime_index(df, overall)
    zone_numbers = available_zone_numbers(df)
    if not zone_numbers:
        raise RuntimeError(f"No TAir_<zone> columns found in {run_files.timeseries_csv}")

    building_payload = load_building_payload_by_id(path_meta["building_id"], BUILDING_DATA_PKL, BUILDING_DATA_CSV)
    dt_h = infer_dt_hours(df)

    heat_total_W = sum_zone_series(df, "HeatDemand", zone_numbers)
    try:
        cool_total_W = sum_zone_series(df, "CoolDemand", zone_numbers)
    except Exception:
        cool_total_W = pd.Series(0.0, index=df.index, dtype=float)

    tair_mean_C = maybe_kelvin_to_celsius(mean_zone_series(df, "TAir", zone_numbers, zone_map, building_payload, USE_AREA_WEIGHTED_TAIR))
    tset_mean_C = maybe_kelvin_to_celsius(mean_zone_series(df, "TSetHeat", zone_numbers, zone_map, building_payload, USE_AREA_WEIGHTED_TSET))
    assert tair_mean_C is not None and tset_mean_C is not None

    tair_zone_df = pd.DataFrame({f"TAir_{z}": maybe_kelvin_to_celsius(safe_series(df, f"TAir_{z}")) for z in zone_numbers if safe_series(df, f"TAir_{z}") is not None})

    out: dict[str, Any] = dict(path_meta)
    out["year"] = int(overall.get("year", 2021))
    out["n_timesteps"] = int(len(df))
    out["dt_hours"] = float(dt_h)
    out["n_zones"] = int(len(zone_numbers))

    task_meta = overall.get("task_meta", {}) or {}
    sample_meta = task_meta.get("sample", {}) or {}
    out["wwr_factor"] = sample_meta.get("wwr_factor")
    out["gains_scale"] = sample_meta.get("gains_scale")
    out["weather_key_meta"] = task_meta.get("weather_key")
    out["sa_tabula_year_class"] = task_meta.get("sa_tabula_year_class")
    out["sa_tset_mean_K"] = task_meta.get("sa_tset_mean_K")
    out["sa_tset_spread_K"] = task_meta.get("sa_tset_spread_K")
    out["sa_retrofit_state"] = task_meta.get("sa_retrofit_state")
    out["sa_retrofit_is_standard"] = task_meta.get("sa_retrofit_is_standard")

    out["annual_heating_kWh"] = float((heat_total_W * dt_h).sum() / 1000.0)
    out["peak_heating_kW"] = float(heat_total_W.max() / 1000.0)
    out["annual_cooling_kWh"] = float((cool_total_W * dt_h).sum() / 1000.0)
    out["peak_cooling_kW"] = float(cool_total_W.max() / 1000.0)

    out["mean_tair_C"] = float(tair_mean_C.mean())
    out["max_tair_C"] = float(tair_mean_C.max())
    out["mean_tsetheat_C"] = float(tset_mean_C.mean())
    out["overheating_hours_meanTair_gt_26C"] = float((tair_mean_C > 26.0).sum() * dt_h)
    out["underheating_hours_meanTair_below_meanTset"] = float((tair_mean_C < tset_mean_C).sum() * dt_h)

    if not tair_zone_df.empty:
        interzone_spread = tair_zone_df.max(axis=1) - tair_zone_df.min(axis=1)
        out["mean_interzone_spread_C"] = float(interzone_spread.mean())
        out["max_interzone_spread_C"] = float(interzone_spread.max())
        out["overheating_hours_any_zone_gt_26C"] = float((tair_zone_df.gt(26.0).any(axis=1)).sum() * dt_h)
    else:
        out["mean_interzone_spread_C"] = np.nan
        out["max_interzone_spread_C"] = np.nan
        out["overheating_hours_any_zone_gt_26C"] = np.nan
    return out


def discover_all_runs(results_root: Path) -> list[RunFiles]:
    runs: list[RunFiles] = []
    for ts_csv in sorted(results_root.rglob("timeseries.csv")):
        if OUTPUT_DIRNAME in ts_csv.parts:
            continue
        run_dir = ts_csv.parent
        runs.append(RunFiles(run_dir, ts_csv, run_dir / "zone_map.json" if (run_dir / "zone_map.json").exists() else None, run_dir / "overall.json" if (run_dir / "overall.json").exists() else None))
    return runs


def build_sample_summary(df_runs: pd.DataFrame, kpis: list[str]) -> pd.DataFrame:
    group_cols = ["base_building_group", "variant", "sample_id"]
    agg: dict[str, list[str]] = {k: ["mean", "std", "min", "max"] for k in kpis}
    keep_one = ["building_id", "weather_key", "wwr_factor", "gains_scale", "sa_tabula_year_class", "sa_tset_mean_K", "sa_tset_spread_K", "sa_retrofit_state", "sa_retrofit_is_standard", "n_zones"]
    for col in keep_one:
        if col in df_runs.columns:
            agg[col] = ["first"]
    df = df_runs.groupby(group_cols, dropna=False).agg(agg)
    df.columns = [f"{c}_{s}" if s != "first" else c for c, s in df.columns.to_flat_index()]
    df = df.reset_index()
    return sort_variants(df)


def build_variant_summary(df_sample: pd.DataFrame, kpis: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant, grp in df_sample.groupby("variant", observed=False):
        if grp.empty:
            continue
        row: dict[str, Any] = {"variant": variant}
        for k in kpis:
            row[f"{k}_p05"] = p5(grp[f"{k}_mean"])
            row[f"{k}_p50"] = p50(grp[f"{k}_mean"])
            row[f"{k}_p95"] = p95(grp[f"{k}_mean"])
            row[f"{k}_std_between_samples"] = float(np.nanstd(grp[f"{k}_mean"], ddof=1))
            row[f"{k}_mean_seed_std"] = float(np.nanmean(grp[f"{k}_std"]))
            row[f"{k}_n_samples"] = int(grp[f"{k}_mean"].notna().sum())
        rows.append(row)
    return sort_variants(pd.DataFrame(rows))


def plot_boxplot_sample_means(df_sample: pd.DataFrame, kpi: str, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    data = [df_sample.loc[df_sample["variant"] == v, f"{kpi}_mean"].dropna().values for v in VARIANT_ORDER]
    labels = [v for v, d in zip(VARIANT_ORDER, data) if len(d) > 0]
    data = [d for d in data if len(d) > 0]
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_title(f"{kpi}: sample means across seeds")
    ax.set_xlabel("Variant")
    ax.set_ylabel(kpi)
    add_grid(ax)
    savefig(fig, out_dir / f"01_boxplot_sample_means_{kpi}.png")


def plot_quantile_interval(df_variant: pd.DataFrame, kpi: str, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    labels, x, y, yerr_low, yerr_high = [], [], [], [], []
    for v in VARIANT_ORDER:
        rows = df_variant.loc[df_variant["variant"] == v]
        if rows.empty:
            continue
        row = rows.iloc[0]
        labels.append(v)
        x.append(len(labels)-1)
        y.append(row[f"{kpi}_p50"])
        yerr_low.append(row[f"{kpi}_p50"] - row[f"{kpi}_p05"])
        yerr_high.append(row[f"{kpi}_p95"] - row[f"{kpi}_p50"])
    ax.errorbar(x, y, yerr=[yerr_low, yerr_high], fmt="o", capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(f"{kpi}: P5 / P50 / P95 across samples")
    ax.set_xlabel("Variant")
    ax.set_ylabel(kpi)
    add_grid(ax)
    savefig(fig, out_dir / f"02_quantile_interval_{kpi}.png")


def plot_seed_std_boxplot(df_sample: pd.DataFrame, kpi: str, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    data = [df_sample.loc[df_sample["variant"] == v, f"{kpi}_std"].dropna().values for v in VARIANT_ORDER]
    labels = [v for v, d in zip(VARIANT_ORDER, data) if len(d) > 0]
    data = [d for d in data if len(d) > 0]
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_title(f"{kpi}: variability across seeds")
    ax.set_xlabel("Variant")
    ax.set_ylabel(f"seed std of {kpi}")
    add_grid(ax)
    savefig(fig, out_dir / f"03_seed_std_boxplot_{kpi}.png")


def plot_between_vs_within(df_variant: pd.DataFrame, kpi: str, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    labels, x, between, within = [], [], [], []
    for v in VARIANT_ORDER:
        rows = df_variant.loc[df_variant["variant"] == v]
        if rows.empty:
            continue
        row = rows.iloc[0]
        labels.append(v)
        x.append(len(labels)-1)
        between.append(row[f"{kpi}_std_between_samples"])
        within.append(row[f"{kpi}_mean_seed_std"])
    x = np.array(x)
    width = 0.38
    ax.bar(x - width / 2, between, width=width, label="Between-sample SD")
    ax.bar(x + width / 2, within, width=width, label="Mean within-sample seed SD")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(f"{kpi}: input uncertainty vs usage stochasticity")
    ax.set_xlabel("Variant")
    ax.set_ylabel("Standard deviation")
    ax.legend()
    add_grid(ax)
    savefig(fig, out_dir / f"04_between_vs_within_{kpi}.png")


def plot_delta_vs_reference(df_sample: pd.DataFrame, kpi: str, out_dir: Path) -> None:
    ref = df_sample.loc[df_sample["variant"] == REFERENCE_VARIANT, ["base_building_group", "sample_id", f"{kpi}_mean"]].rename(columns={f"{kpi}_mean": f"{kpi}_ref"})
    if ref.empty:
        print(f"[WARN] No reference rows found for {REFERENCE_VARIANT}; skipping delta plot for {kpi}.")
        return
    df = df_sample.merge(ref, on=["base_building_group", "sample_id"], how="left", validate="many_to_one")
    df[f"{kpi}_delta_vs_{REFERENCE_VARIANT}"] = df[f"{kpi}_mean"] - df[f"{kpi}_ref"]
    data, labels = [], []
    for v in [v for v in VARIANT_ORDER if v != REFERENCE_VARIANT]:
        arr = df.loc[df["variant"] == v, f"{kpi}_delta_vs_{REFERENCE_VARIANT}"].dropna().values
        if len(arr) > 0:
            data.append(arr)
            labels.append(v)
    if not data:
        print(f"[WARN] Delta plot for {kpi} is empty after merge. Check base_building_group and sample_id alignment.")
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.axhline(0.0, linestyle="--", color="black", linewidth=1)
    ax.set_title(f"{kpi}: delta vs {REFERENCE_VARIANT}")
    ax.set_xlabel("Variant")
    ax.set_ylabel(f"Δ {kpi}")
    add_grid(ax)
    savefig(fig, out_dir / f"05_delta_vs_{REFERENCE_VARIANT}_{kpi}.png")


def plot_heatmap_variant_medians(df_variant: pd.DataFrame, kpis: list[str], out_dir: Path) -> None:
    labels = [v for v in VARIANT_ORDER if not df_variant.loc[df_variant["variant"] == v].empty]
    arr = []
    for v in labels:
        row = df_variant.loc[df_variant["variant"] == v].iloc[0]
        arr.append([row[f"{k}_p50"] for k in kpis])
    arr_np = np.array(arr)
    fig, ax = plt.subplots(figsize=(1.7 * len(kpis), 5))
    im = ax.imshow(arr_np, aspect="auto")
    ax.set_xticks(np.arange(len(kpis)))
    ax.set_xticklabels(kpis, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Median KPI values by variant")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Median value")
    savefig(fig, out_dir / "00_heatmap_variant_medians.png")


def main() -> None:
    results_root = RESULTS_ROOT.resolve()
    out_dir = Path(r"C:\Sharon\mza_sensitivity_analysis\plotting_scripts\new_plots\sa_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    run_files = discover_all_runs(results_root)
    if not run_files:
        raise FileNotFoundError(f"No timeseries.csv files found under {results_root}")

    rows, failed = [], []
    for i, rf in enumerate(run_files, start=1):
        try:
            rows.append(compute_run_kpis(rf))
        except Exception as e:
            failed.append({"timeseries_csv": str(rf.timeseries_csv), "error": str(e)})
        if i % 100 == 0 or i == len(run_files):
            print(f"Processed {i}/{len(run_files)} runs")

    if not rows:
        raise RuntimeError("No KPIs could be extracted from the discovered runs.")

    df_runs = sort_variants(pd.DataFrame(rows))
    df_runs.to_csv(out_dir / "run_level_kpis.csv", index=False)
    if failed:
        pd.DataFrame(failed).to_csv(out_dir / "failed_runs.csv", index=False)
        print(f"Warning: {len(failed)} runs failed. See failed_runs.csv")

    sample_summary = build_sample_summary(df_runs, KPIS_TO_PLOT)
    sample_summary.to_csv(out_dir / "sample_level_kpis.csv", index=False)

    variant_summary = build_variant_summary(sample_summary, KPIS_TO_PLOT)
    variant_summary.to_csv(out_dir / "variant_level_kpis.csv", index=False)

    plot_heatmap_variant_medians(variant_summary, KPIS_TO_PLOT, out_dir)
    for kpi in KPIS_TO_PLOT:
        plot_boxplot_sample_means(sample_summary, kpi, out_dir)
        plot_quantile_interval(variant_summary, kpi, out_dir)
        plot_seed_std_boxplot(sample_summary, kpi, out_dir)
        plot_between_vs_within(variant_summary, kpi, out_dir)
        if REFERENCE_VARIANT in VARIANT_ORDER:
            plot_delta_vs_reference(sample_summary, kpi, out_dir)

    print("\nFinished.")
    print(f"Run-level KPIs:     {out_dir / 'run_level_kpis.csv'}")
    print(f"Sample-level KPIs:  {out_dir / 'sample_level_kpis.csv'}")
    print(f"Variant-level KPIs: {out_dir / 'variant_level_kpis.csv'}")
    print(f"Plots folder:       {out_dir}")


if __name__ == "__main__":
    main()
