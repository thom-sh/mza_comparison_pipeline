#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================
RESULTS_ROOT = Path(
    r"C:\WF\Thomas Sharon\Floorplan_Dataset\Sensitivity Analysis\results\sa_results"
)

OUTPUT_CSV = Path(
    r"C:\Sharon\mza_sensitivity_analysis\plotting_scripts\new_plots\output\check\peak_heating_startup_check.csv"
)

VARIANT_ORDER = ["V1", "V1_20HH", "V2", "V2_20HH", "V3", "V4", "V5", "V6", "V7", "V8"]

YEAR = 2021
SKIP_FIRST_HOURS = 24
# ============================================================


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
        "run_dir": str(ts_csv.parent),
        "timeseries_csv": str(ts_csv),
    }


def available_heat_zones(df: pd.DataFrame) -> list[int]:
    zone_numbers = []

    for col in df.columns:
        m = re.fullmatch(r"HeatDemand_(\d+)", col)
        if m:
            zone_numbers.append(int(m.group(1)))

    return sorted(set(zone_numbers))


def sum_heat_demand(df: pd.DataFrame, zone_numbers: list[int]) -> pd.Series:
    total = pd.Series(0.0, index=df.index, dtype=float)

    for z in zone_numbers:
        col = f"HeatDemand_{z}"
        if col in df.columns:
            total += pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return total


def build_datetime(df: pd.DataFrame) -> pd.DataFrame:
    if "time_s" not in df.columns:
        raise KeyError("Missing time_s column")

    out = df.copy()
    base = pd.Timestamp(f"{YEAR}-01-01 00:00:00")
    out["datetime"] = base + pd.to_timedelta(out["time_s"], unit="s")
    return out


def check_one_run(ts_csv: Path) -> dict:
    meta = parse_run_from_path(ts_csv)

    df = pd.read_csv(ts_csv)
    df = build_datetime(df)

    zone_numbers = available_heat_zones(df)
    if not zone_numbers:
        raise RuntimeError(f"No HeatDemand_<zone> columns found in {ts_csv}")

    heat_total_W = sum_heat_demand(df, zone_numbers)
    heat_total_kW = heat_total_W / 1000.0

    peak_idx = heat_total_kW.idxmax()
    peak_value = float(heat_total_kW.loc[peak_idx])
    peak_time = df.loc[peak_idx, "datetime"]
    peak_time_s = float(df.loc[peak_idx, "time_s"])

    first_idx = df.index[0]
    first_time = df.loc[first_idx, "datetime"]

    skip_limit = first_time + pd.Timedelta(hours=SKIP_FIRST_HOURS)
    mask_after_skip = df["datetime"] >= skip_limit

    if mask_after_skip.any():
        heat_after_skip = heat_total_kW.loc[mask_after_skip]
        peak_skip_idx = heat_after_skip.idxmax()
        peak_skip_value = float(heat_after_skip.loc[peak_skip_idx])
        peak_skip_time = df.loc[peak_skip_idx, "datetime"]
    else:
        peak_skip_value = np.nan
        peak_skip_time = pd.NaT

    reduction_kW = peak_value - peak_skip_value
    reduction_pct = (reduction_kW / peak_value * 100.0) if peak_value > 0 else np.nan

    out = dict(meta)
    out["n_timesteps"] = int(len(df))
    out["n_zones"] = int(len(zone_numbers))

    out["peak_heating_kW"] = peak_value
    out["peak_heating_datetime"] = str(peak_time)
    out["peak_heating_time_s"] = peak_time_s

    out["peak_is_first_timestep"] = bool(peak_idx == first_idx)
    out["peak_is_first_hour"] = bool(peak_time < first_time + pd.Timedelta(hours=1))
    out["peak_is_first_day"] = bool(peak_time < first_time + pd.Timedelta(days=1))

    out[f"peak_heating_kW_skip_first_{SKIP_FIRST_HOURS}h"] = peak_skip_value
    out[f"peak_heating_datetime_skip_first_{SKIP_FIRST_HOURS}h"] = str(peak_skip_time)

    out["peak_reduction_after_skip_kW"] = float(reduction_kW)
    out["peak_reduction_after_skip_pct"] = float(reduction_pct)

    return out


def main() -> None:
    timeseries_files = sorted(RESULTS_ROOT.rglob("timeseries.csv"))

    if not timeseries_files:
        raise FileNotFoundError(f"No timeseries.csv files found under {RESULTS_ROOT}")

    rows = []
    failed = []

    for i, ts_csv in enumerate(timeseries_files, start=1):
        try:
            rows.append(check_one_run(ts_csv))
        except Exception as e:
            failed.append(
                {
                    "timeseries_csv": str(ts_csv),
                    "error": str(e),
                }
            )

        if i % 100 == 0 or i == len(timeseries_files):
            print(f"Checked {i}/{len(timeseries_files)} runs")

    df = pd.DataFrame(rows)

    if not df.empty and "variant" in df.columns:
        df["variant"] = pd.Categorical(df["variant"], categories=VARIANT_ORDER, ordered=True)
        df = df.sort_values(["variant", "building_id", "weather_key", "sample_id", "seed"])

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    if failed:
        failed_csv = OUTPUT_CSV.with_name("peak_heating_startup_check_failed.csv")
        pd.DataFrame(failed).to_csv(failed_csv, index=False)
        print(f"Warning: {len(failed)} failed runs saved to {failed_csv}")

    print("\nFinished.")
    print(f"Saved: {OUTPUT_CSV}")

    if not df.empty:
        summary = (
            df.groupby("variant", observed=False)
            .agg(
                n_runs=("peak_heating_kW", "count"),
                n_first_timestep_peaks=("peak_is_first_timestep", "sum"),
                n_first_day_peaks=("peak_is_first_day", "sum"),
                median_peak_kW=("peak_heating_kW", "median"),
                median_peak_skip_kW=(f"peak_heating_kW_skip_first_{SKIP_FIRST_HOURS}h", "median"),
                median_reduction_pct=("peak_reduction_after_skip_pct", "median"),
            )
            .reset_index()
        )

        summary_csv = OUTPUT_CSV.with_name("peak_heating_startup_check_summary_by_variant.csv")
        summary.to_csv(summary_csv, index=False)
        print(f"Summary saved: {summary_csv}")
        print(summary)


if __name__ == "__main__":
    main()