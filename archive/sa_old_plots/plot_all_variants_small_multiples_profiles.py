#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = (SCRIPT_DIR / "../results/sa_results").resolve()

VARIANT_ORDER = ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8"]
BUILDING_ID_CONTAINS = None
WEATHER_FILTER = None
TIMESTAMP = None

LOW_Q = 0.25
HIGH_Q = 0.75
AGGREGATE_SEEDS_FIRST = True

N_COLS = 2
FIG_WIDTH = 15
FIG_HEIGHT_PER_ROW = 3.8
SHARE_Y = True
DPI = 150
SHOW_PLOTS = True

OUTPUT_DIR = RESULTS_ROOT / "_analysis_plots_fixed" / "variant_profiles"
OUTPUT_FILENAME = "all_variants_small_multiples_heat_profiles.png"


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


def parse_time_token(token: str, df: pd.DataFrame) -> pd.Timestamp:
    token = token.strip()
    try:
        return pd.Timestamp(token)
    except Exception:
        value = float(token)
        seconds = value * 3600.0 if abs(value) <= 8784 else value
        base = df["datetime"].iloc[0] - pd.to_timedelta(float(df["time_s"].iloc[0]), unit="s")
        return base + pd.to_timedelta(seconds, unit="s")


def parse_timestamp_filter(spec: str, df: pd.DataFrame) -> pd.DataFrame:
    spec = spec.strip()
    if not spec:
        return df
    if "," in spec:
        left, right = [s.strip() for s in spec.split(",", 1)]
        start = parse_time_token(left, df)
        end = parse_time_token(right, df)
        if start > end:
            start, end = end, start
        return df[(df["datetime"] >= start) & (df["datetime"] <= end)].copy()
    ts = parse_time_token(spec, df)
    nearest_idx = (df["datetime"] - ts).abs().idxmin()
    return df.loc[[nearest_idx]].copy()


def safe_series(df: pd.DataFrame, column: str):
    if column not in df.columns:
        return None
    s = pd.to_numeric(df[column], errors="coerce")
    if s.isna().all():
        return None
    return s.astype(float)


def available_zone_numbers(df: pd.DataFrame) -> list[int]:
    nums = []
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


def discover_variant_runs(results_root: Path, variant: str, building_id_contains: Optional[str] = None, weather_filter: Optional[str] = None) -> list[RunFiles]:
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
        runs.append(RunFiles(run_dir=run_dir, timeseries_csv=ts_csv, overall_json=run_dir / "overall.json" if (run_dir / "overall.json").exists() else None))
    return runs


def load_run_profile(run_files: RunFiles) -> pd.DataFrame:
    meta_from_path = parse_run_from_path(run_files.timeseries_csv)
    overall = load_json(run_files.overall_json)
    df = pd.read_csv(run_files.timeseries_csv)
    df = build_datetime_index(df, overall)
    if TIMESTAMP:
        df = parse_timestamp_filter(TIMESTAMP, df)
        if df.empty:
            raise RuntimeError("Selected TIMESTAMP filter produced no rows.")
    out = pd.DataFrame({"datetime": df["datetime"], "heat_kW": total_heat_kW(df)})
    out["variant"] = meta_from_path["variant"]
    out["building_id"] = meta_from_path["building_id"]
    out["base_building_group"] = base_building_group_key(meta_from_path["building_id"])
    out["weather_key"] = meta_from_path["weather_key"]
    out["sample_id"] = meta_from_path["sample_id"]
    out["seed"] = meta_from_path["seed"]
    return out


def build_profiles_table(runs: list[RunFiles]) -> pd.DataFrame:
    pieces = []
    for i, rf in enumerate(runs, start=1):
        pieces.append(load_run_profile(rf))
        if i % 50 == 0 or i == len(runs):
            print(f"Loaded {i}/{len(runs)} runs")
    return pd.concat(pieces, ignore_index=True)


def aggregate_across_seeds_first(df_profiles: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["base_building_group", "weather_key", "sample_id", "datetime"]
    return df_profiles.groupby(group_cols, dropna=False)["heat_kW"].mean().reset_index(name="heat_kW")


def compute_quantile_profile(df_profiles: pd.DataFrame) -> pd.DataFrame:
    df_work = aggregate_across_seeds_first(df_profiles) if AGGREGATE_SEEDS_FIRST else df_profiles[["datetime", "heat_kW"]].copy()
    q = df_work.groupby("datetime")["heat_kW"].quantile([LOW_Q, 0.5, HIGH_Q]).unstack().reset_index()
    q.columns = ["datetime", "q_low", "q_med", "q_high"]
    return q


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    variant_quantiles = {}
    variant_counts = {}

    for variant in VARIANT_ORDER:
        runs = discover_variant_runs(RESULTS_ROOT, variant, BUILDING_ID_CONTAINS, WEATHER_FILTER)
        if not runs:
            print(f"[INFO] No runs found for {variant}")
            continue
        df_profiles = build_profiles_table(runs)
        q_df = compute_quantile_profile(df_profiles)
        if AGGREGATE_SEEDS_FIRST:
            n_profiles = int(aggregate_across_seeds_first(df_profiles)[["base_building_group", "weather_key", "sample_id"]].drop_duplicates().shape[0])
        else:
            n_profiles = int(df_profiles[["base_building_group", "weather_key", "sample_id", "seed"]].drop_duplicates().shape[0])
        variant_quantiles[variant] = q_df
        variant_counts[variant] = n_profiles
        q_df.to_csv(OUTPUT_DIR / f"{variant}_quantile_profile_small_multiple.csv", index=False)

    variants_to_plot = [v for v in VARIANT_ORDER if v in variant_quantiles]
    if not variants_to_plot:
        raise RuntimeError("No variant quantile profiles could be built.")

    n_variants = len(variants_to_plot)
    n_rows = math.ceil(n_variants / N_COLS)

    fig, axes = plt.subplots(n_rows, N_COLS, figsize=(FIG_WIDTH, FIG_HEIGHT_PER_ROW * n_rows), sharex=True, sharey=SHARE_Y, constrained_layout=True)
    axes_flat = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for ax_idx, variant in enumerate(variants_to_plot):
        ax = axes_flat[ax_idx]
        q_df = variant_quantiles[variant]
        ax.fill_between(q_df["datetime"], q_df["q_low"], q_df["q_high"], alpha=0.25)
        ax.plot(q_df["datetime"], q_df["q_med"], linewidth=1.8)
        ax.set_title(f"{variant} (n_pr={variant_counts[variant]})")
        ax.grid(True, alpha=0.3)
        if ax_idx % N_COLS == 0:
            ax.set_ylabel("Heat demand [kW]")
        if ax_idx >= (n_rows - 1) * N_COLS:
            ax.set_xlabel("Time")

    for ax in axes_flat[n_variants:]:
        ax.set_visible(False)

    band_label = f"Q{int(LOW_Q*100)}–Q{int(HIGH_Q*100)}"
    fig.suptitle(f"Median total heating-demand profiles with {band_label} band across variants", fontsize=14)

    out_path = OUTPUT_DIR / OUTPUT_FILENAME
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    print(f"Saved: {out_path}")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
