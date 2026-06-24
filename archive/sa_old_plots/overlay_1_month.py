#!/usr/bin/env python3
from __future__ import annotations

import calendar
import json
import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = (SCRIPT_DIR / "../results/sa_results").resolve()

# Variants to compare together
SELECTED_VARIANTS = ["V1", "V3", "V8"]

# Time selection: choose ONE option
USE_MONTH = True
YEAR = 2021
MONTH = 1                    # 1=Jan ... 12=Dec

USE_TIMESTAMP_RANGE = False
TIMESTAMP = "2021-01-15 00:00,2021-01-21 23:00"

# Optional filters
BUILDING_ID_CONTAINS = None  # e.g. "DESHPDHK0000lce9"
WEATHER_FILTER = None        # e.g. "TRY_B"

# Quantile band settings
LOW_Q = 0.05                 # 0.05 / 0.95 = P5/P95
HIGH_Q = 0.95
AGGREGATE_SEEDS_FIRST = True # average seeds first, then quantiles across samples

# Visual settings
COLORS = {
    "V1": "tab:blue",
    "V2": "tab:orange",
    "V3": "tab:green",
    "V4": "tab:red",
    "V5": "tab:purple",
    "V6": "tab:brown",
    "V7": "tab:pink",
    "V8": "tab:gray",
}
BAND_VARIANTS = set(SELECTED_VARIANTS)   # e.g. {"V1", "V3", "V5"}
BAND_ALPHA = 0.10
LINE_ALPHA = 0.95
LINE_WIDTH = 2.0

# Output
DPI = 150
SHOW_PLOTS = True
FIGSIZE = (14, 6)
OUTPUT_DIR = RESULTS_ROOT / "_analysis_plots_fixed" / "overlay_timeseries"
OUTPUT_FILENAME = "overlay_variants_timeseries.png"
# ============================================================


class RunFiles:
    def __init__(self, run_dir: Path, timeseries_csv: Path, overall_json: Optional[Path]):
        self.run_dir = run_dir
        self.timeseries_csv = timeseries_csv
        self.overall_json = overall_json


def month_range(year: int, month: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    _, last_day = calendar.monthrange(year, month)
    start = pd.Timestamp(year=year, month=month, day=1, hour=0, minute=0, second=0)
    end = pd.Timestamp(year=year, month=month, day=last_day, hour=23, minute=59, second=59)
    return start, end


def load_json(path: Optional[Path]) -> dict:
    if path is None or not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_datetime_index(df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    year = int(metadata.get("year", YEAR))
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
    if "," in spec:
        left, right = [s.strip() for s in spec.split(",", 1)]
        start = parse_time_token(left, df)
        end = parse_time_token(right, df)
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

    if USE_MONTH:
        t_start, t_end = month_range(YEAR, MONTH)
        df = df[(df["datetime"] >= t_start) & (df["datetime"] <= t_end)].copy()
    elif USE_TIMESTAMP_RANGE and TIMESTAMP:
        df = parse_timestamp_filter(TIMESTAMP, df)

    if df.empty:
        raise RuntimeError("Selected time filter produced no rows.")

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
    pieces = []
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


def compute_quantile_profile(df_profiles: pd.DataFrame) -> pd.DataFrame:
    df_work = aggregate_across_seeds_first(df_profiles) if AGGREGATE_SEEDS_FIRST else df_profiles[["datetime", "heat_kW"]].copy()
    q = (
        df_work
        .groupby("datetime")["heat_kW"]
        .quantile([LOW_Q, 0.5, HIGH_Q])
        .unstack()
        .reset_index()
    )
    q.columns = ["datetime", "q_low", "q_med", "q_high"]
    return q


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    quantiles_by_variant = {}
    counts_by_variant = {}

    for variant in SELECTED_VARIANTS:
        runs = discover_variant_runs(RESULTS_ROOT, variant, BUILDING_ID_CONTAINS, WEATHER_FILTER)
        if not runs:
            print(f"[INFO] No runs found for {variant}")
            continue

        df_profiles = build_profiles_table(runs)
        q_df = compute_quantile_profile(df_profiles)

        if AGGREGATE_SEEDS_FIRST:
            n_profiles = int(
                aggregate_across_seeds_first(df_profiles)[
                    ["base_building_group", "weather_key", "sample_id"]
                ].drop_duplicates().shape[0]
            )
        else:
            n_profiles = int(
                df_profiles[
                    ["base_building_group", "weather_key", "sample_id", "seed"]
                ].drop_duplicates().shape[0]
            )

        quantiles_by_variant[variant] = q_df
        counts_by_variant[variant] = n_profiles
        q_df.to_csv(OUTPUT_DIR / f"{variant}_overlay_quantile_profile.csv", index=False)

    variants_to_plot = [v for v in SELECTED_VARIANTS if v in quantiles_by_variant]
    if not variants_to_plot:
        raise RuntimeError("No selected variants could be plotted.")

    fig, ax = plt.subplots(figsize=FIGSIZE)

    for variant in variants_to_plot:
        q_df = quantiles_by_variant[variant]
        c = COLORS.get(variant, None)
        if variant in BAND_VARIANTS:
            ax.fill_between(
                q_df["datetime"],
                q_df["q_low"],
                q_df["q_high"],
                color=c,
                alpha=BAND_ALPHA,
                linewidth=0,
                zorder=1,
            )

    for variant in variants_to_plot:
        q_df = quantiles_by_variant[variant]
        c = COLORS.get(variant, None)
        ax.plot(
            q_df["datetime"],
            q_df["q_med"],
            color=c,
            linewidth=LINE_WIDTH,
            alpha=LINE_ALPHA,
            label=f"{variant} (n_pr={counts_by_variant[variant]})",
            zorder=2,
        )

    band_label = f"Q{int(LOW_Q*100)}–Q{int(HIGH_Q*100)}"
    if USE_MONTH:
        title_period = f"{calendar.month_name[MONTH]} {YEAR}"
        out_name = f"overlay_variants_{YEAR}_{MONTH:02d}.png"
    else:
        title_period = TIMESTAMP if TIMESTAMP else "selected period"
        out_name = "overlay_variants_selected_period.png"

    ax.set_title(f"{title_period}: median total heating demand with {band_label} bands")
    ax.set_xlabel("Time")
    ax.set_ylabel("Heat demand [kW]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    out_path = OUTPUT_DIR / out_name
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    print(f"Saved: {out_path}")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()