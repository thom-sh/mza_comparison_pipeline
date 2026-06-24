#!/usr/bin/env python3
from __future__ import annotations

import json
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
AGGREGATE_SEEDS_FIRST = True

LOW_Q = 0.05
HIGH_Q = 0.95

DPI = 150
SHOW_PLOTS = True
FIGSIZE = (14, 6)
LINEWIDTH_P05 = 1.2
LINEWIDTH_P95 = 1.6
ALPHA = 0.95

OUTPUT_DIR = RESULTS_ROOT / "_analysis_plots_fixed" / "variant_profiles"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_START = pd.Timestamp("2021-11-17 00:00:00")
WINDOW_SPECS: list[tuple[str, pd.Timedelta]] = [
    ("1_day", pd.Timedelta(days=1)),
    ("3_days", pd.Timedelta(days=3)),
    ("1_month", pd.Timedelta(days=30)),
    ("3_months", pd.Timedelta(days=91)),
    # ("6_months", pd.Timedelta(days=182)),
]

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


def compute_variant_quantile_profile(df_profiles: pd.DataFrame) -> pd.DataFrame:
    if AGGREGATE_SEEDS_FIRST:
        df_work = aggregate_across_seeds_first(df_profiles)
    else:
        df_work = df_profiles[["datetime", "heat_kW"]].copy()

    q = (
        df_work
        .groupby("datetime", dropna=False)["heat_kW"]
        .quantile([LOW_Q, HIGH_Q])
        .unstack()
        .reset_index()
        .sort_values("datetime")
    )
    q.columns = ["datetime", "p05_heat_kW", "p95_heat_kW"]
    return q


def savefig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def _subset_window(q_df: pd.DataFrame, start: pd.Timestamp, duration: pd.Timedelta) -> pd.DataFrame:
    end = start + duration
    return q_df.loc[(q_df["datetime"] >= start) & (q_df["datetime"] < end)].copy()


def plot_windowed_overlay_both(
    quantiles_by_variant: dict[str, pd.DataFrame],
    counts_by_variant: dict[str, int],
    label: str,
    duration: pd.Timedelta,
) -> Path:
    fig, ax = plt.subplots(figsize=FIGSIZE)

    for variant in VARIANT_ORDER:
        q_df = quantiles_by_variant.get(variant)
        if q_df is None or q_df.empty:
            continue
        plot_df = _subset_window(q_df, WINDOW_START, duration)
        if plot_df.empty:
            continue

        c = COLORS.get(variant, None)
        ax.plot(
            plot_df["datetime"],
            plot_df["p05_heat_kW"],
            color=c,
            linestyle="--",
            linewidth=LINEWIDTH_P05,
            alpha=ALPHA,
            label=f"{variant} P5 (n_pr={counts_by_variant[variant]})",
        )
        ax.plot(
            plot_df["datetime"],
            plot_df["p95_heat_kW"],
            color=c,
            linestyle="-",
            linewidth=LINEWIDTH_P95,
            alpha=ALPHA,
            label=f"{variant} P95 (n_pr={counts_by_variant[variant]})",
        )

    ax.set_title(f"P5 and P95 total heating demand across all variants – {label.replace('_', ' ')}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Total building heat demand [kW]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", ncol=2)

    out_path = OUTPUT_DIR / f"all_variants_p05_p95_overlay_{label}.png"
    savefig(fig, out_path)
    return out_path


def plot_windowed_overlay_single(
    quantiles_by_variant: dict[str, pd.DataFrame],
    counts_by_variant: dict[str, int],
    label: str,
    duration: pd.Timedelta,
    quantile_col: str,
    quantile_label: str,
    linestyle: str,
    linewidth: float,
) -> Path:
    fig, ax = plt.subplots(figsize=FIGSIZE)

    for variant in VARIANT_ORDER:
        q_df = quantiles_by_variant.get(variant)
        if q_df is None or q_df.empty:
            continue
        plot_df = _subset_window(q_df, WINDOW_START, duration)
        if plot_df.empty:
            continue

        c = COLORS.get(variant, None)
        ax.plot(
            plot_df["datetime"],
            plot_df[quantile_col],
            color=c,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=ALPHA,
            label=f"{variant} (n_pr={counts_by_variant[variant]})",
        )

    ax.set_title(f"{quantile_label} total heating demand across all variants – {label.replace('_', ' ')}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Total building heat demand [kW]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", ncol=2)

    out_path = OUTPUT_DIR / f"all_variants_{quantile_col.replace('_heat_kW', '')}_overlay_{label}.png"
    savefig(fig, out_path)
    return out_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    quantiles_by_variant: dict[str, pd.DataFrame] = {}
    counts_by_variant: dict[str, int] = {}
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
        q_df = compute_variant_quantile_profile(df_profiles)
        quantiles_by_variant[variant] = q_df
        q_df.to_csv(OUTPUT_DIR / f"{variant}_p05_p95_profile.csv", index=False)

        if AGGREGATE_SEEDS_FIRST:
            df_count = aggregate_across_seeds_first(df_profiles)
            n_profiles = int(df_count[["base_building_group", "weather_key", "sample_id"]].drop_duplicates().shape[0])
        else:
            n_profiles = int(df_profiles[["base_building_group", "weather_key", "sample_id", "seed"]].drop_duplicates().shape[0])

        counts_by_variant[variant] = n_profiles
        summary_rows.append({
            "variant": variant,
            "n_runs": int(df_profiles[["base_building_group", "weather_key", "sample_id", "seed"]].drop_duplicates().shape[0]),
            "n_profiles_used": n_profiles,
            "n_timesteps": int(len(q_df)),
        })

    if not quantiles_by_variant:
        raise RuntimeError("No variant quantile profiles could be built.")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_DIR / "all_variants_p05_p95_overlay_summary.csv", index=False)

    saved: list[Path] = []
    for label, duration in WINDOW_SPECS:
        saved.append(plot_windowed_overlay_both(quantiles_by_variant, counts_by_variant, label, duration))
        saved.append(
            plot_windowed_overlay_single(
                quantiles_by_variant,
                counts_by_variant,
                label,
                duration,
                quantile_col="p05_heat_kW",
                quantile_label="P5",
                linestyle="--",
                linewidth=LINEWIDTH_P05,
            )
        )
        saved.append(
            plot_windowed_overlay_single(
                quantiles_by_variant,
                counts_by_variant,
                label,
                duration,
                quantile_col="p95_heat_kW",
                quantile_label="P95",
                linestyle="-",
                linewidth=LINEWIDTH_P95,
            )
        )

    print("Saved quantile CSVs per variant and figures:")
    for variant in quantiles_by_variant:
        print(OUTPUT_DIR / f"{variant}_p05_p95_profile.csv")
    print(OUTPUT_DIR / "all_variants_p05_p95_overlay_summary.csv")
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()
