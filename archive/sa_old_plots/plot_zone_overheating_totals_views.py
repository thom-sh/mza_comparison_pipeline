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

OUTPUT_DIR = RESULTS_ROOT / "_analysis_plots_fixed" / "zone_overheating_totals_views"
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
FIG_HEIGHT_PER_ROW = 4.0
SHARE_Y = False
ANNOTATE_P50 = False

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


def build_window_overheating_totals(df_profiles: pd.DataFrame) -> pd.DataFrame:
    if df_profiles.empty:
        return df_profiles.copy()

    aggregated = AGGREGATE_SEEDS_FIRST
    df_work = aggregate_across_seeds_first(df_profiles) if aggregated else df_profiles.copy()
    id_cols = profile_id_columns(df_work, aggregated)

    work = df_work.copy()
    work["overheat_h_step"] = np.where(
        work["tair_C"] > float(OVERHEATING_THRESHOLD_C),
        work["dt_hours"],
        0.0,
    )

    totals = (
        work
        .groupby(id_cols + ["zone_no", "zone_name"], dropna=False)["overheat_h_step"]
        .sum()
        .reset_index(name="overheating_hours")
        .sort_values(["zone_no"] + id_cols)
    )
    return totals


def summarise_window_totals(df_totals: pd.DataFrame) -> pd.DataFrame:
    if df_totals.empty:
        return df_totals.copy()

    summary = (
        df_totals
        .groupby(["zone_no", "zone_name"], dropna=False)["overheating_hours"]
        .agg(
            p05=lambda s: float(np.nanpercentile(s, 5)),
            p50=lambda s: float(np.nanpercentile(s, 50)),
            p95=lambda s: float(np.nanpercentile(s, 95)),
            mean="mean",
            n_profiles="count",
        )
        .reset_index()
        .sort_values("zone_no")
    )
    summary["uncertainty_band"] = summary["p95"] - summary["p05"]
    return summary


def add_grid(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.3)


def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def zone_tick_label(row: pd.Series) -> str:
    zno = int(row["zone_no"])
    zname = str(row["zone_name"])
    if zname and zname != f"Zone_{zno}":
        return f"Z{zno}\n{zname}"
    return f"Z{zno}"


def plot_point_ranges(
    summary_by_variant: dict[str, pd.DataFrame],
    n_profiles_by_variant: dict[str, int],
    window_name: str,
) -> Path:
    variants_to_plot = [v for v in VARIANT_ORDER if v in summary_by_variant]
    if not variants_to_plot:
        raise RuntimeError("No variant summaries available for plotting.")

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

    for ax_idx, variant in enumerate(variants_to_plot):
        ax = axes_flat[ax_idx]
        df = summary_by_variant[variant].sort_values("zone_no")
        if df.empty:
            ax.set_visible(False)
            continue

        x = np.arange(len(df), dtype=float)
        y = df["p50"].to_numpy(dtype=float)
        yerr_low = y - df["p05"].to_numpy(dtype=float)
        yerr_high = df["p95"].to_numpy(dtype=float) - y

        ax.errorbar(x, y, yerr=[yerr_low, yerr_high], fmt="o", capsize=5, linewidth=1.2)
        add_grid(ax)
        ax.set_title(f"{variant} (n_pr={n_profiles_by_variant[variant]})")
        ax.set_xticks(x)
        ax.set_xticklabels([zone_tick_label(r) for _, r in df.iterrows()], rotation=30, ha="right", fontsize=8)
        if ax_idx % N_COLS == 0:
            ax.set_ylabel("Overheating hours [h]")
        if ANNOTATE_P50:
            for xi, yi in zip(x, y):
                ax.text(xi, yi, f"{yi:.0f}", ha="center", va="bottom", fontsize=7)

    for ax in axes_flat[n_variants:]:
        ax.set_visible(False)

    fig.suptitle(
        f"Zone overheating totals (> {OVERHEATING_THRESHOLD_C:.1f}°C) – P5 / P50 / P95 – {window_name.replace('_', ' ')}",
        fontsize=14,
    )

    out_path = OUTPUT_DIR / f"point_ranges_{window_name}_all_variants_zone_overheating_totals.png"
    savefig(fig, out_path)
    return out_path


def build_stat_matrix(summary_by_variant: dict[str, pd.DataFrame], stat_col: str) -> pd.DataFrame:
    rows = []
    for variant in VARIANT_ORDER:
        df = summary_by_variant.get(variant)
        if df is None or df.empty:
            continue
        rec: dict[str, float | str] = {"variant": variant}
        for _, row in df.sort_values("zone_no").iterrows():
            rec[f"Z{int(row['zone_no'])}"] = float(row[stat_col])
        rows.append(rec)
    if not rows:
        return pd.DataFrame()
    mat = pd.DataFrame(rows).set_index("variant")
    zone_cols = sorted(
        [c for c in mat.columns if c.startswith("Z")],
        key=lambda s: int(s[1:]) if s[1:].isdigit() else 10**9,
    )
    return mat[zone_cols]


def plot_heatmaps(summary_by_variant: dict[str, pd.DataFrame], window_name: str) -> Path:
    mats = {
        "P05": build_stat_matrix(summary_by_variant, "p05"),
        "P50": build_stat_matrix(summary_by_variant, "p50"),
        "P95": build_stat_matrix(summary_by_variant, "p95"),
    }
    non_empty = [m for m in mats.values() if not m.empty]
    if not non_empty:
        raise RuntimeError("No matrices available for heatmap plotting.")

    vmax = max(float(np.nanmax(m.to_numpy(dtype=float))) for m in non_empty)
    vmax = max(vmax, 1.0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), constrained_layout=True)
    for ax, (title, mat) in zip(axes, mats.items()):
        if mat.empty:
            ax.set_visible(False)
            continue
        arr = mat.to_numpy(dtype=float)
        masked = np.ma.masked_invalid(arr)
        im = ax.imshow(masked, aspect="auto", vmin=0.0, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks(np.arange(mat.shape[1]))
        ax.set_xticklabels(list(mat.columns), rotation=30, ha="right")
        ax.set_yticks(np.arange(mat.shape[0]))
        ax.set_yticklabels(list(mat.index))
        ax.set_xlabel("Zone")
        if ax is axes[0]:
            ax.set_ylabel("Variant")

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = arr[i, j]
                text = "NA" if np.isnan(val) else f"{val:.0f}"
                ax.text(j, i, text, ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=axes, shrink=0.95)
    cbar.set_label("Overheating hours [h]")

    fig.suptitle(
        f"Zone overheating totals (> {OVERHEATING_THRESHOLD_C:.1f}°C) heatmaps – {window_name.replace('_', ' ')}",
        fontsize=14,
    )
    out_path = OUTPUT_DIR / f"heatmaps_p05_p50_p95_{window_name}_zone_overheating_totals.png"
    savefig(fig, out_path)
    return out_path


def export_window_csvs(summary_by_variant: dict[str, pd.DataFrame], totals_by_variant: dict[str, pd.DataFrame], window_name: str) -> None:
    combined_summary = []
    combined_totals = []

    for variant, df in summary_by_variant.items():
        if df.empty:
            continue
        out_csv = OUTPUT_DIR / "csv" / window_name / f"{variant}_zone_overheating_summary_{window_name}.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        tmp = df.copy()
        tmp.insert(0, "variant", variant)
        combined_summary.append(tmp)

    for variant, df in totals_by_variant.items():
        if df.empty:
            continue
        out_csv = OUTPUT_DIR / "csv" / window_name / f"{variant}_zone_overheating_profile_totals_{window_name}.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        tmp = df.copy()
        tmp.insert(0, "variant", variant)
        combined_totals.append(tmp)

    if combined_summary:
        pd.concat(combined_summary, ignore_index=True).to_csv(
            OUTPUT_DIR / "csv" / window_name / f"all_variants_zone_overheating_summary_{window_name}.csv",
            index=False,
        )
    if combined_totals:
        pd.concat(combined_totals, ignore_index=True).to_csv(
            OUTPUT_DIR / "csv" / window_name / f"all_variants_zone_overheating_profile_totals_{window_name}.csv",
            index=False,
        )


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
        summary_by_variant: dict[str, pd.DataFrame] = {}
        totals_by_variant: dict[str, pd.DataFrame] = {}

        for variant, df_profiles in raw_profiles_by_variant.items():
            df_window = select_window(df_profiles, window_spec)
            if df_window.empty:
                print(f"[INFO] Empty window for {variant}: {window_name}")
                continue

            totals_df = build_window_overheating_totals(df_window)
            summary_df = summarise_window_totals(totals_df)
            totals_by_variant[variant] = totals_df
            summary_by_variant[variant] = summary_df

        if not summary_by_variant:
            print(f"[INFO] No summaries available for window: {window_name}")
            continue

        export_window_csvs(summary_by_variant, totals_by_variant, window_name)

        point_path = plot_point_ranges(
            summary_by_variant=summary_by_variant,
            n_profiles_by_variant=n_profiles_by_variant,
            window_name=window_name,
        )
        heatmap_path = plot_heatmaps(summary_by_variant=summary_by_variant, window_name=window_name)
        saved_figures.extend([point_path, heatmap_path])
        print(f"Saved: {point_path}")
        print(f"Saved: {heatmap_path}")

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
