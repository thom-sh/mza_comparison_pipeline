#!/usr/bin/env python3
"""Create runtime tables from the sensitivity-analysis ``overall.json`` files.

The script reads the runtime fields written by ``sim_wrapper.py``:
- simulation_wall_time_s
- total_wall_time_s

Outputs
-------
1. Run-level table: one row per completed seed run.
2. Sample-level table: runtime aggregated across seeds.
3. Variant summary: runtime distribution across selected samples.
4. Two sample-by-variant matrices for easy comparison.
5. A compact LaTeX variant-summary table.

Set ``N_SAMPLES_PER_VARIANT = 10`` for the first ten completed sample IDs of
 each variant. Set it to ``None`` to use all completed samples.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================
# SCRIPT_DIR = Path(__file__).resolve().parent
# REPO_ROOT = SCRIPT_DIR.parent

# # Expected main sensitivity-analysis results directory:
# # <repo>/results/sa_results/V1/.../overall.json
# RESULTS_ROOT = (SCRIPT_DIR / "sa_results").resolve()
RESULTS_ROOT = Path(r"C:\WF\Thomas Sharon\Floorplan_Dataset\Sensitivity Analysis\results\sa_results")

VARIANT_ORDER = ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8"]

# Use 10 for the first ten available sample IDs per variant.
# Use None to include every completed sample.
N_SAMPLES_PER_VARIANT: Optional[int] = None

OUTPUT_DIR = RESULTS_ROOT / "output" / "_runtime_tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Runtime columns written by sim_wrapper.py
SIM_RUNTIME_KEY = "simulation_wall_time_s"
TOTAL_RUNTIME_KEY = "total_wall_time_s"
# ============================================================================


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read JSON file: {path}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in: {path}")
    return data


def parse_int_suffix(value: str, prefix: str) -> Optional[int]:
    match = re.fullmatch(rf"{re.escape(prefix)}_(\d+)", str(value))
    return int(match.group(1)) if match else None


def parse_path_metadata(overall_path: Path) -> dict[str, Any]:
    """Parse variant/building/weather/sample/seed from the result path."""
    parts = list(overall_path.parts)

    sample_index: Optional[int] = None
    seed_index: Optional[int] = None

    for index, part in enumerate(parts):
        if re.fullmatch(r"sample_\d+", part):
            sample_index = index
        if re.fullmatch(r"seed_\d+", part):
            seed_index = index

    if sample_index is None or seed_index is None:
        raise ValueError(f"Could not parse sample/seed from: {overall_path}")

    if seed_index != sample_index + 1 or sample_index < 3:
        raise ValueError(f"Unexpected result-folder structure: {overall_path}")

    sample_id = parse_int_suffix(parts[sample_index], "sample")
    seed = parse_int_suffix(parts[seed_index], "seed")

    return {
        "variant": parts[sample_index - 3],
        "building_id": parts[sample_index - 2],
        "weather_key": parts[sample_index - 1],
        "sample_id": sample_id,
        "seed": seed,
        "run_dir": str(overall_path.parent),
        "overall_json": str(overall_path),
    }


def to_float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def collect_completed_runs(results_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []

    for overall_path in sorted(results_root.rglob("overall.json")):
        # Avoid recursively reading tables or unrelated analysis directories.
        if any(part.startswith("_") for part in overall_path.relative_to(results_root).parts):
            continue

        try:
            path_meta = parse_path_metadata(overall_path)
        except ValueError:
            # Ignore other overall.json files that do not follow the SA result structure.
            continue

        if path_meta["variant"] not in VARIANT_ORDER:
            continue

        try:
            overall = load_json(overall_path)
        except (RuntimeError, ValueError) as exc:
            errors.append(str(exc))
            continue

        task_meta = overall.get("task_meta", {}) or {}
        sample_meta = task_meta.get("sample", {}) or {}

        simulation_s = to_float_or_nan(overall.get(SIM_RUNTIME_KEY))
        total_s = to_float_or_nan(overall.get(TOTAL_RUNTIME_KEY))

        row = dict(path_meta)
        row.update(
            {
                "n_zones": overall.get("n_zones"),
                "expected_n_zones": overall.get("expected_n_zones"),
                "retrofit_state": task_meta.get("sa_retrofit_state"),
                "wwr_factor": sample_meta.get("wwr_factor"),
                "gains_scale": sample_meta.get("gains_scale"),
                "simulation_wall_time_s": simulation_s,
                "simulation_wall_time_min": simulation_s / 60.0,
                "total_wall_time_s": total_s,
                "total_wall_time_min": total_s / 60.0,
                "overhead_time_s": total_s - simulation_s,
                "overhead_time_min": (total_s - simulation_s) / 60.0,
            }
        )
        rows.append(row)

    if errors:
        error_log = OUTPUT_DIR / "runtime_table_read_errors.txt"
        error_log.write_text("\n".join(errors), encoding="utf-8")
        print(f"Warning: {len(errors)} JSON file(s) could not be read.")
        print(f"Error log: {error_log}")

    if not rows:
        raise RuntimeError(
            "No completed sensitivity-analysis runs were found.\n"
            f"Checked: {results_root}\n"
            "The script expects paths such as:\n"
            "  results/sa_results/V1/<building>/<weather>/sample_0000/seed_1/overall.json"
        )

    df = pd.DataFrame(rows)
    df["variant"] = pd.Categorical(
        df["variant"], categories=VARIANT_ORDER, ordered=True
    )

    numeric_columns = [
        "sample_id",
        "seed",
        "simulation_wall_time_s",
        "simulation_wall_time_min",
        "total_wall_time_s",
        "total_wall_time_min",
        "overhead_time_s",
        "overhead_time_min",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Only rows that actually contain both runtime values are usable.
    df = df.dropna(subset=["simulation_wall_time_s", "total_wall_time_s"])

    return df.sort_values(["variant", "sample_id", "seed"]).reset_index(drop=True)


def select_samples_per_variant(
    df_runs: pd.DataFrame,
    n_samples_per_variant: Optional[int],
) -> pd.DataFrame:
    """Select the first N available sample IDs separately for every variant."""
    selected_parts: list[pd.DataFrame] = []

    for variant in VARIANT_ORDER:
        group = df_runs.loc[df_runs["variant"] == variant].copy()
        if group.empty:
            continue

        sample_ids = sorted(group["sample_id"].dropna().astype(int).unique())
        if n_samples_per_variant is not None:
            sample_ids = sample_ids[: int(n_samples_per_variant)]

        selected_parts.append(group[group["sample_id"].isin(sample_ids)])

    if not selected_parts:
        raise RuntimeError("No variants contained usable runtime records.")

    return pd.concat(selected_parts, ignore_index=True).sort_values(
        ["variant", "sample_id", "seed"]
    )


def build_sample_table(df_runs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate repeated seeds so each row represents one variant/sample."""
    group_columns = ["variant", "building_id", "weather_key", "sample_id"]

    df = (
        df_runs.groupby(group_columns, observed=True, dropna=False)
        .agg(
            retrofit_state=("retrofit_state", "first"),
            n_zones=("n_zones", "first"),
            n_seed_runs=("seed", "nunique"),
            simulation_mean_min=("simulation_wall_time_min", "mean"),
            simulation_median_min=("simulation_wall_time_min", "median"),
            simulation_min_min=("simulation_wall_time_min", "min"),
            simulation_max_min=("simulation_wall_time_min", "max"),
            total_mean_min=("total_wall_time_min", "mean"),
            total_median_min=("total_wall_time_min", "median"),
            total_min_min=("total_wall_time_min", "min"),
            total_max_min=("total_wall_time_min", "max"),
            overhead_mean_min=("overhead_time_min", "mean"),
        )
        .reset_index()
    )

    df["variant"] = pd.Categorical(
        df["variant"], categories=VARIANT_ORDER, ordered=True
    )
    return df.sort_values(["variant", "sample_id"]).reset_index(drop=True)


def percentile(series: pd.Series, q: float) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    return float(np.percentile(values, q)) if len(values) else float("nan")


def build_variant_summary(
    df_sample: pd.DataFrame,
    df_runs: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise sample-level means, avoiding unequal weighting by seed count."""
    rows: list[dict[str, Any]] = []

    for variant in VARIANT_ORDER:
        samples = df_sample.loc[df_sample["variant"] == variant]
        runs = df_runs.loc[df_runs["variant"] == variant]
        if samples.empty:
            continue

        rows.append(
            {
                "variant": variant,
                "n_samples": int(samples["sample_id"].nunique()),
                "n_completed_runs": int(len(runs)),
                "mean_seed_runs_per_sample": float(samples["n_seed_runs"].mean()),
                "simulation_runtime_mean_min": float(samples["simulation_mean_min"].mean()),
                "simulation_runtime_median_min": float(samples["simulation_mean_min"].median()),
                "simulation_runtime_p05_min": percentile(samples["simulation_mean_min"], 5),
                "simulation_runtime_p95_min": percentile(samples["simulation_mean_min"], 95),
                "total_runtime_mean_min": float(samples["total_mean_min"].mean()),
                "total_runtime_median_min": float(samples["total_mean_min"].median()),
                "total_runtime_p05_min": percentile(samples["total_mean_min"], 5),
                "total_runtime_p95_min": percentile(samples["total_mean_min"], 95),
                "mean_overhead_min": float(samples["overhead_mean_min"].mean()),
                "cumulative_completed_runtime_h": float(runs["total_wall_time_s"].sum() / 3600.0),
            }
        )

    return pd.DataFrame(rows)


def build_sample_matrix(
    df_sample: pd.DataFrame,
    value_column: str,
) -> pd.DataFrame:
    matrix = df_sample.pivot_table(
        index="sample_id",
        columns="variant",
        values=value_column,
        aggfunc="first",
        observed=True,
    )
    available = [variant for variant in VARIANT_ORDER if variant in matrix.columns]
    return matrix.reindex(columns=available).sort_index()


def latex_escape(text: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def write_latex_summary(df: pd.DataFrame, path: Path) -> None:
    """Write a compact booktabs-compatible thesis table."""
    lines = [
        r"\begin{table}[tbh]",
        r"    \centering",
        r"    \caption{Simulation runtime by zoning variant}",
        r"    \label{tab:simulation_runtime_variants}",
        r"    \begin{tabular}{lrrrrr}",
        r"        \toprule",
        r"        Variant & Samples & Runs & Simulation [min] & Total [min] & Cumulative [h] \\",
        r"        \midrule",
    ]

    for _, row in df.iterrows():
        simulation = (
            f"{row['simulation_runtime_median_min']:.2f} "
            f"({row['simulation_runtime_p05_min']:.2f}--{row['simulation_runtime_p95_min']:.2f})"
        )
        total = (
            f"{row['total_runtime_median_min']:.2f} "
            f"({row['total_runtime_p05_min']:.2f}--{row['total_runtime_p95_min']:.2f})"
        )
        lines.append(
            "        "
            + " & ".join(
                [
                    latex_escape(str(row["variant"])),
                    str(int(row["n_samples"])),
                    str(int(row["n_completed_runs"])),
                    simulation,
                    total,
                    f"{row['cumulative_completed_runtime_h']:.2f}",
                ]
            )
            + r" \\" 
        )

    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{tabular}",
            r"    \begin{minipage}{0.95\textwidth}",
            r"        \footnotesize\textit{Note:} Values in parentheses are the P5--P95 interval across sample-level mean runtimes. Runtime was averaged across completed seeds before comparing samples.",
            r"    \end{minipage}",
            r"\end{table}",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def output_suffix(n_samples_per_variant: Optional[int]) -> str:
    return "all_samples" if n_samples_per_variant is None else f"first_{n_samples_per_variant}_samples"


def main() -> None:
    if not RESULTS_ROOT.exists():
        raise FileNotFoundError(f"Results directory not found: {RESULTS_ROOT}")

    df_all_runs = collect_completed_runs(RESULTS_ROOT)
    df_runs = select_samples_per_variant(df_all_runs, N_SAMPLES_PER_VARIANT)
    df_sample = build_sample_table(df_runs)
    df_variant = build_variant_summary(df_sample, df_runs)

    suffix = output_suffix(N_SAMPLES_PER_VARIANT)

    run_path = OUTPUT_DIR / f"runtime_run_level_{suffix}.csv"
    sample_path = OUTPUT_DIR / f"runtime_sample_level_{suffix}.csv"
    variant_path = OUTPUT_DIR / f"runtime_variant_summary_{suffix}.csv"
    total_matrix_path = OUTPUT_DIR / f"runtime_total_minutes_matrix_{suffix}.csv"
    simulation_matrix_path = OUTPUT_DIR / f"runtime_simulation_minutes_matrix_{suffix}.csv"
    latex_path = OUTPUT_DIR / f"runtime_variant_summary_{suffix}.tex"

    df_runs.to_csv(run_path, index=False)
    df_sample.to_csv(sample_path, index=False)
    df_variant.to_csv(variant_path, index=False)

    build_sample_matrix(df_sample, "total_mean_min").to_csv(total_matrix_path)
    build_sample_matrix(df_sample, "simulation_mean_min").to_csv(simulation_matrix_path)
    write_latex_summary(df_variant, latex_path)

    print("\nRuntime extraction completed.")
    print(f"Results root: {RESULTS_ROOT}")
    print(f"Completed runtime records found: {len(df_all_runs)}")
    print(f"Selected runtime records: {len(df_runs)}")
    print(f"Selected variant/sample combinations: {len(df_sample)}")
    print("\nSaved:")
    for path in [
        run_path,
        sample_path,
        variant_path,
        total_matrix_path,
        simulation_matrix_path,
        latex_path,
    ]:
        print(f"  {path}")

    print("\nVariant summary:")
    display_columns = [
        "variant",
        "n_samples",
        "n_completed_runs",
        "simulation_runtime_median_min",
        "total_runtime_median_min",
        "cumulative_completed_runtime_h",
    ]
    print(df_variant[display_columns].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
