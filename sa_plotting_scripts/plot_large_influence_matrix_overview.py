#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

INPUT_CSV = PROJECT_DIR / "sa_results" / "sa_main" / "run_level_kpis.csv"

OUTPUT_DIR = PROJECT_DIR / "output" / "influence_matrix_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 450
SHOW_PLOTS = False

VARIANT_ORDER = ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8"]

VARIANT_LABELS = {
    "V1": "1Z",
    "V1_20HH": "1Z/20H",
    "V2": "6Z/SW",
    "V2_20HH": "6Z/20H",
    "V3": "11Z/A",
    "V4": "11Z/B",
    "V5": "21Z/A",
    "V6": "21Z/B",
    "V7": "10Z/noSW",
    "V8": "11Z/SWh",
}

WEATHER_LABELS = {
    "TRY_A": "Napoli",
    "TRY_B": "Munich",
}

# Large overview: set to True if you want every numeric KPI found in the CSV.
PLOT_ALL_NUMERIC_KPIS = False

# Main KPIs for the paper/overview.
KPI_SPECS = {
    "annual_heating_kWh": {
        "label": "Annual heating demand",
        "unit": "MWh",
        "scale": 1.0 / 1000.0,
        "decimals": 1,
    },
    "peak_heating_kW": {
        "label": "Peak heating load",
        "unit": "kW",
        "scale": 1.0,
        "decimals": 1,
    },
    "overheating_hours_any_zone_gt_26C": {
        "label": "Overheating hours, any zone > 26 °C",
        "unit": "h",
        "scale": 1.0,
        "decimals": 0,
    },
    "mean_tair_C": {
        "label": "Mean air temperature",
        "unit": "°C",
        "scale": 1.0,
        "decimals": 2,
    },
    "max_tair_C": {
        "label": "Maximum air temperature",
        "unit": "°C",
        "scale": 1.0,
        "decimals": 2,
    },
    "mean_interzone_spread_C": {
        "label": "Mean interzonal spread",
        "unit": "K",
        "scale": 1.0,
        "decimals": 2,
    },
    "max_interzone_spread_C": {
        "label": "Maximum interzonal spread",
        "unit": "K",
        "scale": 1.0,
        "decimals": 2,
    },
}

# Continuous parameters are compared as high quartile vs low quartile.
# This is intentionally broad: the figure is an overview, not a causal decomposition.
CONTINUOUS_CONTRASTS = [
    ("YoC class: high vs low", "sa_tabula_year_class", 0.75, 0.25),
    ("WWR: high vs low", "wwr_factor", 0.75, 0.25),
    ("Gains: high vs low", "gains_scale", 0.75, 0.25),
    ("Mean setpoint: high vs low", "sa_tset_mean_K", 0.75, 0.25),
    ("Setpoint spread: high vs low", "sa_tset_spread_K", 0.75, 0.25),
]

# Categorical / scenario contrasts.
SCENARIO_CONTRASTS = [
    ("Weather: Munich vs Napoli", "weather_key", "TRY_B", "TRY_A"),
    ("Retrofit: retrofit vs standard", "sa_retrofit_state", "retrofit", "standard"),
]

# Spread/variability columns.
SPREAD_COLUMNS = [
    "Total P95–P05 spread",
    "Napoli P95–P05 spread",
    "Munich P95–P05 spread",
    "Standard P95–P05 spread",
    "Retrofit P95–P05 spread",
    "Seed variability",
]

# Baseline columns compare every variant to a selected reference variant.
REFERENCE_VARIANTS = [
    ("Variant vs 1Z", "V1"),
    ("Variant vs 11Z/A", "V3"),
]

# Values smaller than this are printed as 0 in annotation, but kept in CSV.
ANNOTATION_ZERO_EPS = 0.05


# ============================================================
# Optional BauSim style
# ============================================================

def apply_optional_style() -> None:
    try:
        from bausim_plot_style import apply_bausim_style
        apply_bausim_style(DPI)
    except Exception:
        plt.rcParams.update({
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
        })


# ============================================================
# Helpers
# ============================================================

def sort_variants(df: pd.DataFrame, col: str = "variant") -> pd.DataFrame:
    out = df.copy()
    known = [v for v in VARIANT_ORDER if v in set(out[col].astype(str))]
    extras = sorted(set(out[col].dropna().astype(str)) - set(known))
    order = known + extras
    out[col] = pd.Categorical(out[col].astype(str), categories=order, ordered=True)
    return out.sort_values(col).reset_index(drop=True)


def variant_labels(variants: list[str]) -> list[str]:
    return [VARIANT_LABELS.get(str(v), str(v)) for v in variants]


def safe_median(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return np.nan
    return float(values.median())


def safe_pctl(series: pd.Series, q: float) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return np.nan
    return float(np.nanpercentile(values, q))


def rel_change_percent(new_value: float, base_value: float) -> float:
    if not np.isfinite(new_value) or not np.isfinite(base_value) or abs(base_value) < 1e-12:
        return np.nan
    return 100.0 * (float(new_value) - float(base_value)) / abs(float(base_value))


def abs_change(new_value: float, base_value: float) -> float:
    if not np.isfinite(new_value) or not np.isfinite(base_value):
        return np.nan
    return float(new_value) - float(base_value)


def rel_spread_percent(p05_value: float, p95_value: float, p50_value: float) -> float:
    if not np.isfinite(p05_value) or not np.isfinite(p95_value) or not np.isfinite(p50_value):
        return np.nan
    if abs(p50_value) < 1e-12:
        return np.nan
    return 100.0 * (float(p95_value) - float(p05_value)) / abs(float(p50_value))


def abs_spread(p05_value: float, p95_value: float) -> float:
    if not np.isfinite(p05_value) or not np.isfinite(p95_value):
        return np.nan
    return float(p95_value) - float(p05_value)


def high_low_masks(grp: pd.DataFrame, column: str, high_q: float, low_q: float) -> tuple[pd.Series, pd.Series]:
    values = pd.to_numeric(grp[column], errors="coerce")
    if values.notna().sum() < 4:
        empty = pd.Series(False, index=grp.index)
        return empty, empty

    lo_thr = float(values.quantile(low_q))
    hi_thr = float(values.quantile(high_q))

    low_mask = values <= lo_thr
    high_mask = values >= hi_thr

    return high_mask, low_mask


def median_for_mask(grp: pd.DataFrame, mask: pd.Series, kpi: str) -> float:
    if mask is None or not mask.any():
        return np.nan
    return safe_median(grp.loc[mask, kpi])


def infer_available_kpis(df: pd.DataFrame) -> dict[str, dict]:
    if not PLOT_ALL_NUMERIC_KPIS:
        return {k: v for k, v in KPI_SPECS.items() if k in df.columns}

    exclude = {
        "sample_id", "seed", "year", "n_timesteps", "dt_hours", "n_zones",
        "wwr_factor", "gains_scale", "sa_tabula_year_class",
        "sa_tset_mean_K", "sa_tset_spread_K",
    }
    out = dict(KPI_SPECS)
    for col in df.columns:
        if col in out or col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            out[col] = {
                "label": col,
                "unit": "",
                "scale": 1.0,
                "decimals": 2,
            }
    return out


# ============================================================
# Matrix calculation
# ============================================================

@dataclass
class MatrixBundle:
    relative_percent: pd.DataFrame
    absolute: pd.DataFrame
    support: pd.DataFrame


def build_influence_matrix(df: pd.DataFrame, kpi: str) -> MatrixBundle:
    if kpi not in df.columns:
        raise KeyError(f"KPI column not found: {kpi}")

    variants = [v for v in VARIANT_ORDER if v in set(df["variant"].astype(str))]
    extras = sorted(set(df["variant"].dropna().astype(str)) - set(variants))
    variants = variants + extras

    columns = (
        [name for name, *_ in SCENARIO_CONTRASTS]
        + [name for name, *_ in CONTINUOUS_CONTRASTS]
        + SPREAD_COLUMNS
        + [name for name, _ in REFERENCE_VARIANTS]
    )

    rel = pd.DataFrame(index=variants, columns=columns, dtype=float)
    abs_ = pd.DataFrame(index=variants, columns=columns, dtype=float)
    support = pd.DataFrame(index=variants, columns=columns, dtype=object)

    overall_variant_medians = {
        variant: safe_median(df[df["variant"].astype(str) == str(variant)][kpi])
        for variant in variants
    }

    for variant in variants:
        grp = df[df["variant"].astype(str) == str(variant)].copy()
        if grp.empty:
            continue

        base_median = safe_median(grp[kpi])

        # Scenario contrasts: value A vs B.
        for label, col, new_value, base_value in SCENARIO_CONTRASTS:
            if col not in grp.columns:
                support.loc[variant, label] = "missing column"
                continue

            mask_new = grp[col].astype(str).str.lower() == str(new_value).lower()
            mask_base = grp[col].astype(str).str.lower() == str(base_value).lower()

            med_new = median_for_mask(grp, mask_new, kpi)
            med_base = median_for_mask(grp, mask_base, kpi)

            rel.loc[variant, label] = rel_change_percent(med_new, med_base)
            abs_.loc[variant, label] = abs_change(med_new, med_base)
            support.loc[variant, label] = f"n_new={int(mask_new.sum())}; n_base={int(mask_base.sum())}"

        # Continuous high-low contrasts.
        for label, col, high_q, low_q in CONTINUOUS_CONTRASTS:
            if col not in grp.columns:
                support.loc[variant, label] = "missing column"
                continue

            high_mask, low_mask = high_low_masks(grp, col, high_q, low_q)
            med_high = median_for_mask(grp, high_mask, kpi)
            med_low = median_for_mask(grp, low_mask, kpi)

            rel.loc[variant, label] = rel_change_percent(med_high, med_low)
            abs_.loc[variant, label] = abs_change(med_high, med_low)

            vals = pd.to_numeric(grp[col], errors="coerce")
            support.loc[variant, label] = (
                f"high≥q{int(high_q * 100)}={vals.quantile(high_q):.4g}; "
                f"low≤q{int(low_q * 100)}={vals.quantile(low_q):.4g}; "
                f"n_high={int(high_mask.sum())}; n_low={int(low_mask.sum())}"
            )

        # Total spread.
        p05 = safe_pctl(grp[kpi], 5)
        p50 = safe_pctl(grp[kpi], 50)
        p95 = safe_pctl(grp[kpi], 95)
        rel.loc[variant, "Total P95–P05 spread"] = rel_spread_percent(p05, p95, p50)
        abs_.loc[variant, "Total P95–P05 spread"] = abs_spread(p05, p95)
        support.loc[variant, "Total P95–P05 spread"] = f"p05={p05:.6g}; p50={p50:.6g}; p95={p95:.6g}; n={len(grp)}"

        # Conditional spreads.
        spread_filters = {
            "Napoli P95–P05 spread": ("weather_key", "TRY_A"),
            "Munich P95–P05 spread": ("weather_key", "TRY_B"),
            "Standard P95–P05 spread": ("sa_retrofit_state", "standard"),
            "Retrofit P95–P05 spread": ("sa_retrofit_state", "retrofit"),
        }
        for label, (col, value) in spread_filters.items():
            if col not in grp.columns:
                support.loc[variant, label] = "missing column"
                continue
            mask = grp[col].astype(str).str.lower() == str(value).lower()
            sub = grp.loc[mask]
            p05 = safe_pctl(sub[kpi], 5)
            p50 = safe_pctl(sub[kpi], 50)
            p95 = safe_pctl(sub[kpi], 95)
            rel.loc[variant, label] = rel_spread_percent(p05, p95, p50)
            abs_.loc[variant, label] = abs_spread(p05, p95)
            support.loc[variant, label] = f"p05={p05:.6g}; p50={p50:.6g}; p95={p95:.6g}; n={len(sub)}"

        # Seed variability: median CV across same physical/weather sample.
        id_cols = [c for c in ["building_id", "weather_key", "sample_id"] if c in grp.columns]
        if "seed" in grp.columns and id_cols:
            cvs = []
            abs_sds = []
            for _, sub in grp.groupby(id_cols, dropna=False):
                vals = pd.to_numeric(sub[kpi], errors="coerce").dropna()
                if vals.size >= 2:
                    mean = float(vals.mean())
                    sd = float(vals.std(ddof=1))
                    if abs(mean) > 1e-12:
                        cvs.append(100.0 * sd / abs(mean))
                    abs_sds.append(sd)
            rel.loc[variant, "Seed variability"] = float(np.nanmedian(cvs)) if cvs else np.nan
            abs_.loc[variant, "Seed variability"] = float(np.nanmedian(abs_sds)) if abs_sds else np.nan
            support.loc[variant, "Seed variability"] = f"n_groups={len(cvs)}"
        else:
            support.loc[variant, "Seed variability"] = "missing seed or grouping columns"

        # Variant baseline contrasts.
        for label, reference_variant in REFERENCE_VARIANTS:
            ref_median = overall_variant_medians.get(reference_variant, np.nan)
            this_median = overall_variant_medians.get(variant, np.nan)
            rel.loc[variant, label] = rel_change_percent(this_median, ref_median)
            abs_.loc[variant, label] = abs_change(this_median, ref_median)
            support.loc[variant, label] = (
                f"median_{variant}={this_median:.6g}; "
                f"median_{reference_variant}={ref_median:.6g}"
            )

    return MatrixBundle(relative_percent=rel, absolute=abs_, support=support)


# ============================================================
# Plotting
# ============================================================

def annotate_heatmap(ax: plt.Axes, data: np.ndarray, decimals: int = 1) -> None:
    n_rows, n_cols = data.shape
    for i in range(n_rows):
        for j in range(n_cols):
            value = data[i, j]
            if not np.isfinite(value):
                text = "–"
            else:
                shown = 0.0 if abs(value) < ANNOTATION_ZERO_EPS else value
                text = f"{shown:.{decimals}f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=6.5, color="black")


def plot_matrix(
    matrix: pd.DataFrame,
    kpi: str,
    spec: dict,
    suffix: str,
    cmap: str = "RdBu_r",
) -> Path:
    data = matrix.to_numpy(dtype=float)
    finite = data[np.isfinite(data)]

    if finite.size == 0:
        raise ValueError(f"No finite values for {kpi} / {suffix}")

    # Robust symmetric scale: ignore extreme outliers for readability.
    vmax = float(np.nanpercentile(np.abs(finite), 95))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.nanmax(np.abs(finite)))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    n_rows, n_cols = matrix.shape
    fig_w = max(13.0, 0.62 * n_cols + 4.5)
    fig_h = max(5.2, 0.45 * n_rows + 2.5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(matrix.columns.tolist(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(variant_labels(matrix.index.astype(str).tolist()))

    label = spec.get("label", kpi)
    unit = spec.get("unit", "")
    if suffix == "relative_percent":
        title = f"Influence overview matrix — {label} [%]"
        cbar_label = "Relative change / spread [%]"
        decimals = 1
    else:
        title = f"Influence overview matrix — {label} [{unit}]"
        cbar_label = f"Absolute change / spread [{unit}]".strip()
        decimals = int(spec.get("decimals", 1))

    ax.set_title(title, pad=12)
    ax.set_xlabel("Influence contrast / overview indicator")
    ax.set_ylabel("Zoning variant")

    # Grid lines.
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    annotate_heatmap(ax, data, decimals=decimals)

    cbar = fig.colorbar(im, ax=ax, shrink=0.78)
    cbar.set_label(cbar_label)
    cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=7))

    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.31, top=0.88)

    out_path = OUTPUT_DIR / f"influence_matrix_{kpi}_{suffix}.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)
    return out_path


def plot_combined_dashboard(relative_matrices: dict[str, pd.DataFrame], kpi_specs: dict[str, dict]) -> Path:
    # One large figure with one heatmap per selected KPI.
    n = len(relative_matrices)
    if n == 0:
        raise ValueError("No matrices available for dashboard.")

    ncols = 1
    nrows = n
    example = next(iter(relative_matrices.values()))
    fig_w = max(15.0, 0.62 * example.shape[1] + 5.0)
    fig_h = max(4.2 * nrows, 0.45 * example.shape[0] * nrows + 2.0)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), squeeze=False)

    # Shared scale across panels, robust.
    all_values = np.concatenate([
        m.to_numpy(dtype=float).ravel()
        for m in relative_matrices.values()
    ])
    finite = all_values[np.isfinite(all_values)]
    vmax = float(np.nanpercentile(np.abs(finite), 95)) if finite.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    im = None
    for ax, (kpi, matrix) in zip(axes.ravel(), relative_matrices.items()):
        data = matrix.to_numpy(dtype=float)
        im = ax.imshow(data, cmap="RdBu_r", norm=norm, aspect="auto")

        ax.set_xticks(np.arange(matrix.shape[1]))
        ax.set_xticklabels(matrix.columns.tolist(), rotation=45, ha="right", rotation_mode="anchor")
        ax.set_yticks(np.arange(matrix.shape[0]))
        ax.set_yticklabels(variant_labels(matrix.index.astype(str).tolist()))

        spec = kpi_specs.get(kpi, {})
        ax.set_title(spec.get("label", kpi), pad=8)
        ax.set_ylabel("Variant")

        ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.9)
        ax.tick_params(which="minor", bottom=False, left=False)

        annotate_heatmap(ax, data, decimals=1)

    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.65, pad=0.015)
        cbar.set_label("Relative change / spread [%]")

    fig.suptitle("Large influence overview matrix across variants, parameters and KPIs", y=0.995)
    fig.subplots_adjust(left=0.10, right=0.95, bottom=0.08, top=0.965, hspace=0.70)

    out_path = OUTPUT_DIR / "influence_matrix_dashboard_relative_percent.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)
    return out_path


# ============================================================
# Terminal output
# ============================================================

def print_matrix_block(title: str, matrix: pd.DataFrame) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    with pd.option_context("display.max_rows", 100, "display.max_columns", 100, "display.width", 240):
        print(matrix.round(2).to_string())


def write_readme(kpi_specs: dict[str, dict]) -> Path:
    path = OUTPUT_DIR / "README_influence_matrix.txt"
    text = f"""Influence matrix overview
=========================

Input file:
{INPUT_CSV}

Rows:
- Zoning variants.

Columns:
- Weather: Munich vs Napoli = median(KPI | TRY_B) - median(KPI | TRY_A), relative to Napoli.
- Retrofit: retrofit vs standard = median(KPI | retrofit) - median(KPI | standard), relative to standard.
- Continuous parameters: high quartile vs low quartile within each variant.
- P95-P05 spread columns: uncertainty range relative to P50.
- Seed variability: median coefficient of variation across repeated seeds for same variant/weather/sample.
- Variant vs 1Z / 11Z/A: median variant contrast relative to the selected baseline variant.

Important:
This is an overview/diagnostic matrix, not a causal decomposition. Continuous parameter columns are high-low contrasts from the run-level sample and can include correlations with other sampled inputs.

KPIs included:
{chr(10).join(f"- {k}: {v.get('label', k)}" for k, v in kpi_specs.items())}
"""
    path.write_text(text, encoding="utf-8")
    return path


# ============================================================
# Main
# ============================================================

def main() -> None:
    apply_optional_style()

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Cannot find input CSV: {INPUT_CSV}\n"
            "Place run_level_kpis.csv next to this script or edit INPUT_CSV."
        )

    df = pd.read_csv(INPUT_CSV)
    if "variant" not in df.columns:
        raise KeyError("Input CSV needs a 'variant' column.")

    df = sort_variants(df)
    kpi_specs = infer_available_kpis(df)

    if not kpi_specs:
        raise ValueError("No KPI columns selected/found.")

    print(f"Loaded {INPUT_CSV}")
    print(f"Rows: {len(df)}")
    print(f"Variants: {sorted(df['variant'].dropna().astype(str).unique())}")
    print(f"KPIs: {list(kpi_specs.keys())}")

    saved_paths: list[Path] = []
    dashboard_matrices: dict[str, pd.DataFrame] = {}

    for kpi, spec in kpi_specs.items():
        print("\n" + "#" * 100)
        print(f"KPI: {kpi} — {spec.get('label', kpi)}")
        print("#" * 100)

        bundle = build_influence_matrix(df, kpi)

        # Scale absolute matrix to plotting unit.
        scale = float(spec.get("scale", 1.0))
        abs_scaled = bundle.absolute * scale

        rel_csv = OUTPUT_DIR / f"influence_matrix_{kpi}_relative_percent.csv"
        abs_csv = OUTPUT_DIR / f"influence_matrix_{kpi}_absolute.csv"
        support_csv = OUTPUT_DIR / f"influence_matrix_{kpi}_support.csv"

        bundle.relative_percent.to_csv(rel_csv)
        abs_scaled.to_csv(abs_csv)
        bundle.support.to_csv(support_csv)

        print_matrix_block(f"{kpi} — relative matrix [%]", bundle.relative_percent)
        print_matrix_block(f"{kpi} — absolute matrix [{spec.get('unit', '')}]", abs_scaled)

        saved_paths.extend([rel_csv, abs_csv, support_csv])
        saved_paths.append(plot_matrix(bundle.relative_percent, kpi, spec, "relative_percent"))
        saved_paths.append(plot_matrix(abs_scaled, kpi, spec, "absolute"))

        # Keep only main paper KPIs in the large dashboard to avoid becoming unreadable.
        if kpi in [
            "annual_heating_kWh",
            "peak_heating_kW",
            "overheating_hours_any_zone_gt_26C",
        ]:
            dashboard_matrices[kpi] = bundle.relative_percent

    if dashboard_matrices:
        saved_paths.append(plot_combined_dashboard(dashboard_matrices, kpi_specs))

    saved_paths.append(write_readme(kpi_specs))

    print("\n" + "=" * 100)
    print("Saved files:")
    print("=" * 100)
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
