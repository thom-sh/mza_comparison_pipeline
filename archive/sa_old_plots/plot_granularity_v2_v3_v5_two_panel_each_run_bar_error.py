#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================
INPUT_CSV = Path(r"../new_plots/sa_results/run_level_kpis.csv")
OUTPUT_DIR = Path(r"../new_plots/output/granularity_v2_v3_v5")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Zoning granularity comparison: storey-level -> 2 zones/floor -> 4 zones/floor
GRANULARITY_VARIANTS = ["V2", "V3", "V5"]
PAIR_ORDER = [("V3", "V2"), ("V5", "V3"), ("V5", "V2")]

SHOW_PLOTS = True
DPI = 300

COLOR_P50 = "#7ea6d8"
COLOR_INTERVAL = "#2f5f98"
LEGEND_EDGE_COLOR = "#bdc1c5"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 8,
    "axes.titleweight": "normal",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})
# ============================================================


def base_building_group_key(building_id: str) -> str:
    s = str(building_id).strip()
    m = re.match(r"^(.*?)(?:[_-]?)(\d+)$", s)
    if not m:
        return s
    prefix = str(m.group(1)).strip()
    return prefix or s


def sort_variants(df: pd.DataFrame, col: str = "variant", order: list[str] | None = None) -> pd.DataFrame:
    categories = order if order is not None else GRANULARITY_VARIANTS
    out = df.copy()
    out[col] = pd.Categorical(out[col], categories=categories, ordered=True)
    return out.sort_values(col).reset_index(drop=True)


def p5(x: pd.Series) -> float:
    return float(np.nanpercentile(x, 5))


def p50(x: pd.Series) -> float:
    return float(np.nanpercentile(x, 50))


def p95(x: pd.Series) -> float:
    return float(np.nanpercentile(x, 95))


def add_grid(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", alpha=0.30)
    ax.set_axisbelow(True)


def savefig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def choose_kpis(df: pd.DataFrame) -> list[dict]:
    specs = [
        {
            "key": "annual_heating_kWh",
            "title": "Annual heating demand",
            "ylabel": "Annual heating demand [MWh]",
            "delta_label": "Δ annual heating demand [MWh]",
            "scale": 1.0 / 1000.0,
            "decimals": 1,
        },
        {
            "key": "peak_heating_kW",
            "title": "Peak heating demand",
            "ylabel": "Peak heating demand [kW]",
            "delta_label": "Δ peak heating demand [kW]",
            "scale": 1.0,
            "decimals": 1,
        },
    ]

    if "overheating_hours_any_zone_gt_26C" in df.columns:
        specs.append({
            "key": "overheating_hours_any_zone_gt_26C",
            "title": "Overheating hours",
            "ylabel": "Overheating hours [h]",
            "delta_label": "Δ overheating hours [h]",
            "scale": 1.0,
            "decimals": 1,
        })
    elif "overheating_hours_meanTair_gt_26C" in df.columns:
        specs.append({
            "key": "overheating_hours_meanTair_gt_26C",
            "title": "Overheating hours",
            "ylabel": "Overheating hours [h]",
            "delta_label": "Δ overheating hours [h]",
            "scale": 1.0,
            "decimals": 1,
        })

    if "mean_interzone_spread_C" in df.columns:
        specs.append({
            "key": "mean_interzone_spread_C",
            "title": "Mean interzone spread",
            "ylabel": "Mean interzone spread [°C]",
            "delta_label": "Δ mean interzone spread [°C]",
            "scale": 1.0,
            "decimals": 2,
        })

    return [spec for spec in specs if spec["key"] in df.columns]


def build_run_summary(df_runs: pd.DataFrame, kpi_keys: list[str]) -> pd.DataFrame:
    # One row = one simulation run for one variant.
    # IMPORTANT: no seed averaging is applied here.
    # Runs are matched later by base_building_group + sample_id + seed.
    keep_cols = ["base_building_group", "variant", "sample_id", "seed", *kpi_keys]
    df = df_runs[keep_cols].copy()

    # Keep the old *_mean column names so the plotting functions can be reused.
    # Here, *_mean means the KPI value of this individual run, not a seed average.
    for k in kpi_keys:
        df[f"{k}_mean"] = pd.to_numeric(df[k], errors="coerce")
        df[f"{k}_std"] = np.nan
        df[f"{k}_min"] = df[f"{k}_mean"]
        df[f"{k}_max"] = df[f"{k}_mean"]

    return sort_variants(df, order=GRANULARITY_VARIANTS)


def build_variant_quantiles(df_sample: pd.DataFrame, spec: dict) -> pd.DataFrame:
    mean_col = f"{spec['key']}_mean"
    rows = []
    for variant, grp in df_sample.groupby("variant", observed=False):
        if grp.empty:
            continue
        s = pd.to_numeric(grp[mean_col], errors="coerce").dropna()
        if s.empty:
            continue
        rows.append({
            "variant": variant,
            "p05": p5(s) * spec["scale"],
            "p50": p50(s) * spec["scale"],
            "p95": p95(s) * spec["scale"],
            "mean": float(s.mean()) * spec["scale"],
            "std": float(s.std(ddof=1)) * spec["scale"] if len(s) > 1 else 0.0,
            "n_runs": int(s.notna().sum()),
        })
    return sort_variants(pd.DataFrame(rows), order=GRANULARITY_VARIANTS)


def build_delta_table(df_run: pd.DataFrame, spec: dict) -> pd.DataFrame:
    mean_col = f"{spec['key']}_mean"
    match_cols = ["base_building_group", "sample_id", "seed"]
    rows = []

    for left, right in PAIR_ORDER:
        left_df = (
            df_run.loc[df_run["variant"] == left, [*match_cols, mean_col]]
            .rename(columns={mean_col: "left_value"})
        )
        right_df = (
            df_run.loc[df_run["variant"] == right, [*match_cols, mean_col]]
            .rename(columns={mean_col: "right_value"})
        )
        merged = left_df.merge(
            right_df,
            on=match_cols,
            how="inner",
            validate="one_to_one",
        )
        if merged.empty:
            continue

        merged["pair"] = f"{left} - {right}"
        merged["delta"] = (merged["left_value"] - merged["right_value"]) * spec["scale"]
        rows.append(merged[[*match_cols, "pair", "delta"]])

    if not rows:
        return pd.DataFrame(columns=["base_building_group", "sample_id", "seed", "pair", "delta"])
    return pd.concat(rows, ignore_index=True)


def style_legend(ax: plt.Axes) -> None:
    legend = ax.legend(frameon=True)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor(LEGEND_EDGE_COLOR)
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_alpha(1.0)


def format_value(value: float, decimals: int) -> str:
    return f"{value:,.{decimals}f}"


def plot_quantile_panel(ax: plt.Axes, df_quant: pd.DataFrame, spec: dict, add_legend: bool = False) -> None:
    """Left panel: median bars with asymmetric P5-P95 error bars."""
    labels = df_quant["variant"].tolist()
    x = np.arange(len(labels), dtype=float)

    p05_vals = df_quant["p05"].to_numpy(dtype=float)
    p50_vals = df_quant["p50"].to_numpy(dtype=float)
    p95_vals = df_quant["p95"].to_numpy(dtype=float)

    yerr_low = p50_vals - p05_vals
    yerr_high = p95_vals - p50_vals

    ax.bar(
        x,
        p50_vals,
        width=0.58,
        color=COLOR_P50,
        edgecolor="none",
        label="P50",
        zorder=2,
    )

    ax.errorbar(
        x,
        p50_vals,
        yerr=[yerr_low, yerr_high],
        fmt="none",
        ecolor="black",
        elinewidth=1.2,
        capsize=5,
        capthick=1.0,
        label="P5–P95 range",
        zorder=3,
    )

    ax.set_title(f"{spec['title']} by variant (P50 with P5/P95 error bars)")
    ax.set_ylabel(spec["ylabel"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    add_grid(ax)

    finite_vals = np.concatenate([
        np.array([0.0]),
        p05_vals[np.isfinite(p05_vals)],
        p50_vals[np.isfinite(p50_vals)],
        p95_vals[np.isfinite(p95_vals)],
    ])
    if finite_vals.size:
        ymin = float(np.nanmin(finite_vals))
        ymax = float(np.nanmax(finite_vals))
        pad = 0.08 * (ymax - ymin if ymax > ymin else abs(ymax) if ymax else 1.0)
        ax.set_ylim(max(0.0, ymin - pad), ymax + pad)

    for xi, p50_v in zip(x, p50_vals):
        ax.text(
            xi,
            p50_v + 0.01 * max(np.nanmax(p95_vals), 1.0),
            format_value(p50_v, spec["decimals"]),
            ha="left",
            va="bottom",
            fontsize=7,
        )

    if add_legend:
        style_legend(ax)


def plot_delta_panel(ax: plt.Axes, df_delta: pd.DataFrame, spec: dict) -> None:
    """Right panel: matched-run deltas for each pair."""
    data = []
    used_labels = []

    for left, right in PAIR_ORDER:
        pair_label = f"{left} - {right}"
        arr = df_delta.loc[df_delta["pair"] == pair_label, "delta"].dropna().values
        if len(arr) == 0:
            continue
        data.append(arr)
        used_labels.append(f"{left}-{right}")

    if not data:
        ax.text(0.5, 0.5, "No matched runs", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return

    ax.boxplot(data, labels=used_labels, showfliers=False)
    ax.axhline(0.0, linestyle="--", color="black", linewidth=1.0)
    ax.set_title("Matched-run deltas")
    ax.set_ylabel(spec["delta_label"])
    add_grid(ax)


def plot_combined_figure(
    specs: list[dict],
    quantiles_by_kpi: Dict[str, pd.DataFrame],
    deltas_by_kpi: Dict[str, pd.DataFrame],
) -> None:
    """Create one combined figure with two panels per KPI."""
    n_rows = len(specs)
    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(11.5, max(3.8 * n_rows, 5.0)),
        constrained_layout=True,
    )

    if n_rows == 1:
        axes = np.array([axes])

    for i, spec in enumerate(specs):
        plot_quantile_panel(axes[i, 0], quantiles_by_kpi[spec["key"]], spec, add_legend=(i == 0))
        plot_delta_panel(axes[i, 1], deltas_by_kpi[spec["key"]], spec)

        axes[i, 0].text(-0.12, 1.04, "(a)", transform=axes[i, 0].transAxes, fontsize=9, fontweight="bold")
        axes[i, 1].text(-0.12, 1.04, "(b)", transform=axes[i, 1].transAxes, fontsize=9, fontweight="bold")

    savefig(fig, OUTPUT_DIR / "granularity_v2_v3_v5_two_panel_bar_error_figure.pdf")


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    df_runs = pd.read_csv(INPUT_CSV)
    if "base_building_group" not in df_runs.columns:
        df_runs["base_building_group"] = df_runs["building_id"].astype(str).apply(base_building_group_key)

    df_runs = df_runs[df_runs["variant"].isin(GRANULARITY_VARIANTS)].copy()

    specs = choose_kpis(df_runs)
    if not specs:
        raise RuntimeError("No supported KPI columns found in input CSV.")

    kpi_keys = [spec["key"] for spec in specs]
    required = {"building_id", "variant", "sample_id", "seed", *kpi_keys}
    missing = required.difference(df_runs.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    df_run = build_run_summary(df_runs, kpi_keys)
    df_run = df_run[df_run["variant"].isin(GRANULARITY_VARIANTS)].copy()

    quantile_tables: Dict[str, pd.DataFrame] = {}
    delta_by_kpi: Dict[str, pd.DataFrame] = {}

    quantile_rows = []
    delta_rows = []

    for spec in specs:
        q_df = build_variant_quantiles(df_run, spec)
        quantile_tables[spec["key"]] = q_df

        for _, row in q_df.iterrows():
            quantile_rows.append({
                "kpi": spec["key"],
                "variant": row["variant"],
                "p05": row["p05"],
                "p50": row["p50"],
                "p95": row["p95"],
                "mean": row["mean"],
                "std": row["std"],
                "n_runs": row["n_runs"],
            })

        d_df = build_delta_table(df_run, spec)
        d_df["kpi"] = spec["key"]
        delta_by_kpi[spec["key"]] = d_df
        delta_rows.append(d_df)

    quantiles_long = pd.DataFrame(quantile_rows)
    deltas_long = pd.concat(delta_rows, ignore_index=True) if delta_rows else pd.DataFrame()

    df_run.to_csv(OUTPUT_DIR / "granularity_v2_v3_v5_run_summary.csv", index=False)
    quantiles_long.to_csv(OUTPUT_DIR / "granularity_v2_v3_v5_quantiles.csv", index=False)
    deltas_long.to_csv(OUTPUT_DIR / "granularity_v2_v3_v5_delta_long.csv", index=False)

    plot_combined_figure(
        specs=specs,
        quantiles_by_kpi=quantile_tables,
        deltas_by_kpi=delta_by_kpi,
    )

    print("Saved:")
    print(OUTPUT_DIR / "granularity_v2_v3_v5_run_summary.csv")
    print(OUTPUT_DIR / "granularity_v2_v3_v5_quantiles.csv")
    print(OUTPUT_DIR / "granularity_v2_v3_v5_delta_long.csv")
    print(OUTPUT_DIR / "granularity_v2_v3_v5_two_panel_bar_error_figure.pdf")


if __name__ == "__main__":
    main()
