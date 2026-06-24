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

GRANULARITY_VARIANTS = ["V2", "V3", "V5"]
PAIR_ORDER = [("V3", "V2"), ("V5", "V3"), ("V5", "V2")]

SHOW_PLOTS = True
DPI = 300

COLOR_P05 = "#c9d6ea"
COLOR_P50 = "#7ea6d8"
COLOR_P95 = "#2f5f98"
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
            "heatmap_label": "Difference [MWh]",
            "scale": 1.0 / 1000.0,
            "decimals": 1,
        },
        {
            "key": "peak_heating_kW",
            "title": "Peak heating demand",
            "ylabel": "Peak heating demand [kW]",
            "delta_label": "Δ peak heating demand [kW]",
            "heatmap_label": "Difference [kW]",
            "scale": 1.0,
            "decimals": 1,
        },
    ]

    if "overheating_hours_any_zone_gt_26C" in df.columns:
        specs.append(
            {
                "key": "overheating_hours_any_zone_gt_26C",
                "title": "Overheating hours",
                "ylabel": "Overheating hours [h]",
                "delta_label": "Δ overheating hours [h]",
                "heatmap_label": "Difference [h]",
                "scale": 1.0,
                "decimals": 1,
            }
        )
    elif "overheating_hours_meanTair_gt_26C" in df.columns:
        specs.append(
            {
                "key": "overheating_hours_meanTair_gt_26C",
                "title": "Overheating hours",
                "ylabel": "Overheating hours [h]",
                "delta_label": "Δ overheating hours [h]",
                "heatmap_label": "Difference [h]",
                "scale": 1.0,
                "decimals": 1,
            }
        )

    if "mean_interzone_spread_C" in df.columns:
        specs.append(
            {
                "key": "mean_interzone_spread_C",
                "title": "Mean interzone spread",
                "ylabel": "Mean interzone spread [°C]",
                "delta_label": "Δ mean interzone spread [°C]",
                "heatmap_label": "Difference [°C]",
                "scale": 1.0,
                "decimals": 2,
            }
        )

    return specs


def build_sample_summary(df_runs: pd.DataFrame, kpi_keys: list[str]) -> pd.DataFrame:
    group_cols = ["base_building_group", "variant", "sample_id"]
    agg = {k: ["mean", "std", "min", "max"] for k in kpi_keys}
    df = df_runs.groupby(group_cols, dropna=False).agg(agg).reset_index()

    flat_cols = []
    for col in df.columns:
        if isinstance(col, tuple):
            left, right = col
            flat_cols.append(left if right == "" else f"{left}_{right}")
        else:
            flat_cols.append(col)
    df.columns = flat_cols
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
        rows.append(
            {
                "variant": variant,
                "p05": p5(s) * spec["scale"],
                "p50": p50(s) * spec["scale"],
                "p95": p95(s) * spec["scale"],
                "mean": float(s.mean()) * spec["scale"],
                "std": float(s.std(ddof=1)) * spec["scale"] if len(s) > 1 else 0.0,
                "n_samples": int(s.notna().sum()),
            }
        )
    return sort_variants(pd.DataFrame(rows), order=GRANULARITY_VARIANTS)


def build_pairwise_matrix(df_sample: pd.DataFrame, spec: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    mean_col = f"{spec['key']}_mean"
    pivot = df_sample.pivot_table(
        index=["base_building_group", "sample_id"],
        columns="variant",
        values=mean_col,
        aggfunc="first",
    )

    labels = [v for v in GRANULARITY_VARIANTS if v in pivot.columns]
    pivot = pivot[labels]

    mat = pd.DataFrame(index=labels, columns=labels, dtype=float)
    counts = pd.DataFrame(index=labels, columns=labels, dtype=float)

    for row_v in labels:
        for col_v in labels:
            diffs = (pivot[row_v] - pivot[col_v]).dropna()
            counts.loc[row_v, col_v] = int(len(diffs))
            mat.loc[row_v, col_v] = np.nan if len(diffs) == 0 else float(np.median(diffs)) * spec["scale"]

    return mat, counts


def build_delta_table(df_sample: pd.DataFrame, spec: dict) -> pd.DataFrame:
    mean_col = f"{spec['key']}_mean"
    rows = []
    for left, right in PAIR_ORDER:
        left_df = (
            df_sample.loc[df_sample["variant"] == left, ["base_building_group", "sample_id", mean_col]]
            .rename(columns={mean_col: "left_value"})
        )
        right_df = (
            df_sample.loc[df_sample["variant"] == right, ["base_building_group", "sample_id", mean_col]]
            .rename(columns={mean_col: "right_value"})
        )
        merged = left_df.merge(
            right_df,
            on=["base_building_group", "sample_id"],
            how="inner",
            validate="one_to_one",
        )
        if merged.empty:
            continue
        merged["pair"] = f"{left} - {right}"
        merged["delta"] = (merged["left_value"] - merged["right_value"]) * spec["scale"]
        rows.append(merged[["base_building_group", "sample_id", "pair", "delta"]])

    if not rows:
        return pd.DataFrame(columns=["base_building_group", "sample_id", "pair", "delta"])
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
    labels = df_quant["variant"].tolist()
    x = np.arange(len(labels), dtype=float)
    width = 0.24

    p05_vals = df_quant["p05"].to_numpy(dtype=float)
    p50_vals = df_quant["p50"].to_numpy(dtype=float)
    p95_vals = df_quant["p95"].to_numpy(dtype=float)

    bars_p05 = ax.bar(x - width, p05_vals, width=width, label="P5", color=COLOR_P05)
    bars_p50 = ax.bar(x,         p50_vals, width=width, label="P50", color=COLOR_P50)
    bars_p95 = ax.bar(x + width, p95_vals, width=width, label="P95", color=COLOR_P95)

    ax.set_title(spec["title"])
    ax.set_ylabel(spec["ylabel"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    add_grid(ax)

    ymax = ax.get_ylim()[1]
    offset = 0.012 * ymax
    for bars, values in ((bars_p05, p05_vals), (bars_p50, p50_vals), (bars_p95, p95_vals)):
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                format_value(value, spec["decimals"]),
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=7,
            )

    if add_legend:
        style_legend(ax)


def plot_delta_panel(ax: plt.Axes, df_delta: pd.DataFrame, spec: dict) -> None:
    data = []
    used_labels = []

    for left, right in PAIR_ORDER:
        pair_label = f"{left} - {right}"
        arr = df_delta.loc[df_delta["pair"] == pair_label, "delta"].dropna().values
        if len(arr) == 0:
            continue
        data.append(arr)
        used_labels.append(f"{left}−{right}")

    if not data:
        ax.text(0.5, 0.5, "No matched samples", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return

    ax.boxplot(data, labels=used_labels, showfliers=False)
    ax.axhline(0.0, linestyle="--", color="black", linewidth=1.0)
    ax.set_title("Matched-sample deltas")
    ax.set_ylabel(spec["delta_label"])
    add_grid(ax)


def plot_heatmap_panel(ax: plt.Axes, mat: pd.DataFrame, spec: dict) -> None:
    labels = list(mat.index)
    arr = mat.values.astype(float)

    if np.isfinite(arr).any():
        vmax = np.nanmax(np.abs(arr))
        if vmax == 0:
            vmax = 1.0
        vmin = -vmax
    else:
        vmax, vmin = 1.0, -1.0

    im = ax.imshow(arr, vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Column variant")
    ax.set_ylabel("Row variant")
    ax.set_title("Median pairwise difference\n(row − column)")

    for i in range(len(labels)):
        for j in range(len(labels)):
            value = arr[i, j]
            text = "NA" if np.isnan(value) else format_value(value, spec["decimals"])
            ax.text(j, i, text, ha="center", va="center", fontsize=8)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(spec["heatmap_label"])


def plot_combined_figure(
    specs: list[dict],
    quantiles_by_kpi: Dict[str, pd.DataFrame],
    pairwise_by_kpi: Dict[str, pd.DataFrame],
    deltas_by_kpi: Dict[str, pd.DataFrame],
) -> None:
    n_rows = len(specs)
    fig, axes = plt.subplots(
        n_rows,
        3,
        figsize=(15.5, max(4.0 * n_rows, 8.0)),
        constrained_layout=True,
    )

    if n_rows == 1:
        axes = np.array([axes])

    for i, spec in enumerate(specs):
        plot_quantile_panel(axes[i, 0], quantiles_by_kpi[spec["key"]], spec, add_legend=(i == 0))
        plot_delta_panel(axes[i, 1], deltas_by_kpi[spec["key"]], spec)
        plot_heatmap_panel(axes[i, 2], pairwise_by_kpi[spec["key"]], spec)

    savefig(fig, OUTPUT_DIR / "granularity_v2_v3_v5_combined_figure.pdf")


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

    df_sample = build_sample_summary(df_runs, kpi_keys)
    df_sample = df_sample[df_sample["variant"].isin(GRANULARITY_VARIANTS)].copy()

    quantile_tables: Dict[str, pd.DataFrame] = {}
    pairwise_tables: Dict[str, pd.DataFrame] = {}
    delta_by_kpi: Dict[str, pd.DataFrame] = {}

    quantile_rows = []
    pairwise_rows = []
    delta_rows = []

    for spec in specs:
        q_df = build_variant_quantiles(df_sample, spec)
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
                "n_samples": row["n_samples"],
            })

        pairwise_mat, pairwise_counts = build_pairwise_matrix(df_sample, spec)
        pairwise_tables[spec["key"]] = pairwise_mat

        for row_v in pairwise_mat.index:
            for col_v in pairwise_mat.columns:
                pairwise_rows.append({
                    "kpi": spec["key"],
                    "row_variant": row_v,
                    "col_variant": col_v,
                    "median_diff": pairwise_mat.loc[row_v, col_v],
                    "n_pairs": pairwise_counts.loc[row_v, col_v],
                })

        d_df = build_delta_table(df_sample, spec)
        d_df["kpi"] = spec["key"]
        delta_by_kpi[spec["key"]] = d_df
        delta_rows.append(d_df)

    quantiles_long = pd.DataFrame(quantile_rows)
    pairwise_long = pd.DataFrame(pairwise_rows)
    deltas_long = pd.concat(delta_rows, ignore_index=True) if delta_rows else pd.DataFrame()

    df_sample.to_csv(OUTPUT_DIR / "granularity_v2_v3_v5_sample_summary.csv", index=False)
    quantiles_long.to_csv(OUTPUT_DIR / "granularity_v2_v3_v5_quantiles.csv", index=False)
    pairwise_long.to_csv(OUTPUT_DIR / "granularity_v2_v3_v5_pairwise_median_diffs.csv", index=False)
    deltas_long.to_csv(OUTPUT_DIR / "granularity_v2_v3_v5_delta_long.csv", index=False)

    plot_combined_figure(
        specs=specs,
        quantiles_by_kpi=quantile_tables,
        pairwise_by_kpi=pairwise_tables,
        deltas_by_kpi=delta_by_kpi,
    )

    print("Saved:")
    print(OUTPUT_DIR / "granularity_v2_v3_v5_sample_summary.csv")
    print(OUTPUT_DIR / "granularity_v2_v3_v5_quantiles.csv")
    print(OUTPUT_DIR / "granularity_v2_v3_v5_pairwise_median_diffs.csv")
    print(OUTPUT_DIR / "granularity_v2_v3_v5_delta_long.csv")
    print(OUTPUT_DIR / "granularity_v2_v3_v5_combined_figure.pdf")


if __name__ == "__main__":
    main()
