import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===============================================================
# CONFIG
# ===============================================================

# zoning_comparison_metrics/
PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent

DATASET = "rd"

OUTPUT_DIR = (
    PROJECT_DIR
    / "output"
    / DATASET
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GLOBAL_CSV = (
    REPO_DIR / "zoning_comparison" / "output" / DATASET 
    / "all_global_stats_rd.csv"
)

OUT_DIAGNOSTIC_CSV = (
    OUTPUT_DIR
    / "diagnostic_table_with_topology_rd.csv"
)

OUT_COUNTS_CSV = (
    OUTPUT_DIR
    / "diagnostic_category_counts_with_topology_rd.csv"
)

# Thresholds: adjust if needed
GOOD_IOU = 0.70
WEAK_IOU = 0.50
HIGH_AREA_ERROR = 40.0      # %
HIGH_CENTROID_DISTANCE = 3.0  # m


# ===============================================================
# DIAGNOSTIC CLASSIFICATION
# ===============================================================
def classify_case(row):
    """
    Classify one building using structural, topology, and geometry metrics.
    Priority:
      1. Missing/extra regions
      2. Core-apartment topology mismatch
      3. Geometry, area, and location quality
    """

    fn = row["false_negatives"]
    fp = row["false_positives"]
    zone_diff = row["n_pred_regions"] - row["n_gt_regions"]

    # ---------- 1. Structural mismatch ----------
    if fn > 0 and fp > 0 and zone_diff == 0:
        return "Structural mismatch"

    if fn > 1 or zone_diff < 0:
        return "Under-zoned"

    if fp > 1 or zone_diff > 0:
        return "Over-zoned"

    # ---------- 2. Topology mismatch ----------
    if "topology_correct" in row.index:
        if int(row["topology_correct"]) == 0:
            return "Correct count but topology mismatch"

    # ---------- 3. Geometric mismatch ----------
    if row["mean_iou_overall"] < WEAK_IOU:
        return "Weak geometric match"

    if row["mean_area_error"] > HIGH_AREA_ERROR:
        return "High area error / shifted zones"

    if row["mean_centroid_distance"] > HIGH_CENTROID_DISTANCE:
        return "Correct topology but shifted zones"

    if row["mean_iou_overall"] >= GOOD_IOU:
        return "Good geometric match"

    return "Moderate geometric match"

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ===============================================================
# THESIS PLOT STYLE
# ===============================================================
THESIS_COLORS = {
    "primary_blue": "#7FA6C9",
    "light_blue": "#DCEAF7",
    "mid_blue": "#AFCBE3",
    "core_grey": "#5F666D",
    "edge_grey": "#777D84",
    "legend_edge": "#BDC1C5",
    "light_grey": "#DBDBDB",
    "dark_grey": "#999999",
    "muted_red": "#B94A48",
    "muted_orange": "#C58A45",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.titleweight": "normal",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.linewidth": 0.8,
    "grid.color": "#D9D9D9",
    "grid.linewidth": 0.5,
    "grid.alpha": 1.0,
})


# ===============================================================
# STRUCTURAL STATUS FOR PLOTTING
# ===============================================================
def classify_structural_status(row):
    fn = row["false_negatives"]
    fp = row["false_positives"]
    zone_diff = row["zone_count_diff"]

    if fn > 0 and fp > 0 and zone_diff == 0:
        return "Structural mismatch"

    if fn > 1 or zone_diff < 0:
        return "Under-zoned"

    if fp > 1 or zone_diff > 0:
        return "Over-zoned"

    return "Correct zone count"


# ===============================================================
# PLOTS
# ===============================================================
def plot_diagnostics(df, out_dir=None):
    df = df.copy()

    if "zone_count_diff" not in df.columns:
        df["zone_count_diff"] = df["n_pred_regions"] - df["n_gt_regions"]

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    diagnosis_order = [
        "Good geometric match",
        "Moderate geometric match",
        "Weak geometric match",
        "High area error / shifted zones",
        "Correct topology but shifted zones",
        "Correct count but topology mismatch",
        "Under-zoned",
        "Over-zoned",
        "Structural mismatch",
    ]

    diagnosis_colors_1 = {
        "Good geometric match": THESIS_COLORS["primary_blue"],
        "Moderate geometric match": THESIS_COLORS["mid_blue"],
        "Weak geometric match": THESIS_COLORS["muted_orange"],
        "High area error / shifted zones": THESIS_COLORS["light_grey"],
        "Correct topology but shifted zones": THESIS_COLORS["dark_grey"],
        "Correct count but topology mismatch": THESIS_COLORS["core_grey"],
        "Under-zoned": "#BABABA",
        "Over-zoned": THESIS_COLORS["edge_grey"],
        "Structural mismatch": THESIS_COLORS["muted_red"],
    }

    diagnosis_colors_2 = {
        "Good geometric match": "#6FA9A6",
        "Moderate geometric match": "#A9D1CE",
        "High area error / shifted zones": "#D7ECEA",
        "Under-zoned": "#B8D8D5",
        "Over-zoned": "#7FB5B2",
        "Structural mismatch": "#285452",
    }

    diagnosis_markers = {
        "Good geometric match": "o",
        "Moderate geometric match": "o",
        "Weak geometric match": "D",
        "High area error / shifted zones": "^",
        "Correct topology but shifted zones": "v",
        "Correct count but topology mismatch": "P",
        "Under-zoned": "s",
        "Over-zoned": "X",
        "Structural mismatch": "o",
    }

    # -----------------------------------------------------------
    # PLOT 1: Diagnostic category counts (percentage representation)
    # -----------------------------------------------------------
    counts = (
        df["diagnosis"]
        .value_counts()
        .reindex(diagnosis_order, fill_value=0)
    )

    counts = counts[counts > 0]
    # percentages = counts / counts.sum() * 100

    fig, ax = plt.subplots(figsize=(5.8, 2.6))
    ax.set_axisbelow(True)

    y_pos = np.arange(len(counts))
    bar_colors = [diagnosis_colors_2[c] for c in counts.index]

    bars = ax.barh(
        y_pos,
        # percentages.values,
        counts.values,
        color=bar_colors,
        edgecolor=THESIS_COLORS["edge_grey"],
        linewidth=0.7,
        zorder=2,
    )

    # add percentage + count labels at the end of each bar
    for i, bar in enumerate(bars):
        # pct = percentages.iloc[i]
        cnt = counts.iloc[i]
        ax.text(
            bar.get_width() + 1.0,
            bar.get_y() + bar.get_height() / 2,
            f"{int(cnt)}",
            va="center",
            ha="left",
            fontsize=9,
            color="#333333",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(counts.index)
    ax.invert_yaxis()

    ax.set_xlabel("Number of validation cases", fontsize=10)
    ax.set_xlim(0, counts.max() * 1.20)
    # ax.set_title("Distribution of diagnostic categories", fontsize=10)

    # ax.set_xlim(0, 100)

    ax.grid(axis="x", zorder=0)
    ax.grid(axis="y", visible=False)

    # full box if wanted
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("black")

    ax.tick_params(axis="both", which="both", color="black", labelcolor="black")
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")

    fig.tight_layout()

    if out_dir is not None:
        fig.savefig(
            out_dir / "diagnostic_category_percentages_rd.pdf",
            bbox_inches="tight",
            dpi=300,
        )

    plt.show()

    # -----------------------------------------------------------
    # PLOT 2: Ranked mean IoU by diagnostic category
    # -----------------------------------------------------------
    df_sorted = df.sort_values("mean_iou_overall").reset_index(drop=True)

    if len(df_sorted) > 1:
        df_sorted["rank_percentile"] = np.linspace(0, 100, len(df_sorted))
    else:
        df_sorted["rank_percentile"] = 100

    fig, ax = plt.subplots(figsize=(5.8, 2.4))

    for diagnosis in diagnosis_order:
        part = df_sorted[df_sorted["diagnosis"] == diagnosis]

        if part.empty:
            continue

        ax.scatter(
            part["rank_percentile"],
            part["mean_iou_overall"],
            label=diagnosis,
            color=diagnosis_colors_2[diagnosis],
            marker=diagnosis_markers[diagnosis],
            s=30,
            alpha=0.85,
            edgecolors=THESIS_COLORS["edge_grey"],
            linewidths=0.4,
        )

    ax.axhline(
        WEAK_IOU,
        color="#C87545",
        linewidth=0.8,
        linestyle="--",
        label=f"Weak IoU threshold",
    )

    ax.axhline(
        GOOD_IOU,
        color="#E1B184",
        linewidth=0.8,
        linestyle="--",
        label=f"Good IoU threshold",
    )

    ax.set_xlabel("Validation cases ranked by mean overall IoU percentile [%]")
    ax.set_ylabel("Mean overall IoU")
    # ax.set_title("Ranked mean overall IoU by diagnostic category")
    ax.set_ylim(0, 1.02)

    ax.grid(True, alpha=0.35)

    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)

    ax.legend(
        frameon=True,
        edgecolor=THESIS_COLORS["legend_edge"],
        loc="lower right",
        fontsize=7,
    )

    fig.tight_layout()

    if out_dir is not None:
        fig.savefig(
            out_dir / "mean_iou_by_diagnostic_category_rd.pdf",
            bbox_inches="tight",
            dpi=300,
        )

    plt.show()

def main():
    df = pd.read_csv(GLOBAL_CSV)

    # Zone-count difference
    df["zone_count_diff"] = df["n_pred_regions"] - df["n_gt_regions"]

    # Diagnosis
    df["diagnosis"] = df.apply(classify_case, axis=1)
    OUT_FIG_DIR = OUTPUT_DIR / "figures"
    
    plot_diagnostics(df, OUT_FIG_DIR)

    # Keep useful columns for thesis appendix/table
    columns = [
        "building_id",
        "n_gt_regions",
        "n_pred_regions",
        "zone_count_diff",
        "false_negatives",
        "false_positives",
        "region_recall",
        "region_precision",
        "topology_correct",
        "n_gt_core_apartment_edges",
        "n_pred_core_apartment_edges",
        "n_missing_core_apartment_edges",
        "n_extra_core_apartment_edges",
        "mean_iou_overall",
        "mean_iou_matched",
        "mean_area_error",
        "mean_centroid_distance",
        "mean_fragmentation",
        "diagnosis",
    ]

    # Only keep columns that exist
    columns = [c for c in columns if c in df.columns]

    diagnostic_df = df[columns].copy()
    diagnostic_df.to_csv(OUT_DIAGNOSTIC_CSV, index=False)

    # Count diagnostic categories
    counts = (
        diagnostic_df["diagnosis"]
        .value_counts()
        .rename_axis("diagnostic_category")
        .reset_index(name="number_of_buildings")
    )

    counts.to_csv(OUT_COUNTS_CSV, index=False)

    print("\nDiagnostic category counts:")
    print(counts.to_string(index=False))

    print("\nSaved:")
    print(OUT_DIAGNOSTIC_CSV)
    print(OUT_COUNTS_CSV)


if __name__ == "__main__":
    main()