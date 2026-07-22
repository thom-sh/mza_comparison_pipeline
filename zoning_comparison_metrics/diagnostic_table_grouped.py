import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ===============================================================
# CONFIG
# ===============================================================

# If this script is inside zoning_comparison_metrics/
PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent

MSD_GLOBAL_CSV = (
    REPO_DIR / "zoning_comparison"
    / "output"
    / "msd"
    / "all_global_stats_msd.csv"
)

REAL_ESTATE_GLOBAL_CSV = (
    REPO_DIR / "zoning_comparison"
    / "output"
    / "rd"
    / "all_global_stats_rd.csv"
)

OUT_FIG_DIR = (
    PROJECT_DIR
    / "output"
    / "combined"
    / "figures"
)

OUT_PDF = "diagnostic_outcome_groups_msd_rd_grouped_percentage.pdf"

FIGSIZE = (6.4, 2.8)

# ===============================================================
# THRESHOLDS
# ===============================================================
GOOD_IOU = 0.70
WEAK_IOU = 0.50
HIGH_AREA_ERROR = 40.0        # %
HIGH_CENTROID_DISTANCE = 3.0  # m


# ===============================================================
# THESIS STYLE
# ===============================================================
THESIS_COLORS = {
    "good": "#6FA9A6",
    "moderate": "#A9D1CE",
    "failed": "#D7ECEA",
    "msd": "#6FA9A6",
    "real_estate": "#D7ECEA",
    "edge_grey": "#000000",
    "legend_edge": "#BDC1C5",
    "grid_grey": "#D9D9D9",
    "text_black": "#222222",
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
    "xtick.color": THESIS_COLORS["text_black"],
    "ytick.color": THESIS_COLORS["text_black"],
    "text.color": THESIS_COLORS["text_black"],
    "axes.labelcolor": THESIS_COLORS["text_black"],
    "grid.color": THESIS_COLORS["grid_grey"],
    "grid.linewidth": 0.5,
    "grid.alpha": 1.0,
})


# ===============================================================
# DIAGNOSTIC CLASSIFICATION
# ===============================================================
def classify_case(row):
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
        return "High area error/ Shifted zones"

    if row["mean_centroid_distance"] > HIGH_CENTROID_DISTANCE:
        return "Correct topology but shifted zones"

    if row["mean_iou_overall"] >= GOOD_IOU:
        return "Good geometric match"

    return "Moderate geometric match"


# ===============================================================
# AGGREGATE DIAGNOSTIC CATEGORIES
# ===============================================================
def aggregate_diagnosis(diagnosis):
    if diagnosis in [
        "Good geometric match",
        "Good geometric and topological match",
    ]:
        return "Good"

    elif diagnosis == "Moderate geometric match":
        return "Moderate"

    else:
        return "Failed"


# ===============================================================
# LOAD AND PREPARE DATA
# ===============================================================
def load_diagnostic_data(csv_path):
    df = pd.read_csv(csv_path)

    if "diagnosis" not in df.columns:
        df["diagnosis"] = df.apply(classify_case, axis=1)

    df["diagnosis_group"] = df["diagnosis"].apply(aggregate_diagnosis)

    return df


def get_group_percentages(df, group_order):
    counts = (
        df["diagnosis_group"]
        .value_counts()
        .reindex(group_order, fill_value=0)
    )

    percentages = counts / counts.sum() * 100
    return percentages


# ===============================================================
# PLOT
# ===============================================================
def plot_grouped_percentage_bars(msd_df, real_estate_df, out_dir=None):
    group_order = ["Good", "Moderate", "Failed"]

    msd_percent = get_group_percentages(msd_df, group_order)
    re_percent = get_group_percentages(real_estate_df, group_order)

    data_percent = pd.DataFrame(
        {
            "MSD": msd_percent,
            "Real Estate": re_percent,
        }
    )

    # -----------------------------------------------------------
    # X-axis = diagnostic outcome groups
    # Each group has two bars: MSD and Real Estate
    # -----------------------------------------------------------
    x_labels = group_order
    x = np.arange(len(x_labels))

    width = 0.22

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_axisbelow(True)

    # -----------------------------------------------------------
    # MSD bars
    # -----------------------------------------------------------
    bars_msd = ax.bar(
        x - width / 2,
        data_percent["MSD"].values,
        width=width,
        label="MSD",
        color=THESIS_COLORS["msd"],
        edgecolor=THESIS_COLORS["edge_grey"],
        linewidth=0.7,
        zorder=2,
    )

    # -----------------------------------------------------------
    # Real Estate bars
    # -----------------------------------------------------------
    bars_re = ax.bar(
        x + width / 2,
        data_percent["Real Estate"].values,
        width=width,
        label="German real estate",
        color=THESIS_COLORS["real_estate"],
        edgecolor=THESIS_COLORS["edge_grey"],
        linewidth=0.7,
        zorder=2,
    )

    # -----------------------------------------------------------
    # Percentage labels above bars
    # -----------------------------------------------------------
    for bars in [bars_msd, bars_re]:
        for bar in bars:
            value = bar.get_height()
            if value > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 1.2,
                    f"{value:.1f} %",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color=THESIS_COLORS["text_black"],
                )

    # -----------------------------------------------------------
    # Axes formatting
    # -----------------------------------------------------------
    ax.set_ylabel("Share of validation cases in %")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    
    ax.set_ylim(0, 100)

    ax.grid(axis="y", zorder=0)
    ax.grid(axis="x", visible=False)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("black")

    ax.tick_params(
        axis="both",
        which="both",
        color="black",
        labelcolor="black",
        width=0.8,
        length=3,
    )

    legend = ax.legend(
        frameon=True,
        fancybox=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor=THESIS_COLORS["legend_edge"],
        loc="upper right",
        ncol=1,
        borderpad=0.4,
        handlelength=1.8,
        handletextpad=0.6,
        labelspacing=0.4,
    )

    legend.get_frame().set_linewidth(0.8)

    fig.tight_layout()

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        fig.savefig(
            out_dir / OUT_PDF,
            bbox_inches="tight",
            dpi=300,
        )

        print("Saved:")
        print(out_dir / OUT_PDF)

    plt.show()

    print("\nPercentages:")
    print(data_percent.T.round(1))

# ===============================================================
# MAIN
# ===============================================================
def main():
    msd_df = load_diagnostic_data(MSD_GLOBAL_CSV)
    real_estate_df = load_diagnostic_data(REAL_ESTATE_GLOBAL_CSV)

    plot_grouped_percentage_bars(
        msd_df=msd_df,
        real_estate_df=real_estate_df,
        out_dir=OUT_FIG_DIR,
    )


if __name__ == "__main__":
    main()