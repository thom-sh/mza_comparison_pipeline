import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===============================================================
# CONFIG
# ===============================================================
GLOBAL_CSV = r"C:\WF\Thomas Sharon\Floorplan_Dataset\comparison_logic\all_global_stats_selected_200.csv"
OUT_CSV = r"C:\WF\Thomas Sharon\Floorplan_Dataset\comparison_logic\case_diagnostic_summary_200.csv"

# ===============================================================
# LOAD
# ===============================================================
df = pd.read_csv(GLOBAL_CSV)

# ===============================================================
# BASIC DIAGNOSTIC VARIABLES
# ===============================================================
df["zone_count_diff"] = df["n_pred_regions"] - df["n_gt_regions"]

def structural_status(row):
    if row["false_negatives"] > 0 and row["false_positives"] > 0:
        return "Mixed missing/extra"
    elif row["zone_count_diff"] < 0 or row["false_negatives"] > 0:
        return "Under-zoned / missing"
    elif row["zone_count_diff"] > 0 or row["false_positives"] > 0:
        return "Over-zoned / extra"
    else:
        return "Correct zone count"

df["structural_status"] = df.apply(structural_status, axis=1)

# ===============================================================
# GEOMETRIC STATUS
# Adjust thresholds if needed for your thesis interpretation
# ===============================================================
def geometric_status(row):
    if row["mean_iou_overall"] < 0.50:
        return "Weak geometry"
    elif row["mean_area_error"] > 40:
        return "High area error"
    elif row["mean_centroid_distance"] > 3:
        return "Shifted zones"
    elif row["mean_iou_overall"] >= 0.70:
        return "Good geometry"
    else:
        return "Moderate geometry"

df["geometric_status"] = df.apply(geometric_status, axis=1)

# ===============================================================
# FINAL DIAGNOSTIC CATEGORY
# ===============================================================
def diagnostic_category(row):
    if row["structural_status"] != "Correct zone count":
        return row["structural_status"]
    else:
        return row["geometric_status"]

df["diagnostic_category"] = df.apply(diagnostic_category, axis=1)

# Save one-row-per-building diagnostic table
df.to_csv(OUT_CSV, index=False)

# ===============================================================
# PLOT 1: Diagnostic category counts
# ===============================================================
order = [
    "Good geometry",
    "Moderate geometry",
    "Weak geometry",
    "High area error",
    "Shifted zones",
    "Under-zoned / missing",
    "Over-zoned / extra",
    "Mixed missing/extra",
]

counts = df["diagnostic_category"].value_counts()
counts = counts.reindex([c for c in order if c in counts.index])

plt.figure(figsize=(7, 4))
plt.barh(counts.index, counts.values, edgecolor="black")
plt.xlabel("Number of buildings")
plt.ylabel("Diagnostic category")
plt.title("Case-level diagnostic classification of validation results")
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.show()

# ===============================================================
# PLOT 2: Mean IoU coloured by structural status
# ===============================================================
df_sorted = df.sort_values("mean_iou_overall").reset_index(drop=True)
df_sorted["percentile"] = np.linspace(0, 100, len(df_sorted))

markers = {
    "Correct zone count": "o",
    "Under-zoned / missing": "x",
    "Over-zoned / extra": "^",
    "Mixed missing/extra": "s",
}

plt.figure(figsize=(7, 4))

for status, marker in markers.items():
    part = df_sorted[df_sorted["structural_status"] == status]
    plt.scatter(
        part["percentile"],
        part["mean_iou_overall"],
        label=status,
        marker=marker,
        alpha=0.75
    )

plt.xlabel("Percentile of ranked validation cases")
plt.ylabel("Mean overall IoU [-]")
plt.title("Mean overall IoU by structural zoning status")
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()