import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================================================
# CONFIG
# ===============================================================
GLOBAL_CSV = r"C:\WF\Thomas Sharon\Floorplan_Dataset\comparison_logic\all_global_stats_selected_200.csv"

# ===============================================================
# LOAD DATA
# ===============================================================
df = pd.read_csv(GLOBAL_CSV)

# ===============================================================
# 1️⃣ PERCENTILE CURVE — Mean IoU (overall)
# ===============================================================
iou_sorted = np.sort(df["mean_iou_overall"].values)
percentiles = np.linspace(0, 100, len(iou_sorted))

plt.figure(figsize=(6, 4))
plt.plot(percentiles, iou_sorted, marker="o", markersize=3)
plt.xlabel("Building Percentile")
plt.ylabel("Mean IoU (overall)")
plt.title("Percentile Curve of Mean Overall IoU")
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ===============================================================
# 2️⃣ HISTOGRAM — Distribution of Mean IoU (overall)
# ===============================================================
plt.figure(figsize=(6, 4))
plt.hist(df["mean_iou_overall"], bins=15, edgecolor="black")
plt.xlabel("Mean IoU (overall)")
plt.ylabel("Number of Buildings")
plt.title("Distribution of Mean Overall IoU")
plt.xlim(0, 1)
plt.tight_layout()
plt.show()

# # ===============================================================
# # 3️⃣ SCATTER — Mean IoU (overall) vs Region Recall
# # ===============================================================
# plt.figure(figsize=(6, 6))
# plt.scatter(df["region_recall"], df["mean_iou_overall"], alpha=0.7)

# plt.axhline(0.7, linestyle="--", linewidth=1)
# plt.axvline(0.9, linestyle="--", linewidth=1)

# plt.xlabel("Region Recall (completeness)")
# plt.ylabel("Mean IoU (matched)")
# plt.title("Geometric Accuracy vs Completeness")
# plt.xlim(0, 1)
# plt.ylim(0, 1)
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()

# # ===============================================================
# # 4️⃣ DISTRIBUTION OF IoU GAP (no building_id)
# # ===============================================================
# df["iou_gap"] = df["mean_iou_matched"] - df["mean_iou_overall"]

# plt.figure(figsize=(6, 4))
# plt.hist(df["iou_gap"], bins=15, edgecolor="black")
# plt.xlabel("IoU Gap (matched − overall)")
# plt.ylabel("Number of Buildings")
# plt.title("Penalty Due to Missed Regions")
# plt.tight_layout()
# plt.show()

# # ===============================================================
# # 5️⃣ HEATMAP — Metric correlation (no building axis misuse)
# # ===============================================================
# metrics = [
#     "mean_iou_matched",
#     "mean_iou_overall",
#     "region_recall",
#     "region_precision",
#     "mean_fragmentation",
# ]

# corr = df[metrics].corr()

# # plt.figure(figsize=(6, 5))
# sns.heatmap(corr, annot=True, cmap="viridis", vmin=-1, vmax=1)
# plt.title("Correlation Between Zoning Metrics")
# plt.tight_layout()
# plt.show()

# 6
df["zone_count_diff"] = df["n_pred_regions"] - df["n_gt_regions"]

df_sorted = df.sort_values("mean_iou_overall").reset_index(drop=True)
df_sorted["percentile"] = np.linspace(0, 100, len(df_sorted))

plt.figure(figsize=(6, 4))

# Correct zone count
mask_correct = df_sorted["zone_count_diff"] == 0
plt.scatter(
    df_sorted.loc[mask_correct, "percentile"],
    df_sorted.loc[mask_correct, "mean_iou_overall"],
    label="Correct zone count",
    alpha=0.8
)

# Under-zoning
mask_under = df_sorted["zone_count_diff"] < 0
plt.scatter(
    df_sorted.loc[mask_under, "percentile"],
    df_sorted.loc[mask_under, "mean_iou_overall"],
    label="Under-zoning",
    marker="x",
    alpha=0.9
)

# Over-zoning
mask_over = df_sorted["zone_count_diff"] > 0
plt.scatter(
    df_sorted.loc[mask_over, "percentile"],
    df_sorted.loc[mask_over, "mean_iou_overall"],
    label="Over-zoning",
    marker="^",
    alpha=0.9
)

plt.xlabel("Percentile of ranked validation cases")
plt.ylabel("Mean overall IoU [-]")
plt.title("Mean overall IoU by zone-count status")
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
