import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===============================================================
# PATHS
# ===============================================================
GLOBAL_CSV = r"C:\WF\Thomas Sharon\Floorplan_Dataset\comparison_logic\all_global_stats_selected_200.csv"
REGION_CSV = r"C:\WF\Thomas Sharon\Floorplan_Dataset\comparison_logic\all_region_metrics_selected_200.csv"

# ===============================================================
# LOAD DATA
# ===============================================================
df_g = pd.read_csv(GLOBAL_CSV)
df_r = pd.read_csv(REGION_CSV)

# ===============================================================
# ---------------- MAIN RESULTS ----------------
# ===============================================================

# ===============================================================
# 1️⃣ Percentile curve — Mean IoU (matched)
# ===============================================================
iou_sorted = np.sort(df_g["mean_iou_overall"].values)
percentiles = np.linspace(0, 100, len(iou_sorted))

plt.figure(figsize=(6, 4))
plt.plot(percentiles, iou_sorted, marker="o", markersize=3)
plt.xlabel("Building percentile")
plt.ylabel("Mean IoU (overall)")
plt.title("Percentile Curve of Mean Overall IoU")
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ===============================================================
# 2️⃣ ECDF — Region Recall
# ===============================================================
recall_sorted = np.sort(df_g["region_recall"].values)
ecdf = np.arange(1, len(recall_sorted) + 1) / len(recall_sorted)

plt.figure(figsize=(6, 4))
plt.step(recall_sorted, ecdf, where="post")
plt.xlabel("Region Recall")
plt.ylabel("Fraction of Buildings")
plt.title("ECDF of Region Recall")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ===============================================================
# 3️⃣ IoU vs Recall scatter (quadrants)
# ===============================================================
IOU_REF = 0.70
RECALL_REF = 0.90

x = df_g["region_recall"].values
y = df_g["mean_iou_overall"].values

plt.figure(figsize=(6, 6))
plt.scatter(x, y, alpha=0.6)

plt.axvline(RECALL_REF, linestyle="--", linewidth=1)
plt.axhline(IOU_REF, linestyle="--", linewidth=1)

plt.text(0.05, 0.95, "Low recall\nHigh IoU", va="top")
plt.text(0.95, 0.95, "High recall\nHigh IoU", va="top", ha="right")
plt.text(0.05, 0.05, "Low recall\nLow IoU", va="bottom")
plt.text(0.95, 0.05, "High recall\nLow IoU", va="bottom", ha="right")

plt.xlabel("Region Recall (completeness)")
plt.ylabel("Mean IoU (overall)")
plt.title("Geometric Accuracy vs Zoning Completeness")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ===============================================================
# ---------------- APPENDIX / SUPPORTING ----------------
# ===============================================================

# ===============================================================
# 4️⃣ Histogram — Mean IoU (matched)
# ===============================================================
plt.figure(figsize=(6, 4))
plt.hist(df_g["mean_iou_overall"], bins=15, edgecolor="black")
plt.xlabel("Mean IoU (overall)")
plt.ylabel("Number of Buildings")
plt.title("Distribution of Mean Overall IoU")
plt.xlim(0, 1)
plt.tight_layout()
plt.show()

# ===============================================================
# 5️⃣ Fragmentation vs IoU (region-level)
# ===============================================================
plt.figure(figsize=(6, 5))
plt.scatter(
    df_r["fragmentation"],
    df_r["iou"],
    alpha=0.4
)
plt.xlabel("Fragmentation (per GT region)")
plt.ylabel("IoU")
plt.title("Fragmentation vs Geometric Accuracy")
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ===============================================================
# 6️⃣ Bubble plot — Complexity effect (derived GT count)
# ===============================================================
# Derive GT region count per building from region CSV
gt_counts = (
    df_r.groupby("building_id")
        .size()
        .rename("n_gt_regions")
        .reset_index()
)

df_bubble = df_g.merge(gt_counts, on="building_id", how="left")

x = df_bubble["region_recall"].values
y = df_bubble["mean_iou_overall"].values
s_raw = df_bubble["n_gt_regions"].values

# Scale bubble size
s = 40 + 20 * (s_raw - s_raw.min()) / (s_raw.max() - s_raw.min() + 1e-9)

plt.figure(figsize=(6, 6))
plt.scatter(x, y, s=s, alpha=0.5)

plt.xlabel("Region Recall")
plt.ylabel("Mean IoU (overall)")
plt.title("Recall vs Mean Overall IoU\n(Bubble size = #GT regions)")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
