import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================================================
# CONFIG
# ===============================================================
GLOBAL_CSV = r"C:\WF\Thomas Sharon\Floorplan_Dataset\gml_msd\comparison_output\all_global_stats.csv"

# ===============================================================
# LOAD DATA
# ===============================================================
df = pd.read_csv(GLOBAL_CSV)

# ===============================================================
# 1️⃣ PERCENTILE CURVE — Mean IoU (matched)
# ===============================================================
iou_sorted = np.sort(df["mean_iou_matched"].values)
percentiles = np.linspace(0, 100, len(iou_sorted))

plt.figure(figsize=(6, 4))
plt.plot(percentiles, iou_sorted, marker="o", markersize=3)
plt.xlabel("Building Percentile")
plt.ylabel("Mean IoU (matched)")
plt.title("Percentile Curve of Mean Matched IoU")
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ===============================================================
# 2️⃣ HISTOGRAM — Distribution of Mean IoU (matched)
# ===============================================================
plt.figure(figsize=(6, 4))
plt.hist(df["mean_iou_matched"], bins=15, edgecolor="black")
plt.xlabel("Mean IoU (matched)")
plt.ylabel("Number of Buildings")
plt.title("Distribution of Mean Matched IoU")
plt.xlim(0, 1)
plt.tight_layout()
plt.show()

# ===============================================================
# 3️⃣ SCATTER — Mean IoU (matched) vs Region Recall
# ===============================================================
plt.figure(figsize=(6, 6))
plt.scatter(df["region_recall"], df["mean_iou_matched"], alpha=0.7)

plt.axhline(0.7, linestyle="--", linewidth=1)
plt.axvline(0.9, linestyle="--", linewidth=1)

plt.xlabel("Region Recall (completeness)")
plt.ylabel("Mean IoU (matched)")
plt.title("Geometric Accuracy vs Completeness")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ===============================================================
# 4️⃣ DISTRIBUTION OF IoU GAP (no building_id)
# ===============================================================
df["iou_gap"] = df["mean_iou_matched"] - df["mean_iou_overall"]

plt.figure(figsize=(6, 4))
plt.hist(df["iou_gap"], bins=15, edgecolor="black")
plt.xlabel("IoU Gap (matched − overall)")
plt.ylabel("Number of Buildings")
plt.title("Penalty Due to Missed Regions")
plt.tight_layout()
plt.show()

# ===============================================================
# 5️⃣ HEATMAP — Metric correlation (no building axis misuse)
# ===============================================================
metrics = [
    "mean_iou_matched",
    "mean_iou_overall",
    "region_recall",
    "region_precision",
    "mean_fragmentation",
]

corr = df[metrics].corr()

plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, cmap="viridis", vmin=-1, vmax=1)
plt.title("Correlation Between Zoning Metrics")
plt.tight_layout()
plt.show()
