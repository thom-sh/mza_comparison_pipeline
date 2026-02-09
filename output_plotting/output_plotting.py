import pandas as pd
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
df = df.sort_values("building_id")

# ===============================================================
# 1️⃣ SORTED BAR PLOT — Mean IoU (matched)
# ===============================================================
df_sorted = df.sort_values("mean_iou_matched")

plt.figure(figsize=(12, 5))
plt.bar(df_sorted["building_id"], df_sorted["mean_iou_matched"])
plt.ylim(0, 1)
plt.xlabel("Building ID")
plt.ylabel("Mean IoU (matched)")
plt.title("Mean Matched IoU per Building (sorted)")
plt.xticks(rotation=90)
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
plt.tight_layout()
plt.show()

# ===============================================================
# 3️⃣ SCATTER — Mean IoU (matched) vs Region Recall
# ===============================================================
plt.figure(figsize=(6, 5))
plt.scatter(df["region_recall"], df["mean_iou_matched"])
plt.xlabel("Region Recall")
plt.ylabel("Mean IoU (matched)")
plt.title("Geometric Accuracy vs Completeness")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ===============================================================
# 4️⃣ DIFFERENCE PLOT — Matched vs Overall IoU
# ===============================================================
df["iou_gap"] = df["mean_iou_matched"] - df["mean_iou_overall"]

plt.figure(figsize=(12, 4))
plt.bar(df["building_id"], df["iou_gap"])
plt.axhline(0, color="black", linewidth=0.8)
plt.xlabel("Building ID")
plt.ylabel("IoU Gap (matched − overall)")
plt.title("Penalty Due to Missed Regions")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# ===============================================================
# 5️⃣ BOXPLOT — By Archetype (only if available)
# ===============================================================
if "archetype" in df.columns:
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df, x="archetype", y="mean_iou_matched")
    plt.xlabel("Building Archetype")
    plt.ylabel("Mean IoU (matched)")
    plt.title("Zoning Accuracy by Building Archetype")
    plt.tight_layout()
    plt.show()
else:
    print("[INFO] No 'archetype' column found — skipping boxplot.")

# ===============================================================
# 6️⃣ HEATMAP — Overview of Key Metrics (Appendix-style)
# ===============================================================
metrics = [
    "mean_iou_matched",
    "mean_iou_overall",
    "region_recall",
    "region_precision",
    "mean_fragmentation",
]

df_hm = df.set_index("building_id")[metrics]

plt.figure(figsize=(8, 6))
sns.heatmap(
    df_hm,
    cmap="viridis",
    cbar_kws={"label": "Metric value"},
    linewidths=0.2
)
plt.title("Overview of Zoning Performance Metrics")
plt.xlabel("Metric")
plt.ylabel("Building ID")
plt.tight_layout()
plt.show()