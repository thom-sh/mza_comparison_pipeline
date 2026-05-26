import pandas as pd
import matplotlib.pyplot as plt

# Load data
csv_path = r"C:\WF\Thomas Sharon\Floorplan_Dataset\gml_msd\comparison_output\all_global_stats.csv"
df = pd.read_csv(csv_path)

# Sort by mean IoU (matched)
df_sorted = df.sort_values("mean_iou_matched").reset_index(drop=True)

# Create index for x-axis
x = range(1, len(df_sorted) + 1)
y = df_sorted["mean_iou_matched"]

# Plot
plt.figure(figsize=(8, 4))
plt.plot(x, y, marker="o")
plt.ylim(0, 1)
plt.xlabel("Building rank (sorted by mean IoU)")
plt.ylabel("Mean IoU (matched)")
plt.title("Mean Matched IoU Across Buildings")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# Path to aggregated global stats CSV
csv_path = r"C:\WF\Thomas Sharon\Floorplan_Dataset\gml_msd\comparison_output\all_global_stats.csv"

# Load
df = pd.read_csv(csv_path)

# -------------------------------
# 1) Line plot over building rank
#    (sorted by recall)
# -------------------------------
df_sorted = df.sort_values("region_recall").reset_index(drop=True)

x = range(1, len(df_sorted) + 1)
y = df_sorted["region_recall"]

plt.figure(figsize=(8, 4))
plt.plot(x, y, marker="o")
plt.ylim(0, 1)
plt.xlabel("Building rank (sorted by recall)")
plt.ylabel("Region Recall")
plt.title("Region Recall Across Buildings")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# -------------------------------
# 2) (Optional) Bar plot by ID
#    (can be crowded if many)
# -------------------------------
df_id = df.sort_values("building_id")

plt.figure(figsize=(12, 4))
plt.bar(df_id["building_id"], df_id["region_recall"])
plt.ylim(0, 1)
plt.xlabel("Building ID")
plt.ylabel("Region Recall")
plt.title("Region Recall per Building")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# -------------------------------
# 3) (Optional) Histogram
# -------------------------------
plt.figure(figsize=(6, 4))
plt.hist(df["region_recall"], bins=15, edgecolor="black")
plt.xlim(0, 1)
plt.xlabel("Region Recall")
plt.ylabel("Number of Buildings")
plt.title("Distribution of Region Recall")
plt.tight_layout()
plt.show()
