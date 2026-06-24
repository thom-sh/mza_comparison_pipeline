import pandas as pd
import matplotlib.pyplot as plt

GLOBAL_CSV = r"C:\WF\Thomas Sharon\Floorplan_Dataset\comparison_logic\all_global_stats_selected_200.csv"
df = pd.read_csv(GLOBAL_CSV)

df["zone_count_diff"] = df["n_pred_regions"] - df["n_gt_regions"]

counts = df["zone_count_diff"].value_counts().sort_index()

plt.figure(figsize=(6, 4))
plt.bar(counts.index.astype(str), counts.values, edgecolor="black")
plt.xlabel("Predicted regions − reference regions")
plt.ylabel("Number of buildings")
plt.title("Zone-count deviation across validation cases")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()