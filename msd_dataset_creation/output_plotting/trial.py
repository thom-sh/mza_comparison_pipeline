import pandas as pd

GLOBAL_CSV = r"C:\WF\Thomas Sharon\Floorplan_Dataset\gml_msd\comparison_output\all_global_stats.csv"

df = pd.read_csv(GLOBAL_CSV)

# ----------------------------
# Thresholds (same logic as before)
# ----------------------------
RECALL_THRESHOLD = 0.9
FRAGMENTATION_THRESHOLD = df["mean_fragmentation"].quantile(0.75)

# ----------------------------
# Balanced buildings
# ----------------------------
balanced = df[
    (df["region_recall"] >= RECALL_THRESHOLD) &
    (df["mean_fragmentation"] <= FRAGMENTATION_THRESHOLD)
]

# Sort for clean output (identifier only, no ordinal meaning implied)
balanced_ids = balanced["building_id"].sort_values().tolist()

print("Balanced buildings (not over- or under-segmented):")
print(balanced_ids)

print(f"\nTotal balanced buildings: {len(balanced_ids)}")
