import pandas as pd

REGION_CSV = r"C:\WF\Thomas Sharon\Floorplan_Dataset\comparison_logic\all_region_metrics_selected_200.csv"

df = pd.read_csv(REGION_CSV)

# ----------------------------
# GT zone count per building
# ----------------------------
gt_counts = df.groupby("building_id")["region"].nunique()

# ----------------------------
# Predicted zone count per building
# (ignore NaNs in assigned_zone)
# ----------------------------
pred_counts = (
    df.dropna(subset=["assigned_zone"])
      .groupby("building_id")["assigned_zone"]
      .nunique()
)

# ----------------------------
# Combine into one table
# ----------------------------
comparison = pd.DataFrame({
    "gt_zone_count": gt_counts,
    "pred_zone_count": pred_counts
}).fillna(0)

comparison["pred_zone_count"] = comparison["pred_zone_count"].astype(int)

# ----------------------------
# Detect mismatch
# ----------------------------
comparison["zone_diff"] = (
    comparison["pred_zone_count"] - comparison["gt_zone_count"]
)

comparison["mismatch"] = comparison["zone_diff"] != 0

# ----------------------------
# Extract building IDs
# ----------------------------
mismatch_ids = comparison[comparison["mismatch"]].index.tolist()

print("Buildings with zone count mismatch:")
print(mismatch_ids)
print(f"\nTotal mismatched buildings: {len(mismatch_ids)}")
