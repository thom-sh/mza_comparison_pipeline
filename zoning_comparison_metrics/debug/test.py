import pandas as pd

GLOBAL_CSV = r"C:\WF\Thomas Sharon\Floorplan_Dataset\comparison_logic\all_global_stats_selected_200.csv"

df = pd.read_csv(GLOBAL_CSV)

bid = 2018  # change to your building ID

row = df[df["building_id"] == bid].copy()
row["zone_count_diff"] = row["n_pred_regions"] - row["n_gt_regions"]

print(row[[
    "building_id",
    "n_gt_regions",
    "n_pred_regions",
    "zone_count_diff",
    "false_negatives",
    "false_positives",
    "region_recall",
    "region_precision",
    "mean_iou_overall",
    "mean_iou_matched"
]])