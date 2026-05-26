import pandas as pd
from pathlib import Path

def main():
    csv_path = r"C:\WF\Thomas Sharon\Floorplan_Dataset\gml_msd\comparison_output\all_global_stats.csv"
    threshold = 0.8

    df = pd.read_csv(csv_path)

    # Keep only rows with mean_iou_overall above the threshold
    filtered = df[df["mean_iou_overall"] > threshold].copy()

    # If your CSV has duplicate building IDs, keep the best row per building
    filtered = filtered.sort_values(
        by=["building_id", "mean_iou_overall"],
        ascending=[True, False]
    ).drop_duplicates(subset=["building_id"], keep="first")

    # Display results sorted by building_id
    filtered = filtered.sort_values(by="building_id")

    building_ids = filtered["building_id"].tolist()

    print(f"Buildings with mean_iou_overall > {threshold}:")
    print(building_ids)
    print(f"\nCount: {len(building_ids)}")


if __name__ == "__main__":
    main()
