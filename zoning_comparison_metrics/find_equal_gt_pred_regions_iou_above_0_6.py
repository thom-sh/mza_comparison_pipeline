import pandas as pd


def main():
    # ---- Hard-coded path ----
    csv_path = r"C:\WF\Thomas Sharon\Floorplan_Dataset\comparison_logic\all_global_stats_selected_200.csv"

    # ---- Read CSV ----
    df = pd.read_csv(csv_path)

    # ---- Optional deduplication: keep the best row per building by overall IoU ----
    # This is helpful if the CSV contains multiple rows for the same building.
    if "building_id" in df.columns:
        df = df.sort_values(by="mean_iou_overall", ascending=False)
        df = df.drop_duplicates(subset=["building_id"], keep="first")

    # ---- Filter buildings where:
    #      1) GT region count equals predicted region count
    #      2) mean_iou_overall > 0.6
    # ----
    filtered_df = df[
        (df["n_gt_regions"] != df["n_pred_regions"]) &
        (df["mean_iou_overall"] > 0.8)
    ].copy()

    # ---- Sort for clean display ----
    filtered_df = filtered_df.sort_values(by="building_id")

    building_ids = filtered_df["building_id"].astype(int).tolist()

    print("Count of buildings where n_gt_regions == n_pred_regions "
          "and mean_iou_overall > 0.6: "
          f"{len(building_ids)}")
    print("\nBuilding IDs:")
    print(building_ids)

    # Optional: print a small table
    print("\nDetailed rows:")
    print(
        filtered_df[
            ["building_id", "n_gt_regions", "n_pred_regions", "mean_iou_overall"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
