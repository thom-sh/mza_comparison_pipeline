from dataset_loaders import (
    load_gt_rooms_and_footprint,
    load_predicted_zone_polygons,
    compute_predicted_footprint
)

from simple_alignment import align_simple, apply_transform, plot_alignment_with_overlay
import os

print("\n==============================================================")
print("      BUILDING COMPARISON TOOL (Swiss GT vs Predicted)")
print("==============================================================")

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------

SWISS_DATASET_ROOT = (
    r"N:\9_SF-Public\Austausch\Thomas Sharon\Master_Thesis_Sharon"
    r"\Floorplan_Dataset\archive\modified-swiss-dwellings-v2\train"
)

PREDICTED_FOLDER = (
    r"N:\9_SF-Public\Austausch\Thomas Sharon\Master_Thesis_Sharon"
    r"\Floorplan_Dataset\gml_msd\building_data"
)

building_ids = [696]   # You can put a list: [1, 2, 10, 68]


# -------------------------------------------------------
# MAIN LOOP
# -------------------------------------------------------
for b_id in building_ids:
    print(f"\n--------------------------------------------------------------")
    print(f"Processing Building ID: {b_id}")
    print("--------------------------------------------------------------")

    GT_BASE   = os.path.join(SWISS_DATASET_ROOT, "graph_out", f"{b_id}.pickle")
    PRED_PATH = os.path.join(PREDICTED_FOLDER, f"building_data_{b_id}.pkl")

    # Load data
    gt_fp = load_gt_rooms_and_footprint(GT_BASE)[0]
    pred_fp = compute_predicted_footprint(PRED_PATH)
    pred_zones = load_predicted_zone_polygons(PRED_PATH)

    # Align footprint
    pred_fp_aligned, rot, dx, dy, origin = align_simple(pred_fp, gt_fp)

    # Apply transform to all zones
    pred_zones_aligned = [
        apply_transform(z, rot, dx, dy, origin) for z in pred_zones
    ]

    # Visualize
    plot_alignment_with_overlay(gt_fp, pred_fp, pred_fp_aligned)
