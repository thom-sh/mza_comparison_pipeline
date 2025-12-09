# =====================================================================
#   MAIN LOOP — Evaluate Multiple Buildings with Simple MBR Alignment
# =====================================================================

import os
from dataset_loaders import (
    load_gt_rooms_and_footprint,
    load_predicted_zone_polygons,
    compute_predicted_footprint,
)
from simple_alignment import align_simple, apply_transform, plot_alignment_with_overlay

from matrix import evaluate_building   # <-- the module we prepared

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

OUTPUT_DIR = (
    r"N:\9_SF-Public\Austausch\Thomas Sharon\Master_Thesis_Sharon"
    r"\Floorplan_Dataset\gml_msd\comparison_output"
)

building_ids = [75, 553, 1330]   # Add as many as you want


# -------------------------------------------------------
# MAIN LOOP
# -------------------------------------------------------
print("\n==============================================================")
print("      BUILDING COMPARISON TOOL (Swiss GT vs Predicted)")
print("==============================================================")

for b_id in building_ids:

    print(f"\n--------------------------------------------------------------")
    print(f"Processing Building ID: {b_id}")
    print("--------------------------------------------------------------")

    GT_BASE   = os.path.join(SWISS_DATASET_ROOT, "graph_out", f"{b_id}.pickle")
    PRED_PATH = os.path.join(PREDICTED_FOLDER, f"building_data_{b_id}.pkl")
    OUT_DIR = os.path.join(OUTPUT_DIR, f"comparison_{b_id}.")

    # ----------------------------------------
    # 1. Load data
    # ----------------------------------------
    gt_fp, gt_apts, gt_stairs = load_gt_rooms_and_footprint(GT_BASE)
    pred_fp = compute_predicted_footprint(PRED_PATH)
    pred_zones = load_predicted_zone_polygons(PRED_PATH)

    # ----------------------------------------
    # 2. Align predicted → GT using simple MBR
    # ----------------------------------------
    pred_fp_aligned, rot, dx, dy, origin = align_simple(pred_fp, gt_fp)

    # Transform every predicted zone polygon
    pred_zones_aligned = [
        apply_transform(z, rot, dx, dy, origin) for z in pred_zones
    ]

    # ----------------------------------------
    # 3. (Optional) Show alignment visualization
    # ----------------------------------------
    plot_alignment_with_overlay(gt_fp, pred_fp, pred_fp_aligned)

    # ----------------------------------------
    # 4. Evaluate IoU, metrics, similarity score, FP/FN
    # ----------------------------------------
    metrics, stats, iou_mat = evaluate_building(
        b_id,
        GT_BASE,
        PRED_PATH,
            )

    print("\n--- Global Stats ---")
    for k, v in stats.items():
        print(f"{k}: {v}")

print("\n===============================")
print("   ALL BUILDINGS PROCESSED")
print("===============================")
