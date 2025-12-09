# ===============================================================
#   SWISS vs PREDICTED BUILDING COMPARISON TOOL
# ===============================================================
import os
# from footprint_visualization import visualize_alignment, visualize_rooms_and_zones
# from footprint_visualization import align_pred
# print("ALIGN_PRED SOURCE:", align_pred.__code__.co_filename)


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

building_ids = [1330]   # You can put a list: [1, 2, 10, 68]


# -------------------------------------------------------
# MAIN LOOP
# -------------------------------------------------------
for b_id in building_ids:
    print(f"\n--------------------------------------------------------------")
    print(f"Processing Building ID: {b_id}")
    print("--------------------------------------------------------------")

    GT_BASE   = os.path.join(SWISS_DATASET_ROOT, "graph_out", f"{b_id}.pickle")
    PRED_PATH = os.path.join(PREDICTED_FOLDER, f"building_data_{b_id}.pkl")

    # ---------------------------------------------------
    # Validate paths
    # ---------------------------------------------------
    if not os.path.exists(GT_BASE):
        print(f"❌ ERROR: Swiss GT file not found:\n   {GT_BASE}\nSkipping…")
        continue

    if not os.path.exists(PRED_PATH):
        print(f"❌ ERROR: Predicted building file not found:\n   {PRED_PATH}\nSkipping…")
        continue

    # ---------------------------------------------------
    # Run visualizations
    # ---------------------------------------------------
    from footprint_visualization import align_pred, apply_transform

pred_fp_aligned, rot, dx, dy, origin = align_pred(PRED_PATH, GT_BASE)
pred_zones_aligned = [apply_transform(z, rot, dx, dy, origin) for z in pred_zones]


print("\n✔️ Finished processing all buildings.\n")
