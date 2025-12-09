# =====================================================================
#   MERGED EVALUATION PIPELINE
#   - GT & prediction loading from Script 2
#   - ALIGNMENT using Script 2 (align_simple + apply_transform)
#   - FULL comparison logic from Script 1
# =====================================================================

import os
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon


# =====================================================================
# LOADERS (from Script 2)
# =====================================================================
from dataset_loaders import (
    load_gt_rooms_and_footprint,
    load_predicted_zone_polygons,
    compute_predicted_footprint,
)

# =====================================================================
# ALIGNMENT (USE SCRIPT 2 ONLY)
# =====================================================================
from simple_alignment import (
    align_simple,
    apply_transform,
)


# =====================================================================
# IoU + Overlap Matrix (from Script 1)
# =====================================================================

def compute_overlap_matrix(gt_regions, pred_zones):
    n_gt = len(gt_regions)
    n_pr = len(pred_zones)

    iou_mat = np.zeros((n_gt, n_pr))
    inter_mat = np.zeros((n_gt, n_pr))

    for i, g in enumerate(gt_regions):
        for j, p in enumerate(pred_zones):
            inter = g.intersection(p).area
            union = g.union(p).area
            iou_mat[i, j] = inter / union if union > 0 else 0
            inter_mat[i, j] = inter

    return iou_mat, inter_mat


# =====================================================================
# Hungarian Matching (Script 1)
# =====================================================================

def hungarian_matching(iou_mat):
    if iou_mat.size == 0:
        return np.array([]), np.array([]), np.array([])

    cost = 1 - iou_mat
    row_ind, col_ind = linear_sum_assignment(cost)
    matched_iou = iou_mat[row_ind, col_ind]
    return row_ind, col_ind, matched_iou


# =====================================================================
# FULL REGION METRICS (Script 1)
# =====================================================================

def compute_region_metrics(
    gt_regions, pred_zones, iou_mat, inter_mat, region_labels, iou_threshold=0.15
):
    metrics = []
    n_gt, n_pred = iou_mat.shape

    row_ind, col_ind, matched_iou = hungarian_matching(iou_mat)
    match_map = {gi: pj for gi, pj in zip(row_ind, col_ind)}

    pred_areas = [p.area for p in pred_zones]
    pred_centroids = [p.centroid for p in pred_zones]

    matched_pred_set = set()
    false_negatives = 0

    for gi, gt_poly in enumerate(gt_regions):
        label = region_labels[gi]
        gt_area = gt_poly.area
        gt_centroid = gt_poly.centroid

        # Coverage
        total_overlap = inter_mat[gi, :].sum()
        coverage = total_overlap / gt_area if gt_area > 0 else 0

        # Fragmentation
        frac = inter_mat[gi] / gt_area if gt_area > 0 else np.zeros(n_pred)
        fragmentation = int(np.sum(frac > 0.05))

        # Assigned predicted zone
        assigned_pred = match_map.get(gi)
        assigned_iou = iou_mat[gi, assigned_pred] if assigned_pred is not None else 0

        # IoU thresholding
        if assigned_iou < iou_threshold:
            assigned_pred = None
            assigned_iou = 0

        # Matched prediction zone
        if assigned_pred is not None:
            pred_poly = pred_zones[assigned_pred]
            pred_area = pred_areas[assigned_pred]
            pred_centroid = pred_centroids[assigned_pred]
            overlap = inter_mat[gi, assigned_pred]

            purity = overlap / pred_area if pred_area > 0 else 0
            area_err = abs(pred_area - gt_area) / gt_area * 100 if gt_area > 0 else 100
            centroid_dist = gt_centroid.distance(pred_centroid)

            matched_pred_set.add(assigned_pred)

        else:
            pred_area = 0
            purity = 0
            area_err = 100
            centroid_dist = np.nan
            false_negatives += 1

        # Similarity Score
        sim_area = max(0, 1 - area_err / 100)
        sim_centroid = max(0, 1 - (centroid_dist / 10 if centroid_dist == centroid_dist else 1))

        similarity_score = (
            0.5 * assigned_iou +
            0.25 * sim_area +
            0.25 * sim_centroid
        )

        metrics.append({
            "region": label,
            "assigned_zone": assigned_pred,
            "iou": assigned_iou,
            "coverage": coverage,
            "fragmentation": fragmentation,
            "purity": purity,
            "gt_area": gt_area,
            "pred_area": pred_area,
            "area_error_percent": area_err,
            "centroid_distance": centroid_dist,
            "similarity_score": similarity_score,
        })

    # Global stats
    matched = [m for m in metrics if m["assigned_zone"] is not None]
    false_positives = n_pred - len(matched_pred_set)

    matched_iou_vals = [m["iou"] for m in matched]
    matched_area_vals = [m["area_error_percent"] for m in matched]
    matched_cent_vals = [m["centroid_distance"] for m in matched]

    stats = {
        "mean_iou_matched": float(np.mean(matched_iou_vals)) if matched_iou_vals else 0.0,
        "mean_iou_overall": float(np.mean([m["iou"] for m in metrics])),
        "region_recall": len(matched) / n_gt if n_gt > 0 else 0,
        "region_precision": len(matched) / (len(matched) + false_positives) if matched else 0,
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "mean_area_error": float(np.mean(matched_area_vals)) if matched_area_vals else 100,
        "mean_centroid_distance": float(np.nanmean(matched_cent_vals)) if matched_cent_vals else np.nan,
        "mean_fragmentation": float(np.mean([m["fragmentation"] for m in metrics])),
        "mean_similarity_score": float(np.mean([m["similarity_score"] for m in metrics])),
    }

    return metrics, stats


# =====================================================================
# FINAL EVALUATION PIPELINE (Script 2 alignment + Script 1 comparison)
# =====================================================================

def evaluate_building(ID, GT_PATH, PRED_PATH, save_dir=None):
    print(f"\n========== Evaluating Building {ID} ==========")

    # ------------------------------------------------------------------
    # 1. Load GT using Script 2’s loader
    # ------------------------------------------------------------------
    gt_fp, gt_polys, gt_types = load_gt_rooms_and_footprint(GT_PATH)

    gt_apts = [p for p, t in zip(gt_polys, gt_types) if t == "APARTMENT"]
    gt_stairs = [p for p, t in zip(gt_polys, gt_types) if t == "STAIRS"]

    gt_regions = gt_apts + gt_stairs
    region_labels = (
        [f"A{i}" for i in range(len(gt_apts))] +
        [f"S{i}" for i in range(len(gt_stairs))]
    )

    # ------------------------------------------------------------------
    # 2. Load predictions using Script 2
    # ------------------------------------------------------------------
    pred_fp = compute_predicted_footprint(PRED_PATH)
    pred_zones_raw = load_predicted_zone_polygons(PRED_PATH)

    # ------------------------------------------------------------------
    # 3. ALIGN using Script 2 (align_simple)
    # ------------------------------------------------------------------
    pred_fp_aligned, rot, dx, dy, origin = align_simple(pred_fp, gt_fp)

    pred_zones = [
        apply_transform(z, rot, dx, dy, origin)
        for z in pred_zones_raw
    ]

    # ------------------------------------------------------------------
    # 4. Compute IoU + Metrics (from Script 1)
    # ------------------------------------------------------------------
    iou_mat, inter_mat = compute_overlap_matrix(gt_regions, pred_zones)
    metrics, stats = compute_region_metrics(gt_regions, pred_zones, iou_mat, inter_mat, region_labels)

    # ------------------------------------------------------------------
    # 5. Save CSV results
    # ------------------------------------------------------------------
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        pd.DataFrame(metrics).to_csv(os.path.join(save_dir, f"{ID}_region_metrics.csv"), index=False)
        pd.DataFrame([stats]).to_csv(os.path.join(save_dir, f"{ID}_global_stats.csv"), index=False)

    return metrics, stats, iou_mat

# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------
if __name__ == "__main__":

    SWISS_DATASET_ROOT = r"N:\9_SF-Public\Austausch\Thomas Sharon\Master_Thesis_Sharon\Floorplan_Dataset\archive\modified-swiss-dwellings-v2\train"
    PREDICTED_FOLDER   = r"N:\9_SF-Public\Austausch\Thomas Sharon\Master_Thesis_Sharon\Floorplan_Dataset\gml_msd\building_data"

    building_ids = [696]

    for bid in building_ids:
        GT_BASE = os.path.join(SWISS_DATASET_ROOT, "graph_out", f"{bid}.pickle")
        PRED_PATH = os.path.join(PREDICTED_FOLDER, f"building_data_{bid}.pkl")

        evaluate_building(bid, GT_BASE, PRED_PATH)