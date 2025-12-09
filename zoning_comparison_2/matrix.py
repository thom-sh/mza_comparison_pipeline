# =====================================================================
#   ZONE EVALUATION PIPELINE (Swiss GT vs Predicted Zones)
#   Uses: SIMPLE MBR ALIGNMENT (align_simple + apply_transform)
# =====================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

from shapely.geometry import Polygon

# -----------------------------
# Loaders from your dataset
# -----------------------------
from dataset_loaders import (
    load_gt_rooms_and_footprint,
    load_predicted_zone_polygons,
    compute_predicted_footprint,
)

# -----------------------------
# Alignment from simple_alignment.py
# -----------------------------
from simple_alignment import (
    align_simple,
    apply_transform,
)

# =====================================================================
# IoU + Overlap Matrix
# =====================================================================

def compute_iou_matrix(gt_regions, pred_regions):
    """Compute IoU + Intersection Area matrices."""
    n_gt = len(gt_regions)
    n_pr = len(pred_regions)

    iou = np.zeros((n_gt, n_pr))
    inter = np.zeros((n_gt, n_pr))

    for i, g in enumerate(gt_regions):
        for j, p in enumerate(pred_regions):
            A = g.intersection(p).area
            U = g.union(p).area
            inter[i, j] = A
            iou[i, j] = A / U if U > 0 else 0

    return iou, inter


# =====================================================================
# Hungarian Assignment
# =====================================================================

def hungarian_assignment(iou_mat):
    if iou_mat.size == 0:
        return [], [], []
    cost = 1 - iou_mat          # maximize IoU = minimize (1-IoU)
    row_idx, col_idx = linear_sum_assignment(cost)
    matched_scores = iou_mat[row_idx, col_idx]
    return row_idx, col_idx, matched_scores


# =====================================================================
# Region Metrics
# =====================================================================

def compute_metrics(gt_regions, pred_zones, iou_mat, inter_mat, labels, iou_thr=0.15):

    metrics = []
    n_gt, n_pr = iou_mat.shape

    # Hungarian matching
    r_idx, c_idx, matched_iou = hungarian_assignment(iou_mat)
    assignment = {gi: pj for gi, pj in zip(r_idx, c_idx)}

    pred_areas = [p.area for p in pred_zones]
    pred_centroids = [p.centroid for p in pred_zones]

    used_pred = set()
    FN = 0

    for gi, gt in enumerate(gt_regions):
        label = labels[gi]
        gt_area = gt.area
        gt_cent = gt.centroid

        # Coverage
        total_overlap = inter_mat[gi].sum()
        coverage = total_overlap / gt_area if gt_area > 0 else 0

        # Fragmentation
        frac = inter_mat[gi] / gt_area if gt_area > 0 else np.zeros(n_pr)
        fragmentation = int(np.sum(frac > 0.05))

        # Hungarian-assigned predicted zone
        pj = assignment.get(gi)
        iou_assign = iou_mat[gi, pj] if pj is not None else 0

        # IoU thresholding
        if iou_assign < iou_thr:
            pj = None
            iou_assign = 0

        # If matched:
        if pj is not None:
            pred = pred_zones[pj]
            p_area = pred_areas[pj]
            p_cent = pred_centroids[pj]
            A = inter_mat[gi, pj]

            purity = A / p_area if p_area > 0 else 0
            area_err = abs(p_area - gt_area) / gt_area * 100 if gt_area > 0 else 100
            cent_dist = gt_cent.distance(p_cent)

            used_pred.add(pj)

        # If unmatched = FN
        else:
            p_area = 0
            purity = 0
            area_err = 100
            cent_dist = np.nan
            FN += 1

        # Similarity Score
        sim_area = max(0, 1 - area_err / 100)
        sim_cent = max(0, 1 - (cent_dist / 10 if cent_dist == cent_dist else 1))
        sim_score = 0.5 * iou_assign + 0.25 * sim_area + 0.25 * sim_cent

        metrics.append({
            "region": label,
            "assigned_zone": pj,
            "iou": iou_assign,
            "coverage": coverage,
            "fragmentation": fragmentation,
            "purity": purity,
            "gt_area": gt_area,
            "pred_area": p_area,
            "area_error_percent": area_err,
            "centroid_distance": cent_dist,
            "similarity_score": sim_score,
        })

    FP = n_pr - len(used_pred)
    matched = [m for m in metrics if m["assigned_zone"] is not None]

    stats = {
        "mean_iou_matched": float(np.mean([m["iou"] for m in matched])) if matched else 0,
        "mean_iou_overall": float(np.mean([m["iou"] for m in metrics])),
        "region_recall": len(matched) / n_gt if n_gt > 0 else 0,
        "region_precision": len(matched) / (len(matched) + FP) if matched else 0,
        "false_negatives": FN,
        "false_positives": FP,
        "mean_area_error": float(np.mean([m["area_error_percent"] for m in matched])) if matched else 100,
        "mean_centroid_distance": float(np.nanmean([m["centroid_distance"] for m in matched])) if matched else np.nan,
        "mean_fragmentation": float(np.mean([m["fragmentation"] for m in metrics])),
        "mean_similarity_score": float(np.mean([m["similarity_score"] for m in metrics])),
    }

    return metrics, stats


# =====================================================================
# Final Pipeline
# =====================================================================

def evaluate_building(ID, GT_BASE, PRED_PATH, save_dir=None):

    print(f"\n========== Evaluating Building {ID} ==========")

    # -------- Load GT correctly --------
    gt_fp, gt_polys, gt_types = load_gt_rooms_and_footprint(GT_BASE)

    # Split into apartments + stairs
    gt_apts   = [p for p, t in zip(gt_polys, gt_types) if t == "APARTMENT"]
    gt_stairs = [p for p, t in zip(gt_polys, gt_types) if t == "STAIRS"]

    gt_regions = gt_apts + gt_stairs
    labels = (
        [f"A{i}" for i in range(len(gt_apts))] +
        [f"S{i}" for i in range(len(gt_stairs))]
    )

    # -------- Load Predictions --------
    pred_fp = compute_predicted_footprint(PRED_PATH)
    pred_zones = load_predicted_zone_polygons(PRED_PATH)

    # -------- ALIGN using SIMPLE MBR method --------
    pred_fp_aligned, rot, dx, dy, origin = align_simple(pred_fp, gt_fp)

    pred_zones_aligned = [
        apply_transform(z, rot, dx, dy, origin) for z in pred_zones
    ]

    # -------- IoU + Metrics --------
    iou_mat, inter_mat = compute_iou_matrix(gt_regions, pred_zones_aligned)
    metrics, stats = compute_metrics(
        gt_regions, pred_zones_aligned, iou_mat, inter_mat, labels
    )

    # -------- CSV save --------
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        pd.DataFrame(metrics).to_csv(os.path.join(save_dir, f"{ID}_region_metrics.csv"), index=False)
        pd.DataFrame([stats]).to_csv(os.path.join(save_dir, f"{ID}_global_stats.csv"), index=False)

    return metrics, stats, iou_mat

