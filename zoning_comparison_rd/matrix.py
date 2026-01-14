# ===============================================================
#   APARTMENT + STAIRS vs PREDICTED ZONE EVALUATION (IoU + METRICS)
# ===============================================================
import os
import numpy as np
import matplotlib.pyplot as plt

from shapely.geometry import Polygon
from shapely.ops import unary_union

from footprint_visualization import (
    extract_gt_apartments,
    load_predicted_zone_polygons,
    compute_predicted_footprint,
    align_pred,
    align_shape,
)

from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------
#  Helper: polygon IoU
# ---------------------------------------------------------------
def compute_iou(poly_a, poly_b):
    if poly_a.is_empty or poly_b.is_empty:
        return 0.0
    inter = poly_a.intersection(poly_b).area
    union = poly_a.union(poly_b).area
    if union == 0:
        return 0.0
    return inter / union


# ---------------------------------------------------------------
#  Overlap and IoU matrices
# ---------------------------------------------------------------
def compute_overlap_matrix(gt_regions, pred_zones):
    n_gt = len(gt_regions)
    n_pred = len(pred_zones)

    iou_mat = np.zeros((n_gt, n_pred), dtype=float)
    inter_mat = np.zeros((n_gt, n_pred), dtype=float)

    for i, g in enumerate(gt_regions):
        for j, p in enumerate(pred_zones):
            inter = g.intersection(p).area
            union = g.union(p).area
            iou = 0.0 if union == 0 else inter / union
            iou_mat[i, j] = iou
            inter_mat[i, j] = inter

    return iou_mat, inter_mat


# ---------------------------------------------------------------
#  Hungarian assignment
# ---------------------------------------------------------------
def hungarian_matching(iou_mat):
    if iou_mat.size == 0:
        return np.array([]), np.array([]), np.array([])

    cost = 1 - iou_mat
    row_ind, col_ind = linear_sum_assignment(cost)

    matched_iou = iou_mat[row_ind, col_ind]
    return row_ind, col_ind, matched_iou


# ---------------------------------------------------------------
#  Compute metrics FOR EACH region: A0..An + S0..Sk
#  UPDATED VERSION (Robust, FP/FN, IoU threshold, similarity score)
# ---------------------------------------------------------------
def compute_region_metrics(
    gt_regions,
    pred_zones,
    iou_mat,
    inter_mat,
    region_labels,
    iou_threshold=0.15    # NEW: threshold for valid matches
):
    metrics = []
    n_gt, n_pred = iou_mat.shape

    # Hungarian matching
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
        coverage = total_overlap / gt_area if gt_area > 0 else 0.0

        # Fragmentation
        frac = inter_mat[gi, :] / gt_area if gt_area > 0 else np.zeros(n_pred)
        fragmentation = int(np.sum(frac > 0.05))

        # Initial assigned zone
        assigned_pred = match_map.get(gi, None)
        assigned_iou = float(iou_mat[gi, assigned_pred]) if assigned_pred is not None else 0.0

        # NEW: IoU thresholding
        if assigned_iou < iou_threshold:
            assigned_pred = None
            assigned_iou = 0.0

        # ---------- Matched case ----------
        if assigned_pred is not None:
            pred_poly = pred_zones[assigned_pred]
            pred_area = pred_areas[assigned_pred]
            pred_centroid = pred_centroids[assigned_pred]
            overlap = inter_mat[gi, assigned_pred]

            purity = overlap / pred_area if pred_area > 0 else 0.0
            area_err = abs(pred_area - gt_area) / gt_area * 100 if gt_area > 0 else 100.0
            centroid_dist = gt_centroid.distance(pred_centroid)

            matched_pred_set.add(assigned_pred)

        # ---------- Unmatched (FN) ----------
        else:
            pred_area = 0.0
            purity = 0.0
            area_err = 100.0
            centroid_dist = np.nan
            false_negatives += 1

        # ---------- Combined similarity score (NEW) ----------
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
            "similarity_score": similarity_score,     # NEW
        })

    # ---------- GLOBAL STATS (NEW) ----------
    matched_gt = [m for m in metrics if m["assigned_zone"] is not None]
    unmatched_gt = [m for m in metrics if m["assigned_zone"] is None]

    false_positives = n_pred - len(matched_pred_set)

    global_stats = {
        "mean_iou_matched": float(np.mean([m["iou"] for m in matched_gt])) if matched_gt else 0.0,
        "mean_iou_overall": float(np.mean([m["iou"] for m in metrics])),
        "region_recall": float(len(matched_gt) / n_gt) if n_gt > 0 else 0.0,
        "region_precision": float(len(matched_gt) / (len(matched_gt) + false_positives)) if (len(matched_gt) + false_positives) else 0.0,
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "mean_area_error": float(np.mean([m["area_error_percent"] for m in matched_gt])) if matched_gt else 100.0,
        "mean_centroid_distance": float(np.nanmean([m["centroid_distance"] for m in matched_gt])) if matched_gt else np.nan,
        "mean_fragmentation": float(np.mean([m["fragmentation"] for m in metrics])),
        "mean_similarity_score": float(np.mean([m["similarity_score"] for m in metrics])),
    }

    return metrics, global_stats


# ---------------------------------------------------------------
#  Heatmap visualization
# ---------------------------------------------------------------
def plot_iou_heatmap(ID, iou_mat, labels_gt, labels_pred):
    if iou_mat.size == 0:
        print("⚠️ Empty IoU matrix.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(iou_mat, vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(labels_pred)))
    ax.set_yticks(np.arange(len(labels_gt)))
    ax.set_xticklabels(labels_pred)
    ax.set_yticklabels(labels_gt)

    ax.set_xlabel("Predicted Zones")
    ax.set_ylabel("GT Regions (Apts + Stair Pieces)")
    ax.set_title(f"Building {ID} — IoU Matrix")

    for i in range(iou_mat.shape[0]):
        for j in range(iou_mat.shape[1]):
            v = iou_mat[i, j]
            if v > 0.05:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)

    plt.colorbar(im)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------
# DEBUG VISUALIZATION — SEE ALIGNMENT + OVERLAPS
# ---------------------------------------------------------------
def debug_visualize_alignment(ID, gt_regions, region_labels, pred_zones_raw, pred_zones_aligned, gt_fp, pred_fp, pred_fp_aligned):
    fig, axs = plt.subplots(1, 4, figsize=(30, 7))

    # Panel 1: GT regions
    axs[0].set_title("GT Apartments + Stairs")
    cmap = plt.get_cmap("tab20")
    for i, poly in enumerate(gt_regions):
        xs, ys = poly.exterior.xy
        axs[0].fill(xs, ys, color=cmap(i), alpha=0.6)
        axs[0].text(poly.centroid.x, poly.centroid.y, region_labels[i], fontsize=12)
    axs[0].set_aspect("equal")
    axs[0].set_axis_off()

    # Panel 2: Predicted Zones (raw)
    axs[1].set_title("Predicted Zones (RAW)")
    for i, poly in enumerate(pred_zones_raw):
        xs, ys = poly.exterior.xy
        axs[1].fill(xs, ys, color=cmap(i), alpha=0.6)
        axs[1].text(poly.centroid.x, poly.centroid.y, f"Z{i}", fontsize=12)
    axs[1].set_aspect("equal")
    axs[1].set_axis_off()

    # Panel 3: Predicted Zones (Aligned)
    axs[2].set_title("Predicted Zones (ALIGNED)")
    for i, poly in enumerate(pred_zones_aligned):
        xs, ys = poly.exterior.xy
        axs[2].fill(xs, ys, color=cmap(i), alpha=0.6)
        axs[2].text(poly.centroid.x, poly.centroid.y, f"Z{i}", fontsize=12)
    axs[2].set_aspect("equal")
    axs[2].set_axis_off()

    # Panel 4: Overlay
    axs[3].set_title("Overlay: GT vs Predicted (Aligned)")

    # GT footprint
    gx, gy = gt_fp.exterior.xy
    axs[3].fill(gx, gy, color="gray", alpha=0.5, label="GT")

    # Pred footprint
    px, py = pred_fp_aligned.exterior.xy
    axs[3].fill(px, py, color="green", alpha=0.5, label="Pred")

    axs[3].legend()
    axs[3].set_aspect("equal")
    axs[3].set_axis_off()

    plt.suptitle(f"DEBUG — Building {ID}", fontsize=22)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------
# VISUALIZE OVERLAP PER GT REGION
# ---------------------------------------------------------------
def debug_region_overlap(ID, gt_regions, region_labels, pred_zones, iou_mat):
    cmap = plt.get_cmap("tab10")

    for gi, gt_poly in enumerate(gt_regions):
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.set_title(f"GT Region {region_labels[gi]} — Overlaps with Predicted Zones")

        gx, gy = gt_poly.exterior.xy
        ax.fill(gx, gy, color="black", alpha=0.3)

        for pj, pred_poly in enumerate(pred_zones):
            if iou_mat[gi, pj] > 0:
                xs, ys = pred_poly.exterior.xy
                ax.fill(xs, ys, color=cmap(pj), alpha=0.4)
                ax.text(pred_poly.centroid.x, pred_poly.centroid.y, f"Z{pj}", fontsize=12)

        ax.set_aspect("equal")
        ax.set_axis_off()
        plt.show()


def plot_similarity_scores(ID, metrics):
    """
    Produce a bar chart of per-region similarity scores.
    Useful for seeing which zones match well vs poorly.
    """
    regions = [m["region"] for m in metrics]
    scores = [m["similarity_score"] for m in metrics]

    # Sort by score
    sorted_pairs = sorted(zip(regions, scores), key=lambda x: x[1])
    sorted_regions, sorted_scores = zip(*sorted_pairs)

    plt.figure(figsize=(10, 6))
    plt.bar(sorted_regions, sorted_scores)
    plt.ylim(0, 1)
    plt.ylabel("Similarity Score (0–1)")
    plt.title(f"Building {ID} — Region Similarity Scores")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

import csv
import pandas as pd


def export_region_metrics_to_csv(metrics, out_path):
    """
    Export per-region metrics (A0, A1, S0, ...) to CSV.
    """
    df = pd.DataFrame(metrics)
    df.to_csv(out_path, index=False)
    print(f"[CSV] Region metrics saved to: {out_path}")


def export_global_stats_to_csv(stats, out_path):
    """
    Export global building-level statistics to CSV.
    """
    df = pd.DataFrame([stats])  # one-row dataframe
    df.to_csv(out_path, index=False)
    print(f"[CSV] Global stats saved to: {out_path}")

# ---------------------------------------------------------------
#  MAIN EVALUATION (Option B)
# ---------------------------------------------------------------
def evaluate_building(ID, GT_BASE, PRED_PATH, heatmap=True):
    print(f"\n================ Evaluating Building {ID} (Option B) ================")

    gt_apts, stairs_polys, gt_fp = extract_gt_apartments(GT_BASE)

    gt_regions = list(gt_apts)
    region_labels = [f"A{i}" for i in range(len(gt_apts))]

    for si, poly in enumerate(stairs_polys):
        gt_regions.append(poly)
        region_labels.append(f"S{si}")

    pred_zones_raw = load_predicted_zone_polygons(PRED_PATH)
    pred_fp = compute_predicted_footprint(PRED_PATH)

    pred_fp_aligned, rot, dx, dy = align_pred(pred_fp, gt_fp)
    pred_zones_aligned = [
        align_shape(z, rot, dx, dy, origin=pred_fp.centroid)
        for z in pred_zones_raw
    ]

    labels_pred = [f"Z{j}" for j in range(len(pred_zones_aligned))]

    debug_visualize_alignment(
        ID,
        gt_regions,
        region_labels,
        pred_zones_raw,
        pred_zones_aligned,
        gt_fp,
        pred_fp,
        pred_fp_aligned
    )

    iou_mat, inter_mat = compute_overlap_matrix(gt_regions, pred_zones_aligned)

    metrics, stats = compute_region_metrics(gt_regions, pred_zones_aligned, iou_mat, inter_mat, region_labels)

    print("\nRegion | Pred | IoU | Cov | Frag | Purity | AreaErr% | Dist | Sim")
    print("--------------------------------------------------------------------")
    for m in metrics:
        print(f"{m['region']:6s}  "
              f"{str(m['assigned_zone']).rjust(3)}   "
              f"{m['iou']:.3f}  "
              f"{m['coverage']:.3f}   "
              f"{m['fragmentation']:3d}   "
              f"{m['purity']:.3f}   "
              f"{m['area_error_percent']:.1f}   "
              f"{m['centroid_distance']:.2f}   "
              f"{m['similarity_score']:.3f}")

    if heatmap:
        plot_iou_heatmap(ID, iou_mat, region_labels, labels_pred)

    debug_region_overlap(ID, gt_regions, region_labels, pred_zones_aligned, iou_mat)

    print("\nGLOBAL STATS:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

        # ===== SIMILARITY PLOT =====
    plot_similarity_scores(ID, metrics)

    # ===== CSV EXPORT =====
    output_path = r"C:\WF\Thomas Sharon\Floorplan_Dataset\gml_rd\comparison_output"
    csv_region_path = os.path.join(output_path, f"building_{ID}_region_metrics.csv")
    csv_global_path = os.path.join(output_path, f"building_{ID}_global_stats.csv")
    export_region_metrics_to_csv(metrics, csv_region_path)
    export_global_stats_to_csv(stats, csv_global_path)

    return metrics, stats, iou_mat



# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------
if __name__ == "__main__":

    SWISS_DATASET_ROOT = r"C:\WF\Thomas Sharon\Floorplan_Dataset\rd_pickle"
    PREDICTED_FOLDER   = r"C:\WF\Thomas Sharon\Floorplan_Dataset\gml_rd\building_data"

    building_ids = [2]

    for bid in building_ids:
        GT_BASE = os.path.join(SWISS_DATASET_ROOT, f"{bid}.pickle")
        PRED_PATH = os.path.join(PREDICTED_FOLDER, f"building_data_{bid}.pkl")

        evaluate_building(bid, GT_BASE, PRED_PATH, heatmap=True)





