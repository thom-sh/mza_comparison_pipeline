# ===============================================================
#   APARTMENT + STAIRS vs PREDICTED ZONE EVALUATION (IoU + METRICS)
# ===============================================================
import os
import numpy as np
import matplotlib.pyplot as plt

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

    metrics, stats = compute_region_metrics(
        gt_regions,
        pred_zones_aligned,
        iou_mat,
        inter_mat,
        region_labels
    )

    # -----------------------------------------------------------
    # Core-apartment topology check
    # -----------------------------------------------------------
    topology_stats = check_core_apartment_topology(
        gt_regions=gt_regions,
        region_labels=region_labels,
        pred_zones_aligned=pred_zones_aligned,
        metrics=metrics,
        tol=0.05,
        min_shared_length=0.30
    )

    # Add topology results to global stats
    stats["topology_correct"] = int(topology_stats["topology_correct"])
    stats["n_gt_core_apartment_edges"] = topology_stats["n_gt_core_apartment_edges"]
    stats["n_pred_core_apartment_edges"] = topology_stats["n_pred_core_apartment_edges"]
    stats["n_missing_core_apartment_edges"] = topology_stats["n_missing_core_apartment_edges"]
    stats["n_extra_core_apartment_edges"] = topology_stats["n_extra_core_apartment_edges"]

    stats["gt_core_apartment_edges"] = "; ".join(
        [f"{a}-{b}" for a, b in topology_stats["gt_core_apartment_edges"]]
    )
    stats["pred_core_apartment_edges"] = "; ".join(
        [f"{a}-{b}" for a, b in topology_stats["pred_core_apartment_edges"]]
    )
    stats["missing_core_apartment_edges"] = "; ".join(
        [f"{a}-{b}" for a, b in topology_stats["missing_core_apartment_edges"]]
    )
    stats["extra_core_apartment_edges"] = "; ".join(
        [f"{a}-{b}" for a, b in topology_stats["extra_core_apartment_edges"]]
    )

    # Existing building-level columns
    stats["n_gt_regions"] = len(gt_regions)
    stats["n_pred_regions"] = len(pred_zones_aligned)


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
    # plot_similarity_scores(ID, metrics)

    return metrics, stats, iou_mat

def append_or_create_csv(df_new, csv_path):
    """
    Append to CSV if it exists, otherwise create it.
    """
    if os.path.exists(csv_path):
        df_new.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df_new.to_csv(csv_path, mode="w", header=True, index=False)

def deduplicate_and_sort(region_csv, global_csv):
    """
    Ensure one entry per building (global)
    and one entry per (building, region) (region-level).
    """

    # ---- Global stats: one row per building ----
    df_g = pd.read_csv(global_csv)
    df_g = df_g.drop_duplicates(
        subset=["building_id"],
        keep="last"
    )
    df_g = df_g.sort_values(by="building_id")
    df_g.to_csv(global_csv, index=False)

    # ---- Region metrics: one row per building + region ----
    df_r = pd.read_csv(region_csv)
    df_r = df_r.drop_duplicates(
        subset=["building_id", "region"],
        keep="last"
    )
    df_r = df_r.sort_values(by=["building_id", "region"])
    df_r.to_csv(region_csv, index=False)

def region_metrics_to_rows(metrics, building_id):
    rows = []
    for m in metrics:
        row = dict(m)
        row["building_id"] = building_id
        rows.append(row)
    return rows

def global_stats_to_row(stats, building_id):
    row = dict(stats)
    row["building_id"] = building_id
    return row


### for topology comparison
# ---------------------------------------------------------------
#  Core-apartment topology check
# ---------------------------------------------------------------

def is_apartment_label(label):
    """
    Apartment labels in your comparison are A0, A1, A2, ...
    """
    return str(label).startswith("A")


def is_core_label(label):
    """
    Core / stair labels in your comparison are S0, S1, ...
    """
    return str(label).startswith("S")


def shared_boundary_length(poly_a, poly_b, tol=0.05):
    """
    Estimate shared or near-shared boundary length between two polygons.

    Exact touches can fail because of tiny gaps/overlaps, so a small tolerance
    is used around the boundaries.
    """
    if poly_a is None or poly_b is None:
        return 0.0

    if poly_a.is_empty or poly_b.is_empty:
        return 0.0

    # Exact shared boundary
    exact = poly_a.boundary.intersection(poly_b.boundary).length
    if exact > 0:
        return exact

    # Tolerance-based near-boundary overlap
    near_ab = poly_a.boundary.buffer(tol).intersection(poly_b.boundary).length
    near_ba = poly_b.boundary.buffer(tol).intersection(poly_a.boundary).length

    return max(near_ab, near_ba)


def build_core_apartment_edges(labels, polygons, tol=0.05, min_shared_length=0.30):
    """
    Build adjacency edges only between core and apartment regions.

    Apartment-apartment and core-core contacts are ignored.

    Returns
    -------
    edges : set of tuples
        Example: {("A0", "S0"), ("A1", "S0")}
    edge_details : list of dict
        Useful for debugging / reporting shared boundary lengths.
    """
    edges = set()
    edge_details = []

    for i in range(len(polygons)):
        for j in range(i + 1, len(polygons)):
            li = labels[i]
            lj = labels[j]

            # Only core-apartment pairs are valid
            valid_pair = (
                (is_apartment_label(li) and is_core_label(lj)) or
                (is_core_label(li) and is_apartment_label(lj))
            )

            if not valid_pair:
                continue

            shared_len = shared_boundary_length(polygons[i], polygons[j], tol=tol)

            if shared_len >= min_shared_length:
                edge = tuple(sorted([li, lj]))
                edges.add(edge)

                edge_details.append({
                    "edge": edge,
                    "shared_boundary_length": shared_len
                })

    return edges, edge_details


def check_core_apartment_topology(
    gt_regions,
    region_labels,
    pred_zones_aligned,
    metrics,
    tol=0.05,
    min_shared_length=0.30
):
    """
    Compare core-apartment adjacency topology between GT and prediction.

    Steps:
      1. Build core-apartment adjacency edges for GT regions.
      2. Relabel matched predicted zones using GT labels.
      3. Build core-apartment adjacency edges for matched predicted zones.
      4. Compare missing and additional topology edges.

    Parameters
    ----------
    gt_regions : list[Polygon]
        GT apartment and core polygons.
    region_labels : list[str]
        GT labels, e.g. ["A0", "A1", "S0"].
    pred_zones_aligned : list[Polygon]
        Predicted zones after footprint alignment.
    metrics : list[dict]
        Output from compute_region_metrics().
    tol : float
        Boundary tolerance in metres.
    min_shared_length : float
        Minimum shared boundary length to count as adjacency.

    Returns
    -------
    topology_stats : dict
    """

    # ---------- 1. GT topology ----------
    gt_edges, gt_edge_details = build_core_apartment_edges(
        labels=region_labels,
        polygons=gt_regions,
        tol=tol,
        min_shared_length=min_shared_length
    )

    # ---------- 2. Relabel matched predicted zones ----------
    pred_label_to_poly = {}

    for m in metrics:
        gt_label = m["region"]
        assigned_zone = m["assigned_zone"]

        if assigned_zone is None:
            continue

        try:
            assigned_zone = int(assigned_zone)
        except Exception:
            continue

        if assigned_zone < 0 or assigned_zone >= len(pred_zones_aligned):
            continue

        pred_label_to_poly[gt_label] = pred_zones_aligned[assigned_zone]

    matched_pred_labels = []
    matched_pred_polys = []

    for label in region_labels:
        if label in pred_label_to_poly:
            matched_pred_labels.append(label)
            matched_pred_polys.append(pred_label_to_poly[label])

    # ---------- 3. Predicted topology after relabelling ----------
    pred_edges, pred_edge_details = build_core_apartment_edges(
        labels=matched_pred_labels,
        polygons=matched_pred_polys,
        tol=tol,
        min_shared_length=min_shared_length
    )

    # ---------- 4. Compare topology ----------
    missing_edges = gt_edges - pred_edges
    extra_edges = pred_edges - gt_edges

    topology_correct = (len(missing_edges) == 0 and len(extra_edges) == 0)

    topology_stats = {
        "topology_correct": topology_correct,
        "n_gt_core_apartment_edges": len(gt_edges),
        "n_pred_core_apartment_edges": len(pred_edges),
        "n_missing_core_apartment_edges": len(missing_edges),
        "n_extra_core_apartment_edges": len(extra_edges),
        "gt_core_apartment_edges": sorted(list(gt_edges)),
        "pred_core_apartment_edges": sorted(list(pred_edges)),
        "missing_core_apartment_edges": sorted(list(missing_edges)),
        "extra_core_apartment_edges": sorted(list(extra_edges)),
        "gt_edge_details": gt_edge_details,
        "pred_edge_details": pred_edge_details,
    }

    return topology_stats


if __name__ == "__main__":

    # SWISS_DATASET_ROOT = r"C:\WF\Thomas Sharon\Floorplan_Dataset\msd_pickle"
    # PREDICTED_FOLDER   = r"C:\WF\Thomas Sharon\Floorplan_Dataset\gml_msd\building_data"

    # OUTPUT_DIR = r"C:\WF\Thomas Sharon\Floorplan_Dataset\comparison_logic"

    SWISS_DATASET_ROOT = r"C:\WF\Thomas Sharon\Floorplan_Dataset\rd_pickle"
    PREDICTED_FOLDER   = r"C:\WF\Thomas Sharon\Floorplan_Dataset\gml_rd\building_data"

    OUTPUT_DIR = r"C:\WF\Thomas Sharon\Floorplan_Dataset\gml_rd\comparison_output"


    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # REGION_CSV = os.path.join(OUTPUT_DIR, "all_region_metrics_selected_200.csv")
    # GLOBAL_CSV = os.path.join(OUTPUT_DIR, "all_global_stats_selected_200.csv")

    REGION_CSV = os.path.join(OUTPUT_DIR, "all_region_metrics_rd_final.csv")
    GLOBAL_CSV = os.path.join(OUTPUT_DIR, "all_global_stats_rd_final.csv")


    all_region_rows = []
    all_global_rows = []

    # building_ids = [1827]
    # building_ids = [23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44]
    # building_ids = [
    # 68, 75, 179, 329, 467, 696, 807, 1291, 1321, 1361, 1575, 1588, 1595,
    # 1601, 1663, 1686, 1712, 1728, 1817, 1925, 1934, 1953, 1996, 2018, 2030,
    # 2049, 2136, 2244, 2389, 2401, 2410, 2540, 2568, 2896, 3002, 3057, 3283,
    # 3594, 3669, 4026, 4234, 4239, 4258, 4321, 4832, 5069, 5086, 5102, 5103,
    # 5319, 5322, 5863, 6362, 6370, 6599, 6676, 7299, 7343, 7737, 7760, 7792,
    # 7824, 7869, 7899, 7914, 7916, 8039, 8202, 8241, 8260, 8264, 8308, 8309,
    # 8314, 8412, 8413, 8443, 8447, 8460, 8549, 8851, 8860, 8863, 8866, 8877,
    # 8881, 9205, 9678, 9729, 10277, 10388, 10405, 10655, 10959, 11226, 11434,
    # 11818, 11906, 11967, 13488, 13544, 13858, 14016, 14063, 14123, 14128,
    # 14131, 14747, 14818, 14819, 14881, 14897, 15364, 22206, 22211, 22844,
    # 22886, 23213, 23246, 23562, 23865, 23871, 24153, 24173, 24288, 24472,
    # 24476, 24501, 24542, 24966, 25184, 25307, 25320, 25947, 26170, 26175,
    # 26471, 26593, 26653, 26838, 26858, 26939, 28611, 28949, 29270, 29399,
    # 29686, 29729, 30405, 30453, 39307, 42392, 43687, 44248, 44871, 45570,
    # 45576, 45631, 45644, 45658, 45724, 46073, 46492, 47229, 47492, 48408,
    # 48966, 49004, 49035, 49051, 49320, 49951, 50528, 50530, 50537, 51001,
    # 51680, 51693, 176, 322, 405, 553, 712, 721, 803, 993, 1827, 1943, 1976,
    # 2801, 3039, 3616, 5325, 7801, 8364, 8424, 9222, 9682, 10288, 10376]

    # building_ids = [1827, 1925, 2030, 6599, 7801, 8364, 11818, 14819, 23871, 26175, 28611, 30405, 30453, 45570, 45631, 45658, 46492]
    # building_ids = [30405, 28611, 26175, 23871, 14819, 11818, 8364, 7801, 6599, 2030, 1925, 1827]
    building_ids = [39]


    for bid in building_ids:
        GT_BASE = os.path.join(SWISS_DATASET_ROOT, f"{bid}.pickle")
        PRED_PATH = os.path.join(PREDICTED_FOLDER, f"building_data_{bid}.pkl")

        metrics, stats, iou_mat = evaluate_building(
            bid, GT_BASE, PRED_PATH, heatmap=True
        )

        all_region_rows.extend(region_metrics_to_rows(metrics, bid))
        all_global_rows.append(global_stats_to_row(stats, bid))

    # ---- Convert to DataFrames ----
    df_regions = pd.DataFrame(all_region_rows)
    df_globals = pd.DataFrame(all_global_rows)

    # ---- Append or create ----
    append_or_create_csv(df_regions, REGION_CSV)
    append_or_create_csv(df_globals, GLOBAL_CSV)

    # ---- Deduplicate + sort (CRITICAL STEP) ----
    deduplicate_and_sort(REGION_CSV, GLOBAL_CSV)

    print("[DONE] Aggregated CSVs updated safely:")
    print("  - all_region_metrics.csv")
    print("  - all_global_stats.csv")


