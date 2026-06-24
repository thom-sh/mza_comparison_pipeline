# ===============================================================
#   MSD APARTMENT + CORE vs PREDICTED ZONE EVALUATION
#   Uses updated msd_processing logic with simultaneous apartment growth
# ===============================================================
import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.affinity import rotate as shp_rotate, translate as shp_translate
from scipy.optimize import linear_sum_assignment

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from MZA_Thesis.archive.zoning_comparison_msd.utils_apt import load_pickle
from MZA_Thesis.archive.zoning_comparison_msd.msd_processing_updated import (
    get_type_sets,
    remove_auxiliary_rooms,
    detect_apartments_and_core_nodes,
    extract_apartment_polygons,
    extract_core_union_from_nodes,
    extract_building_footprint_from_apts_and_core,
    simultaneous_apartment_growth,
    polygon_parts,
)


# ======================================================================
# SECTION 1 — Generic Geometry Utilities
# ======================================================================

def largest_poly(g):
    """Return the largest polygon from Polygon or MultiPolygon."""
    if g is None or g.is_empty:
        return g
    if g.geom_type == "Polygon":
        return g
    if g.geom_type == "MultiPolygon":
        return max(g.geoms, key=lambda p: p.area)
    return g.buffer(0)


def align_shape(poly, angle, dx, dy, origin):
    """
    Apply the same rigid transform used for the footprint:
      1) rotate by angle about origin
      2) translate by (dx, dy)
    """
    p = shp_rotate(poly, angle, origin=origin, use_radians=False)
    p = shp_translate(p, xoff=dx, yoff=dy)
    return p


def align_pred(pred_poly, gt_poly, coarse_step=10.0, fine_step=1.0, fine_window=10.0):
    """Align predicted footprint to GT footprint using outline-based IoU search."""
    pred = largest_poly(pred_poly)
    gt = largest_poly(gt_poly)

    origin = pred.centroid
    gt_cent = gt.centroid

    best_iou = -1.0
    best_rot = 0.0
    best_dx = 0.0
    best_dy = 0.0
    best_poly = None

    for rot in np.arange(0.0, 360.0, coarse_step):
        p_rot = shp_rotate(pred, rot, origin=origin, use_radians=False)
        dx = gt_cent.x - p_rot.centroid.x
        dy = gt_cent.y - p_rot.centroid.y
        p_aligned = shp_translate(p_rot, xoff=dx, yoff=dy)

        inter = p_aligned.intersection(gt).area
        union = p_aligned.union(gt).area
        iou = 0.0 if union == 0 else inter / union

        if iou > best_iou:
            best_iou = iou
            best_rot = rot
            best_dx = dx
            best_dy = dy
            best_poly = p_aligned

    coarse_best_rot = best_rot
    coarse_best_iou = best_iou

    fine_start = coarse_best_rot - fine_window
    fine_end = coarse_best_rot + fine_window

    for rot in np.arange(fine_start, fine_end + fine_step, fine_step):
        rot_norm = rot % 360.0
        p_rot = shp_rotate(pred, rot_norm, origin=origin, use_radians=False)
        dx = gt_cent.x - p_rot.centroid.x
        dy = gt_cent.y - p_rot.centroid.y
        p_aligned = shp_translate(p_rot, xoff=dx, yoff=dy)

        inter = p_aligned.intersection(gt).area
        union = p_aligned.union(gt).area
        iou = 0.0 if union == 0 else inter / union

        if iou > best_iou:
            best_iou = iou
            best_rot = rot_norm
            best_dx = dx
            best_dy = dy
            best_poly = p_aligned

    if best_poly is None:
        best_poly = pred

    print(
        f"[align_pred] coarse_best_rot={coarse_best_rot:.3f}°, "
        f"coarse_IoU={coarse_best_iou:.3f}, "
        f"best_rot={best_rot:.3f}°, best_IoU={best_iou:.3f}"
    )

    return best_poly, best_rot, best_dx, best_dy


# ======================================================================
# SECTION 2 — GT Extraction using UPDATED MSD logic
# ======================================================================

def extract_gt_apartments(
    gt_path,
    apartment_buffer=0.08,
    core_buffer=0.15,
    outer_buffer=0.45,
    inner_buffer=-0.35,
    footprint_simplify_tol=0.03,
    use_gap_free_growth=True,
    growth_step=0.03,
    growth_max_iter=400,
    growth_min_residual_area=1e-4,
):
    """
    Extract apartment polygons and core polygons from MSD Swiss GT using the
    updated graph-based logic.
    """
    G = load_pickle(gt_path)

    _, private_types, auxiliary_types = get_type_sets()

    remove_auxiliary_rooms(G, auxiliary_types)
    apartments, core_nodes = detect_apartments_and_core_nodes(G, private_types)

    apartment_polygons = extract_apartment_polygons(
        G,
        apartments,
        auxiliary_types,
        buffer_amt=apartment_buffer,
    )

    core_union = extract_core_union_from_nodes(
        G,
        core_nodes,
        buffer_amt=core_buffer,
    )

    gt_footprint = extract_building_footprint_from_apts_and_core(
        apartment_polygons=apartment_polygons,
        core_union=core_union,
        outer_buffer=outer_buffer,
        inner_buffer=inner_buffer,
        simplify_tol=footprint_simplify_tol,
    )

    if use_gap_free_growth and apartment_polygons:
        apartment_polygons, residual_gap = simultaneous_apartment_growth(
            apartment_polygons=apartment_polygons,
            core_union=core_union,
            footprint=gt_footprint,
            step=growth_step,
            max_iter=growth_max_iter,
            min_residual_area=growth_min_residual_area,
            simplify_tol=0.0,
        )
        apartment_polygons = [g for g in apartment_polygons if g is not None and not g.is_empty]
    else:
        residual_gap = None

    core_polys = core_union_to_polygons(core_union)
    return apartment_polygons, core_polys, gt_footprint, residual_gap


def core_union_to_polygons(core_union):
    """Normalize core union to list[Polygon]."""
    if core_union is None or core_union.is_empty:
        return []
    if core_union.geom_type == "Polygon":
        return [core_union]
    if core_union.geom_type == "MultiPolygon":
        return list(core_union.geoms)
    return []


# ======================================================================
# SECTION 3 — Predicted Extraction
# ======================================================================

def load_predicted_zone_polygons(pred_path):
    """Load predicted building and extract zone polygons as unions."""
    with open(pred_path, "rb") as fh:
        bldg = pickle.load(fh)[0]

    storey1 = next(
        s for s in bldg["polygons"]["storeys"]
        if s["storey_name"] == "Storey_1"
    )

    zone_polys = []
    for zone in storey1["zones"]:
        polys = [
            f[0] for f in zone["floors"]
            if isinstance(f, tuple) and isinstance(f[0], Polygon)
        ]
        if polys:
            zone_polys.append(unary_union(polys))

    return zone_polys


def compute_predicted_footprint(pred_path):
    """Compute building footprint from predicted floors."""
    zones = load_predicted_zone_polygons(pred_path)
    if len(zones) == 0:
        return None
    return largest_poly(unary_union(zones))


# ======================================================================
# SECTION 4 — Metrics
# ======================================================================

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


def hungarian_matching(iou_mat):
    if iou_mat.size == 0:
        return np.array([]), np.array([]), np.array([])

    cost = 1 - iou_mat
    row_ind, col_ind = linear_sum_assignment(cost)
    matched_iou = iou_mat[row_ind, col_ind]
    return row_ind, col_ind, matched_iou


def compute_region_metrics(
    gt_regions,
    pred_zones,
    iou_mat,
    inter_mat,
    region_labels,
    iou_threshold=0.15,
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

        total_overlap = inter_mat[gi, :].sum()
        coverage = total_overlap / gt_area if gt_area > 0 else 0.0

        frac = inter_mat[gi, :] / gt_area if gt_area > 0 else np.zeros(n_pred)
        fragmentation = int(np.sum(frac > 0.05))

        assigned_pred = match_map.get(gi, None)
        assigned_iou = float(iou_mat[gi, assigned_pred]) if assigned_pred is not None else 0.0

        if assigned_iou < iou_threshold:
            assigned_pred = None
            assigned_iou = 0.0

        if assigned_pred is not None:
            pred_area = pred_areas[assigned_pred]
            pred_centroid = pred_centroids[assigned_pred]
            overlap = inter_mat[gi, assigned_pred]

            purity = overlap / pred_area if pred_area > 0 else 0.0
            area_err = abs(pred_area - gt_area) / gt_area * 100 if gt_area > 0 else 100.0
            centroid_dist = gt_centroid.distance(pred_centroid)
            matched_pred_set.add(assigned_pred)
        else:
            pred_area = 0.0
            purity = 0.0
            area_err = 100.0
            centroid_dist = np.nan
            false_negatives += 1

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

    matched_gt = [m for m in metrics if m["assigned_zone"] is not None]
    false_positives = n_pred - len(matched_pred_set)

    global_stats = {
        "mean_iou_matched": float(np.mean([m["iou"] for m in matched_gt])) if matched_gt else 0.0,
        "mean_iou_overall": float(np.mean([m["iou"] for m in metrics])) if metrics else 0.0,
        "region_recall": float(len(matched_gt) / n_gt) if n_gt > 0 else 0.0,
        "region_precision": float(len(matched_gt) / (len(matched_gt) + false_positives)) if (len(matched_gt) + false_positives) else 0.0,
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "mean_area_error": float(np.mean([m["area_error_percent"] for m in matched_gt])) if matched_gt else 100.0,
        "mean_centroid_distance": float(np.nanmean([m["centroid_distance"] for m in matched_gt])) if matched_gt else np.nan,
        "mean_fragmentation": float(np.mean([m["fragmentation"] for m in metrics])) if metrics else 0.0,
        "mean_similarity_score": float(np.mean([m["similarity_score"] for m in metrics])) if metrics else 0.0,
    }

    return metrics, global_stats


# ======================================================================
# SECTION 5 — Visualization helpers
# ======================================================================

def draw_geometry(ax, geom, facecolor, alpha=0.6, edgecolor="black", linewidth=0.8, label=None):
    parts = polygon_parts(geom)
    first = True
    for part in parts:
        xs, ys = part.exterior.xy
        ax.fill(
            xs,
            ys,
            color=facecolor,
            alpha=alpha,
            ec=edgecolor,
            lw=linewidth,
            label=label if first else None,
        )
        first = False


def label_geometry(ax, geom, text, fontsize=12):
    parts = polygon_parts(geom)
    if not parts:
        return
    part = max(parts, key=lambda g: g.area)
    rp = part.representative_point()
    ax.text(rp.x, rp.y, text, fontsize=fontsize)


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
    ax.set_ylabel("GT Regions (Apts + Core Pieces)")
    ax.set_title(f"Building {ID} — IoU Matrix")

    for i in range(iou_mat.shape[0]):
        for j in range(iou_mat.shape[1]):
            v = iou_mat[i, j]
            if v > 0.05:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)

    plt.colorbar(im)
    plt.tight_layout()
    plt.show()


def debug_visualize_alignment(ID, gt_regions, region_labels, pred_zones_raw, pred_zones_aligned, gt_fp, pred_fp_aligned):
    fig, axs = plt.subplots(1, 3, figsize=(24, 7))
    cmap = plt.get_cmap("tab20")

    axs[0].set_title("GT Apartments + Core")
    for i, poly in enumerate(gt_regions):
        draw_geometry(axs[0], poly, facecolor=cmap(i), alpha=0.6)
        label_geometry(axs[0], poly, region_labels[i], fontsize=12)
    axs[0].set_aspect("equal")
    axs[0].set_axis_off()

    axs[1].set_title("Predicted Zones (Aligned)")
    for i, poly in enumerate(pred_zones_aligned):
        draw_geometry(axs[1], poly, facecolor=cmap(i), alpha=0.6)
        label_geometry(axs[1], poly, f"Z{i}", fontsize=12)
    axs[1].set_aspect("equal")
    axs[1].set_axis_off()

    axs[2].set_title("Overlay: GT vs Predicted (Aligned)")
    draw_geometry(axs[2], gt_fp, facecolor="gray", alpha=0.5, edgecolor="none", label="GT")
    draw_geometry(axs[2], pred_fp_aligned, facecolor="green", alpha=0.5, edgecolor="none", label="Pred")
    axs[2].legend()
    axs[2].set_aspect("equal")
    axs[2].set_axis_off()

    plt.suptitle(f"DEBUG — Building {ID}", fontsize=20)
    plt.tight_layout()
    plt.show()


def plot_similarity_scores(ID, metrics):
    regions = [m["region"] for m in metrics]
    scores = [m["similarity_score"] for m in metrics]

    sorted_pairs = sorted(zip(regions, scores), key=lambda x: x[1])
    if not sorted_pairs:
        print(f"[WARN] No similarity scores available for building {ID}.")
        return

    sorted_regions, sorted_scores = zip(*sorted_pairs)

    plt.figure(figsize=(10, 6))
    plt.bar(sorted_regions, sorted_scores)
    plt.ylim(0, 1)
    plt.ylabel("Similarity Score (0–1)")
    plt.title(f"Building {ID} — Region Similarity Scores")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# ======================================================================
# SECTION 6 — Main evaluation
# ======================================================================

def evaluate_building(ID, GT_BASE, PRED_PATH, heatmap=True, debug_plots=True):
    print(f"\n================ Evaluating MSD Building {ID} ================")

    gt_apts, core_polys, gt_fp, residual_gap = extract_gt_apartments(GT_BASE)

    gt_regions = list(gt_apts)
    region_labels = [f"A{i}" for i in range(len(gt_apts))]

    for si, poly in enumerate(core_polys):
        gt_regions.append(poly)
        region_labels.append(f"C{si}")

    pred_zones_raw = load_predicted_zone_polygons(PRED_PATH)
    pred_fp = compute_predicted_footprint(PRED_PATH)
    if pred_fp is None:
        raise ValueError(f"No predicted footprint found for building {ID}.")

    pred_fp_aligned, rot, dx, dy = align_pred(pred_fp, gt_fp)
    pred_zones_aligned = [
        align_shape(z, rot, dx, dy, origin=pred_fp.centroid)
        for z in pred_zones_raw
    ]

    labels_pred = [f"Z{j}" for j in range(len(pred_zones_aligned))]

    if debug_plots:
        debug_visualize_alignment(
            ID,
            gt_regions,
            region_labels,
            pred_zones_raw,
            pred_zones_aligned,
            gt_fp,
            pred_fp_aligned,
        )

    iou_mat, inter_mat = compute_overlap_matrix(gt_regions, pred_zones_aligned)
    metrics, stats = compute_region_metrics(
        gt_regions,
        pred_zones_aligned,
        iou_mat,
        inter_mat,
        region_labels,
    )

    stats["n_gt_regions"] = len(gt_regions)
    stats["n_pred_regions"] = len(pred_zones_aligned)
    stats["residual_gap_area"] = 0.0 if residual_gap is None else residual_gap.area

    print("\nRegion | Pred | IoU | Cov | Frag | Purity | AreaErr% | Dist | Sim")
    print("--------------------------------------------------------------------")
    for m in metrics:
        dist_txt = f"{m['centroid_distance']:.2f}" if m['centroid_distance'] == m['centroid_distance'] else "nan"
        print(
            f"{m['region']:6s}  "
            f"{str(m['assigned_zone']).rjust(3)}   "
            f"{m['iou']:.3f}  "
            f"{m['coverage']:.3f}   "
            f"{m['fragmentation']:3d}   "
            f"{m['purity']:.3f}   "
            f"{m['area_error_percent']:.1f}   "
            f"{dist_txt:>5s}   "
            f"{m['similarity_score']:.3f}"
        )

    if heatmap:
        plot_iou_heatmap(ID, iou_mat, region_labels, labels_pred)

    if debug_plots:
        plot_similarity_scores(ID, metrics)

    print("\nGLOBAL STATS:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    return metrics, stats, iou_mat


def append_or_create_csv(df_new, csv_path):
    if os.path.exists(csv_path):
        df_new.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df_new.to_csv(csv_path, mode="w", header=True, index=False)


def deduplicate_and_sort(region_csv, global_csv):
    df_g = pd.read_csv(global_csv)
    df_g = df_g.drop_duplicates(subset=["building_id"], keep="last")
    df_g = df_g.sort_values(by="building_id")
    df_g.to_csv(global_csv, index=False)

    df_r = pd.read_csv(region_csv)
    df_r = df_r.drop_duplicates(subset=["building_id", "region"], keep="last")
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


def main():
    SWISS_DATASET_ROOT = r"C:\WF\Thomas Sharon\Floorplan_Dataset\archive\modified-swiss-dwellings-v2\train"
    PREDICTED_FOLDER = r"C:\WF\Thomas Sharon\Floorplan_Dataset\gml_msd\building_data"

    OUTPUT_DIR = r"C:\WF\Thomas Sharon\Floorplan_Dataset\gml_msd\comparison_output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    REGION_CSV = os.path.join(OUTPUT_DIR, "all_region_metrics_updated.csv")
    GLOBAL_CSV = os.path.join(OUTPUT_DIR, "all_global_stats_updated.csv")

    all_region_rows = []
    all_global_rows = []

    # building_ids = [50023, 50528, 50530, 50537, 50543, 50911, 50948, 50953, 51001, 51076, 51657, 51662, 51680, 51693, 51722]
    building_ids = [75]

    for bid in building_ids:
        GT_BASE = os.path.join(SWISS_DATASET_ROOT, "graph_out", f"{bid}.pickle")
        PRED_PATH = os.path.join(PREDICTED_FOLDER, f"building_data_{bid}.pkl")

        metrics, stats, iou_mat = evaluate_building(
            bid,
            GT_BASE,
            PRED_PATH,
            heatmap=True,
            debug_plots=True,
        )

        all_region_rows.extend(region_metrics_to_rows(metrics, bid))
        all_global_rows.append(global_stats_to_row(stats, bid))

    df_regions = pd.DataFrame(all_region_rows)
    df_globals = pd.DataFrame(all_global_rows)

    append_or_create_csv(df_regions, REGION_CSV)
    append_or_create_csv(df_globals, GLOBAL_CSV)
    deduplicate_and_sort(REGION_CSV, GLOBAL_CSV)

    print("[DONE] Aggregated CSVs updated safely:")
    print("  - all_region_metrics_updated.csv")
    print("  - all_global_stats_updated.csv")


if __name__ == "__main__":
    main()
