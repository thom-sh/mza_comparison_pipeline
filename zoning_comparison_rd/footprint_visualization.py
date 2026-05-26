import numpy as np
import matplotlib.pyplot as plt
import pickle

from shapely.geometry import JOIN_STYLE
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.affinity import rotate as shp_rotate, translate as shp_translate

# ======================================================================
# SECTION 1 — Generic Geometry Utilities
# ======================================================================

def largest_poly(g):
    """Return the largest polygon from Polygon or MultiPolygon."""
    if g.geom_type == "Polygon":
        return g
    if g.geom_type == "MultiPolygon":
        return max(g.geoms, key=lambda p: p.area)
    return g.buffer(0)


def pca_angle(poly):
    """Compute PCA orientation angle of a polygon (in degrees)."""
    poly = largest_poly(poly)
    pts = np.array(poly.exterior.coords)
    pts_centered = pts - pts.mean(axis=0)

    cov = np.cov(pts_centered.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    direction = eigvecs[:, np.argmax(eigvals)]
    return np.degrees(np.arctan2(direction[1], direction[0]))


def align_shape(poly, angle, dx, dy, origin):
    """
    Apply the SAME rigid transform used for the footprint:

      1) rotate by 'angle' about 'origin'
      2) translate by (dx, dy)

    This MUST match the order used inside `align_pred`.
    """
    p = shp_rotate(poly, angle, origin=origin, use_radians=False)
    p = shp_translate(p, xoff=dx, yoff=dy)
    return p


def align_pred(pred_poly, gt_poly,
               coarse_step=10.0,
               fine_step=1.0,
               fine_window=10.0):
    """
    Align predicted footprint to GT footprint using OUTLINE-BASED IoU search.

    Logic:
      1) Treat both as arbitrary polygons – we only assume the outer footprint
         outline is comparable, inner polygons may differ completely.
      2) First do a COARSE rotation search over [0, 360) with step `coarse_step`.
         For each angle:
            - rotate predicted around its centroid
            - translate so centroids coincide
            - compute IoU(pred_rotated, gt)
         Keep the best coarse angle.
      3) Then do a FINE search around that angle ± `fine_window` with step `fine_step`.
      4) Return the aligned footprint and the best rotation + translation.

    Returns:
      pred_best      (Polygon)  aligned predicted footprint
      best_rot       (float)    rotation angle in degrees
      best_dx, best_dy          translation (x, y) after rotation
    """
    pred = largest_poly(pred_poly)
    gt   = largest_poly(gt_poly)

    origin = pred.centroid
    gt_cent = gt.centroid

    # ---------------------------
    # COARSE SEARCH
    # ---------------------------
    best_iou = -1.0
    best_rot = 0.0
    best_dx = 0.0
    best_dy = 0.0
    best_poly = None

    for rot in np.arange(0.0, 360.0, coarse_step):
        # rotate about predicted centroid
        p_rot = shp_rotate(pred, rot, origin=origin, use_radians=False)

        # translate so centroids coincide
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

    # ---------------------------
    # FINE SEARCH AROUND BEST COARSE ANGLE
    # ---------------------------
    fine_start = coarse_best_rot - fine_window
    fine_end = coarse_best_rot + fine_window

    for rot in np.arange(fine_start, fine_end + fine_step, fine_step):
        # keep angle within [0, 360) for cleanliness
        rot_norm = rot % 360.0

        # rotate about predicted centroid
        p_rot = shp_rotate(pred, rot_norm, origin=origin, use_radians=False)

        # translate so centroids coincide
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

    # Fallback (should not happen)
    if best_poly is None:
        best_poly = pred

    print(
        f"[align_pred] coarse_best_rot={coarse_best_rot:.3f}°, "
        f"coarse_IoU={coarse_best_iou:.3f}, "
        f"best_rot={best_rot:.3f}°, best_IoU={best_iou:.3f}"
    )

    return best_poly, best_rot, best_dx, best_dy


# ======================================================================
# SECTION 2 — Ground Truth Extraction (Swiss)
# ======================================================================

def load_gt_rooms_and_footprint(gt_path, footprint_smooth=(0.5, -0.4), simplify_tol=0.05):
    """
    Load SIMPLE Swiss-format pickle:
      {"floor_plan": [{"polygon": [(x,y),...], "room_type": 0/1}, ...]}

    Returns:
      fp         : Polygon (footprint built from ALL polygons)
      rooms      : list[Polygon] (all polygons)
      room_types : list[int]     (0/1)
    """
    with open(gt_path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict) or "floor_plan" not in data:
        raise ValueError("Invalid simple GT pickle: expected dict with key 'floor_plan'")

    rooms = []
    room_types = []

    for entry in data["floor_plan"]:
        geom = entry.get("polygon", None)
        rt   = entry.get("room_type", None)
        if geom is None or rt is None:
            continue

        try:
            poly = Polygon(geom).buffer(0)  # buffer(0) to fix minor invalidities
            if poly.is_valid and not poly.is_empty and poly.area > 0:
                rooms.append(poly)
                room_types.append(int(rt))
        except Exception:
            pass

    if not rooms:
        return None, [], []

    merged = unary_union(rooms)

    # mimic your original smoothing pipeline
    fp = merged
    if footprint_smooth is not None:
        b1, b2 = footprint_smooth
        fp = fp.buffer(b1, join_style=JOIN_STYLE.mitre).buffer(b2, join_style=JOIN_STYLE.mitre)

    fp = largest_poly(fp.simplify(simplify_tol, preserve_topology=True))
    return fp, rooms, room_types


def extract_gt_apartments(gt_path,
                                stair_label=1,
                                nonstair_label=0,
                                footprint_smooth=(0.5, -0.4),
                                simplify_tol=0.05,
                                merge_nonstair=False):
    """
    Replacement for extract_gt_apartments() when GT pickle is the SIMPLE format:
      {"floor_plan": [{"polygon": [...], "room_type": 0/1}, ...]}

    Because this format has NO adjacency graph, we cannot infer connected components
    via entrance edges. So we interpret:
      - room_type == stair_label  -> stairs polygons
      - room_type == nonstair_label -> apartment polygons (either kept separate or merged)

    Returns:
      apartment_polygons : list[Polygon or MultiPolygon]
      stairs_polys       : list[Polygon]
      footprint          : Polygon
    """
    fp, rooms, room_types = load_gt_rooms_and_footprint(
        gt_path, footprint_smooth=footprint_smooth, simplify_tol=simplify_tol
    )
    if fp is None:
        return [], [], None

    stairs_polys = [r for r, t in zip(rooms, room_types) if t == stair_label]
    nonstair_polys = [r for r, t in zip(rooms, room_types) if t == nonstair_label]

    # Apartment polygons: either keep each polygon as an "apartment", or merge all non-stairs
    if merge_nonstair:
        if nonstair_polys:
            merged_apt = unary_union(nonstair_polys).buffer(0.2).buffer(-0.2)
            apartment_polygons = [merged_apt]
        else:
            apartment_polygons = []
    else:
        # keep separate (closest analogue to "one polygon per zone")
        apartment_polygons = []
        for p in nonstair_polys:
            pp = p.buffer(0.2).buffer(-0.2)
            apartment_polygons.append(pp)

    return apartment_polygons, stairs_polys, fp



# ======================================================================
# SECTION 3 — Predicted Extraction (Your Model)
# ======================================================================

def load_predicted_zone_polygons(pred_path):
    """Load predicted building and extract zone polygons as unions."""
    with open(pred_path, "rb") as fh:
        bldg = pickle.load(fh)[0]

    storey1 = next(s for s in bldg["polygons"]["storeys"]
                   if s["storey_name"] == "Storey_1")

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
# SECTION 4 — VISUALIZATION FUNCTIONS
# ======================================================================

def visualize_alignment(ID, GT_BASE, PRED_PATH):
    """3-panel visualization: GT, predicted raw, predicted aligned."""

    gt_fp, _, _ = load_gt_rooms_and_footprint(GT_BASE)
    pred_fp     = compute_predicted_footprint(PRED_PATH)
    if pred_fp is None:
        print(f"[WARN] No predicted footprint for building {ID}.")
        return

    pred_aligned, rot, dx, dy = align_pred(pred_fp, gt_fp)

    # Print info
    print("\n===================================================")
    print(f"🔥 Building {ID} – Alignment Results")
    print("===================================================")
    print(f"Swiss GT area:             {gt_fp.area:.2f} m²")
    print(f"Predicted raw area:        {pred_fp.area:.2f} m²")
    print(f"Predicted aligned area:    {pred_aligned.area:.2f} m²")
    print("---------------------------------------------------")
    print(f"Rotation used:             {rot:.2f}°")
    print(f"Translation used:          dx={dx:.2f}, dy={dy:.2f}")
    print("===================================================\n")

    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(24, 8))
    polys = [gt_fp, pred_fp, pred_aligned]
    titles = ["Swiss GT", "Predicted (Raw)", "Predicted (Aligned)"]
    colors = ["#bfbfbf", "#ff7f7f", "#7fbf7f"]

    for ax, poly, title, col in zip(axs, polys, titles, colors):
        x, y = poly.exterior.xy
        ax.fill(x, y, color=col, alpha=0.85)
        ax.set_axis_off()
        ax.set_aspect("equal")
        ax.set_title(f"{title}\nArea = {poly.area:.1f} m²", fontsize=16)

    plt.suptitle(f"Building {ID} — GT vs Predicted Footprint Alignment", fontsize=20)
    plt.tight_layout()
    plt.show()


def visualize_rooms_and_zones(ID, GT_BASE, PRED_PATH):
    """
    3-panel view:
      1) Swiss GT apartments (stairs shown in black)
      2) Predicted aligned zones
      3) Overlay of GT footprint vs predicted aligned footprint
    """

    # --- Swiss GT apartments ---
    apt_polys, stairs_polys, gt_fp = extract_gt_apartments(GT_BASE)

    # --- Predicted zones + footprint ---
    pred_zones = load_predicted_zone_polygons(PRED_PATH)
    pred_fp    = compute_predicted_footprint(PRED_PATH)
    if pred_fp is None:
        print(f"[WARN] No predicted footprint for building {ID}.")
        return

    # Align predicted footprint to GT (outline-based IoU search)
    pred_aligned, rot, dx, dy = align_pred(pred_fp, gt_fp)

    # Align each zone with the SAME rigid transform
    origin = pred_fp.centroid
    aligned_zones = [
        align_shape(z, rot, dx, dy, origin=origin)
        for z in pred_zones
    ]

    # --- Plot ---
    fig, axs = plt.subplots(1, 3, figsize=(27, 9))

    cmap_apts  = plt.get_cmap("tab10")
    cmap_zones = plt.get_cmap("tab10")

    # ------------------------------------------------------------------
    # Panel 1 — Swiss GT Apartments (stairs in black)
    # ------------------------------------------------------------------
    ax = axs[0]
    ax.set_title(f"Swiss GT Apartments ({len(apt_polys)})", fontsize=16)

    for i, apt in enumerate(apt_polys):
        if apt.geom_type == "Polygon":
            xs, ys = apt.exterior.xy
            ax.fill(xs, ys, color=cmap_apts(i % 10), alpha=0.6, ec="black", lw=0.8)
        else:
            for part in apt.geoms:
                xs, ys = part.exterior.xy
                ax.fill(xs, ys, color=cmap_apts(i % 10), alpha=0.6, ec="black", lw=0.8)

    # stairs
    for poly in stairs_polys:
        xs, ys = poly.exterior.xy
        ax.fill(xs, ys, color="black", alpha=0.9)
        ax.plot(xs, ys, color="white", lw=1.2)

    ax.set_aspect("equal")
    ax.set_axis_off()

    # ------------------------------------------------------------------
    # Panel 2 — Predicted Zones (Aligned)
    # ------------------------------------------------------------------
    ax = axs[1]
    ax.set_title("Predicted Zones (Aligned)", fontsize=16)

    for i, poly in enumerate(aligned_zones):
        xs, ys = poly.exterior.xy
        ax.fill(xs, ys, color=cmap_zones(i % 10), alpha=0.75, ec="black", lw=0.7)

    px, py = pred_aligned.exterior.xy
    ax.plot(px, py, "k--", lw=1.2)

    ax.set_aspect("equal")
    ax.set_axis_off()

    # ------------------------------------------------------------------
    # Panel 3 — Overlay: GT vs Predicted (Aligned)
    # ------------------------------------------------------------------
    ax = axs[2]
    ax.set_title("Overlay: GT vs Predicted (Aligned)", fontsize=16)

    gx, gy = gt_fp.exterior.xy
    ax.fill(gx, gy, color="gray", alpha=0.5, ec="none", label="Swiss GT")
    ax.fill(px, py, color="green", alpha=0.5, ec="none", label="Predicted")

    ax.legend()
    ax.set_aspect("equal")
    ax.set_axis_off()

    plt.suptitle(f"Building {ID} — Apartments vs Predicted Zones", fontsize=20)
    plt.tight_layout()
    plt.show()
