# ===============================================================
#   SIMPLE EDGE-BASED ALIGNMENT USING MINIMUM BOUNDING RECTANGLE
# ===============================================================

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.affinity import rotate as shp_rotate, translate as shp_translate


# ---------------------------------------------------------------
def largest_poly(g):
    if g is None:
        return None
    if g.geom_type == "Polygon":
        return g
    if g.geom_type == "MultiPolygon":
        return max(g.geoms, key=lambda p: p.area)
    return g.buffer(0)


# ---------------------------------------------------------------
# Compute orientation from minimum bounding rectangle (MBR)
# ---------------------------------------------------------------
def mbr_angle(poly):
    poly = largest_poly(poly)
    rect = poly.minimum_rotated_rectangle
    coords = np.array(rect.exterior.coords)

    # longest edge direction
    p0, p1 = coords[0], coords[1]
    dx, dy = p1 - p0
    ang = np.degrees(np.arctan2(dy, dx))

    return ang


# ---------------------------------------------------------------
# ALIGNMENT: rotate by MBR angle difference + centroid translation
# ---------------------------------------------------------------
def align_simple(pred_fp, gt_fp):
    pred_fp = largest_poly(pred_fp)
    gt_fp   = largest_poly(gt_fp)

    # --- 1) orientation from bounding rectangle ---
    ang_pred = mbr_angle(pred_fp)
    ang_gt   = mbr_angle(gt_fp)

    rot_deg = ang_gt - ang_pred

    # --- 2) rotate around predicted centroid ---
    origin = pred_fp.centroid
    pred_rot = shp_rotate(pred_fp, rot_deg, origin=origin, use_radians=False)

    # --- 3) centroid translation ---
    c_pred = pred_rot.centroid
    c_gt   = gt_fp.centroid
    dx = c_gt.x - c_pred.x
    dy = c_gt.y - c_pred.y

    pred_align = shp_translate(pred_rot, xoff=dx, yoff=dy)

    return pred_align, rot_deg, dx, dy, origin


# ---------------------------------------------------------------
def apply_transform(poly, rot_deg, dx, dy, origin):
    p = shp_rotate(poly, rot_deg, origin=origin, use_radians=False)
    p = shp_translate(p, xoff=dx, yoff=dy)
    return p


# ---------------------------------------------------------------
def plot_alignment_with_overlay(gt_fp, pred_raw, pred_aligned):
    fig, axs = plt.subplots(1, 3, figsize=(22, 7))

    # GT
    gx, gy = gt_fp.exterior.xy
    axs[0].fill(gx, gy, alpha=0.6, color="gray")
    axs[0].set_title("GT Footprint")
    axs[0].set_aspect("equal")
    axs[0].axis("off")

    # RAW PRED
    px, py = pred_raw.exterior.xy
    axs[1].fill(px, py, alpha=0.6, color="red")
    axs[1].set_title("Predicted (Raw)")
    axs[1].set_aspect("equal")
    axs[1].axis("off")

    # OVERLAY
    axs[2].fill(gx, gy, alpha=0.4, color="gray", label="GT")
    pax, pay = pred_aligned.exterior.xy
    axs[2].fill(pax, pay, alpha=0.4, color="green", label="Aligned Pred")

    axs[2].set_title("Overlay")
    axs[2].set_aspect("equal")
    axs[2].axis("off")
    axs[2].legend()

    plt.tight_layout()
    plt.show()
