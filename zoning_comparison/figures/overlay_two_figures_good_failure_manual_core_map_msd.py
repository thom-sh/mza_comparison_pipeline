# ===============================================================
# REPRESENTATIVE REFERENCE–PREDICTION OVERLAYS
# Thesis-style successful and failure overlay figures
#
# Final thesis-consistent version:
# - No information boxes inside subplot panels
# - Ground-truth dwellings use calm transparent fill
# - Ground-truth stairwells use grey fill
# - Predicted dwellings and predicted stairwells use the same dash rhythm
# - Predicted stairwell is drawn on top
# - Predicted dwelling boundary parts near predicted stairwells are removed
#   before plotting, so common boundaries are not visually filled/double drawn
# - No white halo/path effects
# - Shared/common polygon boundaries are drawn only once where appropriate
# - Black subplot boxes are retained
# ===============================================================

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.patches import Polygon as MplPolygon, Patch
from matplotlib.lines import Line2D

from shapely.geometry import LineString
from shapely.ops import unary_union, linemerge
from scipy.optimize import linear_sum_assignment


# ===============================================================
# IMPORT PATH CONFIGURATION - COMMON COMPARISON MODULE
# ===============================================================

try:
    PROJECT_DIR = Path(__file__).resolve().parent
except NameError:
    # For VS Code / Jupyter interactive execution
    PROJECT_DIR = Path.cwd()

REPO_DIR = PROJECT_DIR.parents[1]

COMMON_COMPARISON_DIR = REPO_DIR / "zoning_comparison"
sys.path.insert(0, str(COMMON_COMPARISON_DIR))


from shapely.geometry import LineString
from shapely.ops import unary_union, linemerge
from scipy.optimize import linear_sum_assignment

from footprint_visualization import (
    extract_gt_apartments,
    load_predicted_zone_polygons,
    compute_predicted_footprint,
    align_pred,
    align_shape,
)

# ===============================================================
# THESIS STYLE
# ===============================================================

THESIS_COLORS = {
    "primary_blue": "#7FA6C9",
    "light_blue": "#DCEAF7",
    "mid_blue": "#AFCBE3",
    "core_grey": "#5F666D",
    "edge_grey": "#777D84",
    "legend_edge": "#BDC1C5",
    "light_grey": "#DBDBDB",
    "dark_grey": "#999999",
    "muted_red": "#B94A48",
    "muted_orange": "#C58A45",
}

DWELLING_COLORS = [
    "#DCEAF7",
]

CORE_COLOR = "#777D84"
POLYGON_EDGE = "#777D84"
LEGEND_EDGE = "#BDC1C5"

# ---------------------------------------------------------------
# Prediction styles
# ---------------------------------------------------------------
# Important:
# Predicted dwelling and predicted stairwell use the SAME dash pattern.
# Otherwise, common edges can look inconsistent.
#
# The stairwell is drawn after the dwelling and slightly thicker.
# Dwelling boundary parts close to stairwell boundaries are removed before
# plotting so the orange stairwell edge is not visually filled by blue dashes.
# ---------------------------------------------------------------

PRED_DWELLING_COLOR = "#5F8FB8"
PRED_CORE_COLOR = "#C58A45"

PRED_LINESTYLE = (0, (4.0, 2.0))
PRED_DWELLING_LINESTYLE = PRED_LINESTYLE
PRED_CORE_LINESTYLE = PRED_LINESTYLE

PRED_DWELLING_LW = 1.05
PRED_CORE_LW = 1.25

PRED_LINE_EFFECTS = None

# Distance tolerance in drawing units for removing predicted-dwelling
# boundary parts near predicted-core boundaries.
# If blue dashes still appear in orange stairwell gaps, increase to 0.22.
# If too much blue line disappears near cores, reduce to 0.12.
PRED_CORE_INTERFACE_TOL = 0.18

# Reference boundary style
REF_DWELLING_EDGE_LW = 0.70
REF_CORE_EDGE_LW = 0.70

# Optional manual control of which predicted zones should be shown as core
# in the overlay. Use 0-based predicted-zone indices.
MANUAL_PREDICTED_CORE_ZONES = {
    1925: [0],
    2030: [0],
    6599: [0],
    53: [0],
}

# Optional debugging aid: show predicted-zone index numbers in the overlay
# so you can identify which zone index to place in MANUAL_PREDICTED_CORE_ZONES.
SHOW_PREDICTED_ZONE_IDS = False
PRED_ZONE_ID_FONT_SIZE = 7

plt.rcParams.update({
    "font.family": "cmr10",
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 9,
    "axes.titleweight": "normal",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.linewidth": 0.8,
    "grid.color": "#E6E8EA",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.8,
})

# LaTeX text width:
# A4 width 21 cm - 2.7 cm left - 2.7 cm right = 15.6 cm
TEXTWIDTH_IN = 15.6 / 2.54

FIG_GOOD_GRID = (TEXTWIDTH_IN, 2.55)
FIG_FAILURE_GRID_4 = (TEXTWIDTH_IN, 4.30)
FIG_FAILURE_GRID_6 = (TEXTWIDTH_IN, 4.35)
FIG_OVERLAY_GRID = FIG_FAILURE_GRID_6


# ===============================================================
# GEOMETRY / PLOTTING HELPERS
# ===============================================================

def fix_geom(poly):
    """Make polygon robust for plotting and overlay operations."""
    if poly is None:
        return None
    if poly.is_empty:
        return poly
    return poly.buffer(0)


def iter_polygons(geom):
    """Yield Polygon parts from Polygon or MultiPolygon."""
    geom = fix_geom(geom)

    if geom is None or geom.is_empty:
        return

    if geom.geom_type == "Polygon":
        yield geom

    elif geom.geom_type == "MultiPolygon":
        for part in geom.geoms:
            if not part.is_empty:
                yield part


def iter_lines(geom):
    """Yield LineString parts from LineString, MultiLineString, or GeometryCollection."""
    if geom is None or geom.is_empty:
        return

    if geom.geom_type == "LineString":
        yield geom

    elif geom.geom_type == "MultiLineString":
        for line in geom.geoms:
            if not line.is_empty:
                yield line

    elif geom.geom_type == "GeometryCollection":
        for part in geom.geoms:
            yield from iter_lines(part)


def safe_linemerge(line_geom):
    """
    Try to merge line fragments into longer lines.

    This improves dash consistency because Matplotlib restarts the dash pattern
    for every separately plotted LineString.
    """
    if line_geom is None or line_geom.is_empty:
        return line_geom

    try:
        return linemerge(line_geom)
    except ValueError:
        return line_geom


def plot_filled_polygon(
    ax,
    geom,
    facecolor,
    edgecolor="none",
    alpha=0.35,
    lw=0.0,
    zorder=1,
):
    """
    Plot filled Polygon/MultiPolygon.

    Edges are normally plotted separately to avoid double-thick shared edges.
    """
    for poly in iter_polygons(geom):
        exterior = np.asarray(poly.exterior.coords)

        patch = MplPolygon(
            exterior,
            closed=True,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=lw,
            alpha=alpha,
            zorder=zorder,
        )
        ax.add_patch(patch)

        for interior in poly.interiors:
            hole = np.asarray(interior.coords)
            hole_patch = MplPolygon(
                hole,
                closed=True,
                facecolor="white",
                edgecolor="none",
                linewidth=0.0,
                alpha=1.0,
                zorder=zorder + 0.1,
            )
            ax.add_patch(hole_patch)


def plot_outline(
    ax,
    geom,
    color,
    lw=1.0,
    linestyle="-",
    alpha=1.0,
    zorder=5,
):
    """Plot outline of Polygon/MultiPolygon."""
    for poly in iter_polygons(geom):
        x, y = poly.exterior.xy
        ax.plot(
            x,
            y,
            color=color,
            linewidth=lw,
            linestyle=linestyle,
            alpha=alpha,
            zorder=zorder,
            solid_joinstyle="miter",
            dash_joinstyle="miter",
            solid_capstyle="butt",
            dash_capstyle="butt",
        )

        for interior in poly.interiors:
            hx, hy = interior.xy
            ax.plot(
                hx,
                hy,
                color=color,
                linewidth=max(lw - 0.35, 0.35),
                linestyle=linestyle,
                alpha=alpha,
                zorder=zorder,
                solid_joinstyle="miter",
                dash_joinstyle="miter",
                solid_capstyle="butt",
                dash_capstyle="butt",
            )


def plot_unique_polygon_boundaries(
    ax,
    polygons,
    color,
    lw=1.0,
    linestyle="-",
    alpha=1.0,
    zorder=6,
    path_effects=None,
):
    """
    Plot polygon boundaries only once.

    This avoids shared boundaries between adjacent reference dwellings becoming
    thicker because the same line was drawn multiple times.
    """
    boundaries = []

    for geom in polygons:
        for poly in iter_polygons(geom):
            boundaries.append(LineString(poly.exterior.coords))

            for interior in poly.interiors:
                boundaries.append(LineString(interior.coords))

    if not boundaries:
        return

    merged_boundaries = unary_union(boundaries)
    merged_boundaries = safe_linemerge(merged_boundaries)

    for line in iter_lines(merged_boundaries):
        x, y = line.xy
        line_obj, = ax.plot(
            x,
            y,
            color=color,
            linewidth=lw,
            linestyle=linestyle,
            alpha=alpha,
            zorder=zorder,
            solid_joinstyle="miter",
            dash_joinstyle="miter",
            solid_capstyle="butt",
            dash_capstyle="butt",
        )

        if path_effects is not None:
            line_obj.set_path_effects(path_effects)


def collect_polygon_boundaries(polygons):
    """
    Collect polygon boundaries as one merged line geometry.
    """
    boundaries = []

    for geom in polygons:
        for poly in iter_polygons(geom):
            boundaries.append(LineString(poly.exterior.coords))

            for interior in poly.interiors:
                boundaries.append(LineString(interior.coords))

    if not boundaries:
        return None

    merged = unary_union(boundaries)
    return safe_linemerge(merged)


def remove_lines_near_mask(line_geom, mask_geom, tol=PRED_CORE_INTERFACE_TOL):
    """
    Remove parts of a line geometry that lie close to a mask geometry.

    Used here to remove predicted dwelling boundaries near predicted stairwell
    boundaries. This prevents common dwelling--stairwell edges from being drawn
    twice, where blue dwelling dashes would otherwise appear inside the orange
    stairwell dash gaps.
    """
    if line_geom is None or line_geom.is_empty:
        return line_geom

    if mask_geom is None or mask_geom.is_empty:
        return line_geom

    mask = mask_geom.buffer(
        tol,
        cap_style=2,
        join_style=2,
    )

    cleaned = line_geom.difference(mask)

    if cleaned is None or cleaned.is_empty:
        return cleaned

    return safe_linemerge(cleaned)


def plot_line_geometry(
    ax,
    line_geom,
    color,
    lw=1.0,
    linestyle="-",
    alpha=1.0,
    zorder=6,
    path_effects=None,
):
    """
    Plot LineString, MultiLineString, or GeometryCollection.
    """
    if line_geom is None or line_geom.is_empty:
        return

    for line in iter_lines(line_geom):
        x, y = line.xy
        line_obj, = ax.plot(
            x,
            y,
            color=color,
            linewidth=lw,
            linestyle=linestyle,
            alpha=alpha,
            zorder=zorder,
            solid_joinstyle="miter",
            dash_joinstyle="miter",
            solid_capstyle="butt",
            dash_capstyle="butt",
        )

        if path_effects is not None:
            line_obj.set_path_effects(path_effects)


def set_equal_square_limits(ax, geometries, pad_ratio=0.10):
    """
    Set square x/y limits so every subplot keeps the same graph-box shape.
    """
    valid_geoms = [g for g in geometries if g is not None and not g.is_empty]

    if not valid_geoms:
        return

    union = unary_union(valid_geoms)
    minx, miny, maxx, maxy = union.bounds

    cx = 0.5 * (minx + maxx)
    cy = 0.5 * (miny + maxy)

    width = maxx - minx
    height = maxy - miny

    half_range = 0.5 * max(width, height)
    pad = half_range * pad_ratio
    half_range = half_range + pad

    ax.set_xlim(cx - half_range, cx + half_range)
    ax.set_ylim(cy - half_range, cy + half_range)


# ===============================================================
# GEOMETRY / MATCHING HELPERS
# ===============================================================

def compute_iou_matrix(gt_regions, pred_zones):
    """Pairwise IoU matrix between GT and predicted polygons."""
    n_gt = len(gt_regions)
    n_pred = len(pred_zones)
    iou_mat = np.zeros((n_gt, n_pred), dtype=float)

    for i, g in enumerate(gt_regions):
        for j, p in enumerate(pred_zones):
            inter = g.intersection(p).area
            union = g.union(p).area
            iou_mat[i, j] = 0.0 if union == 0 else inter / union

    return iou_mat


def match_zones(iou_mat, iou_threshold=0.15):
    """
    Hungarian matching using IoU.
    Matches below the threshold are treated as invalid.
    """
    if iou_mat.size == 0:
        return {}

    cost = 1.0 - iou_mat
    row_ind, col_ind = linear_sum_assignment(cost)

    match_map = {}
    for gi, pj in zip(row_ind, col_ind):
        if iou_mat[gi, pj] >= iou_threshold:
            match_map[int(gi)] = int(pj)

    return match_map


def prepare_case_geometry(
    building_id,
    gt_path,
    pred_path,
    manual_predicted_core_zones=None,
):
    """
    Load GT and MZA prediction, align prediction to GT footprint,
    and compute matching information.
    """
    gt_apts, gt_cores, gt_fp = extract_gt_apartments(gt_path)

    gt_apts = [fix_geom(p) for p in gt_apts]
    gt_cores = [fix_geom(p) for p in gt_cores]
    gt_regions = gt_apts + gt_cores

    region_labels = (
        [f"A{i}" for i in range(len(gt_apts))]
        + [f"S{i}" for i in range(len(gt_cores))]
    )

    pred_zones_raw = load_predicted_zone_polygons(pred_path)
    pred_zones_raw = [fix_geom(p) for p in pred_zones_raw]

    pred_fp = compute_predicted_footprint(pred_path)
    pred_fp = fix_geom(pred_fp)

    if pred_fp is None or pred_fp.is_empty:
        raise ValueError(f"No predicted footprint found for building {building_id}")

    pred_fp_aligned, rot, dx, dy = align_pred(pred_fp, gt_fp)

    pred_zones_aligned = [
        fix_geom(align_shape(z, rot, dx, dy, origin=pred_fp.centroid))
        for z in pred_zones_raw
    ]

    iou_mat = compute_iou_matrix(gt_regions, pred_zones_aligned)
    match_map = match_zones(iou_mat, iou_threshold=0.15)

    # -----------------------------------------------------------
    # Classify predicted zones for visualisation
    # -----------------------------------------------------------
    pred_type = ["unmatched"] * len(pred_zones_aligned)

    # First: classify by Hungarian match
    for gi, pj in match_map.items():
        gt_label = region_labels[gi]

        if gt_label.startswith("S"):
            pred_type[pj] = "core"
        else:
            pred_type[pj] = "dwelling"

    # Second: optional manual override for predicted-core display.
    manual_core_indices = []
    if manual_predicted_core_zones is not None:
        manual_core_indices = manual_predicted_core_zones.get(int(building_id), [])

    for pj in manual_core_indices:
        if 0 <= int(pj) < len(pred_type):
            pred_type[int(pj)] = "core"

    # -----------------------------------------------------------
    # Building-level summary statistics
    # -----------------------------------------------------------
    mean_iou_overall = 0.0

    if len(gt_regions) > 0:
        matched_ious = []

        for gi in range(len(gt_regions)):
            pj = match_map.get(gi, None)
            matched_ious.append(0.0 if pj is None else iou_mat[gi, pj])

        mean_iou_overall = float(np.mean(matched_ious))

    stats = {
        "mean_iou_overall": mean_iou_overall,
        "n_gt_regions": len(gt_regions),
        "n_pred_regions": len(pred_zones_aligned),
        "delta_n": len(pred_zones_aligned) - len(gt_regions),
    }

    return {
        "building_id": building_id,
        "gt_apts": gt_apts,
        "gt_cores": gt_cores,
        "gt_fp": gt_fp,
        "pred_zones": pred_zones_aligned,
        "pred_type": pred_type,
        "pred_fp": pred_fp_aligned,
        "pred_zone_ids": list(range(len(pred_zones_aligned))),
        "manual_core_indices": manual_core_indices,
        "stats": stats,
    }


# ===============================================================
# DIAGNOSTIC TABLE HELPERS
# ===============================================================

def normalise_diagnosis(text):
    """Shorten diagnosis labels for possible panel display."""
    text = str(text)

    replacements = {
        "Good geometric and topological match": "Good match",
        "Moderate geometric match": "Moderate match",
        "Weak geometric match": "Weak geometric match",
        "Correct topology but high area error": "High area error",
        "Correct topology but shifted zones": "Shifted zones",
        "Correct count but topology mismatch": "Topology mismatch",
        "Under-zoned / missing regions": "Under-zoned",
        "Over-zoned / extra regions": "Over-zoned",
        "Missing + extra regions": "Structural mismatch",
    }

    return replacements.get(text, text)


def get_case_info_from_csv(diagnostic_csv, building_id):
    """
    Read diagnosis, IoU and delta N from diagnostic CSV if available.
    Falls back gracefully if columns are missing.
    """
    if diagnostic_csv is None or not os.path.exists(diagnostic_csv):
        return {}

    df = pd.read_csv(diagnostic_csv)

    if "building_id" not in df.columns:
        return {}

    row = df[df["building_id"].astype(str) == str(building_id)]

    if row.empty:
        return {}

    row = row.iloc[0].to_dict()

    info = {}

    if "diagnosis" in row:
        info["diagnosis"] = row["diagnosis"]

    if "mean_iou_overall" in row:
        info["mean_iou_overall"] = row["mean_iou_overall"]

    if "n_gt_regions" in row and "n_pred_regions" in row:
        info["delta_n"] = int(row["n_pred_regions"]) - int(row["n_gt_regions"])

    elif "zone_diff" in row:
        info["delta_n"] = int(row["zone_diff"])

    return info


def pick_cases_automatically(diagnostic_csv, category_map):
    """
    Pick one representative building ID for each panel category from the diagnostic CSV.
    """
    df = pd.read_csv(diagnostic_csv)

    if "building_id" not in df.columns or "diagnosis" not in df.columns:
        raise ValueError(
            "Diagnostic CSV must contain 'building_id' and 'diagnosis' columns."
        )

    selected = {}

    for panel_title, possible_diagnoses in category_map.items():
        subset = df[df["diagnosis"].isin(possible_diagnoses)].copy()

        if subset.empty:
            selected[panel_title] = None
            continue

        if "mean_iou_overall" in subset.columns:
            if panel_title == "Good match":
                subset = subset.sort_values("mean_iou_overall", ascending=False)

            elif panel_title == "Moderate match":
                subset["dist_to_060"] = (
                    subset["mean_iou_overall"] - 0.60
                ).abs()
                subset = subset.sort_values("dist_to_060", ascending=True)

            else:
                subset = subset.sort_values("mean_iou_overall", ascending=True)

        selected[panel_title] = int(subset.iloc[0]["building_id"])

    return selected


def pick_good_cases_automatically(diagnostic_csv, n=3):
    """Pick n high-IoU good-match examples from the diagnostic table."""
    df = pd.read_csv(diagnostic_csv)

    if "building_id" not in df.columns or "diagnosis" not in df.columns:
        raise ValueError(
            "Diagnostic CSV must contain 'building_id' and 'diagnosis' columns."
        )

    subset = df[df["diagnosis"] == "Good geometric and topological match"].copy()

    if subset.empty:
        return []

    if "mean_iou_overall" in subset.columns:
        subset = subset.sort_values("mean_iou_overall", ascending=False)

    return [int(x) for x in subset["building_id"].head(n).tolist()]


def fill_good_cases_if_needed(good_cases, diagnostic_csv):
    """Fill None values in the good-cases dictionary using automatic selection."""
    if not any(bid is None for bid in good_cases.values()):
        return good_cases

    auto_ids = pick_good_cases_automatically(
        diagnostic_csv=diagnostic_csv,
        n=len(good_cases),
    )

    auto_iter = iter(auto_ids)
    filled = {}

    for key, bid in good_cases.items():
        if bid is None:
            filled[key] = next(auto_iter, None)
        else:
            filled[key] = bid

    return filled


def base_failure_category(label):
    """
    Convert display labels such as 'Structural mismatch 1' back to the
    diagnostic category used in the CSV.
    """
    label = str(label).strip()

    if label.startswith("Structural mismatch"):
        return "Structural mismatch"
    if label.startswith("Moderate geometric match") or label.startswith("Moderate match"):
        return "Moderate geometric match"
    if label.startswith("High area error") or label.startswith("Shifted zones"):
        return "High area error / shifted zones"
    if label.startswith("Under-zoned"):
        return "Under-zoned"
    if label.startswith("Over-zoned"):
        return "Over-zoned"
    if label.startswith("Topology mismatch"):
        return "Topology mismatch"
    if label.startswith("Weak geometric match"):
        return "Weak geometric match"

    return label


def fill_failure_cases_if_needed(failure_cases, diagnostic_csv):
    """
    Fill None values in the failure-cases dictionary using the category map.
    """
    if not any(bid is None for bid in failure_cases.values()):
        return failure_cases

    category_map = {
        "Under-zoned": [
            "Under-zoned",
            "Under-zoned / missing regions",
        ],
        "Over-zoned": [
            "Over-zoned",
            "Over-zoned / extra regions",
        ],
        "Structural mismatch": [
            "Structural mismatch",
            "Missing + extra regions",
        ],
        "High area error / shifted zones": [
            "Correct topology but high area error",
            "Correct topology but shifted zones",
        ],
        "Topology mismatch": [
            "Correct count but topology mismatch",
        ],
        "Weak geometric match": [
            "Weak geometric match",
        ],
        "Moderate geometric match": [
            "Moderate geometric match",
        ],
    }

    auto_cases = pick_cases_automatically(
        diagnostic_csv=diagnostic_csv,
        category_map=category_map,
    )

    filled = {}

    for category, bid in failure_cases.items():
        if bid is None:
            lookup_category = base_failure_category(category)
            filled[category] = auto_cases.get(lookup_category, None)
        else:
            filled[category] = bid

    return filled


# ===============================================================
# PLOT ONE PANEL
# ===============================================================

def annotate_predicted_zone_ids(ax, pred_zones, pred_zone_ids):
    """Optionally annotate predicted-zone index numbers for manual core selection."""
    for zone_id, geom in zip(pred_zone_ids, pred_zones):
        if geom is None or geom.is_empty:
            continue

        c = geom.representative_point()
        ax.text(
            c.x,
            c.y,
            str(zone_id),
            ha="center",
            va="center",
            fontsize=PRED_ZONE_ID_FONT_SIZE,
            color="#222222",
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor="white",
                edgecolor=LEGEND_EDGE,
                linewidth=0.4,
                alpha=0.9,
            ),
            zorder=10,
        )


def plot_overlay_panel(
    ax,
    case_data,
    title=None,
    diagnosis=None,
    mean_iou=None,
    delta_n=None,
    bottom_label=None,
):
    """
    Plot one reference-prediction overlay panel.
    """
    gt_apts = case_data["gt_apts"]
    gt_cores = case_data["gt_cores"]
    gt_fp = case_data["gt_fp"]

    pred_zones = case_data["pred_zones"]
    pred_type = case_data["pred_type"]
    pred_zone_ids = case_data.get("pred_zone_ids", list(range(len(pred_zones))))

    stats = case_data["stats"]

    if diagnosis is None:
        diagnosis = title

    if mean_iou is None:
        mean_iou = stats["mean_iou_overall"]

    if delta_n is None:
        delta_n = stats["delta_n"]

    # -----------------------------------------------------------
    # Ground-truth dwellings: transparent filled polygons
    # -----------------------------------------------------------
    for i, poly in enumerate(gt_apts):
        plot_filled_polygon(
            ax,
            poly,
            facecolor=DWELLING_COLORS[i % len(DWELLING_COLORS)],
            edgecolor="none",
            alpha=0.34,
            lw=0.0,
            zorder=1,
        )

    # -----------------------------------------------------------
    # Ground-truth stairwell/core fill
    # -----------------------------------------------------------
    for core in gt_cores:
        plot_filled_polygon(
            ax,
            core,
            facecolor=CORE_COLOR,
            edgecolor="none",
            alpha=0.45,
            lw=0.0,
            zorder=2,
        )

    # -----------------------------------------------------------
    # Ground-truth boundaries
    # -----------------------------------------------------------
    plot_unique_polygon_boundaries(
        ax,
        gt_apts,
        color=POLYGON_EDGE,
        lw=REF_DWELLING_EDGE_LW,
        linestyle="-",
        alpha=0.75,
        zorder=3,
    )

    plot_unique_polygon_boundaries(
        ax,
        gt_cores,
        color=POLYGON_EDGE,
        lw=REF_CORE_EDGE_LW,
        linestyle="-",
        alpha=0.85,
        zorder=4,
    )

    # -----------------------------------------------------------
    # Prediction: category-priority plotting
    # -----------------------------------------------------------
    pred_dwelling_like = []
    pred_core_polys = []

    for poly, ptype in zip(pred_zones, pred_type):
        if ptype == "core":
            pred_core_polys.append(poly)
        else:
            # Dwellings + unmatched predictions are shown as predicted dwelling outlines.
            pred_dwelling_like.append(poly)

    pred_dwelling_boundary = collect_polygon_boundaries(pred_dwelling_like)
    pred_core_boundary = collect_polygon_boundaries(pred_core_polys)

    # Remove blue predicted-dwelling boundary near orange predicted-core boundary.
    # This prevents common dwelling--core edges from becoming visually filled.
    pred_dwelling_boundary = remove_lines_near_mask(
        pred_dwelling_boundary,
        pred_core_boundary,
        tol=PRED_CORE_INTERFACE_TOL,
    )

    plot_line_geometry(
        ax,
        pred_dwelling_boundary,
        color=PRED_DWELLING_COLOR,
        lw=PRED_DWELLING_LW,
        linestyle=PRED_DWELLING_LINESTYLE,
        alpha=1.0,
        zorder=8,
        path_effects=PRED_LINE_EFFECTS,
    )

    plot_line_geometry(
        ax,
        pred_core_boundary,
        color=PRED_CORE_COLOR,
        lw=PRED_CORE_LW,
        linestyle=PRED_CORE_LINESTYLE,
        alpha=1.0,
        zorder=9,
        path_effects=PRED_LINE_EFFECTS,
    )

    if SHOW_PREDICTED_ZONE_IDS:
        annotate_predicted_zone_ids(ax, pred_zones, pred_zone_ids)

    # -----------------------------------------------------------
    # Axis / graph box style
    # -----------------------------------------------------------
    if title:
        ax.set_title(title, pad=4)
    else:
        ax.set_title("")

    if bottom_label:
        ax.set_xlabel(bottom_label, labelpad=3, fontsize=8)
    else:
        ax.set_xlabel("")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)

    all_geoms = gt_apts + gt_cores + [gt_fp] + pred_zones
    set_equal_square_limits(ax, all_geoms, pad_ratio=0.10)


# ===============================================================
# LEGEND AND GRID PLOTTING
# ===============================================================

def make_overlay_legend_handles():
    """Create shared legend handles."""
    return [
        Patch(
            facecolor=THESIS_COLORS["light_blue"],
            edgecolor=POLYGON_EDGE,
            linewidth=REF_DWELLING_EDGE_LW,
            alpha=0.85,
            label="Ground-truth dwelling",
        ),
        Patch(
            facecolor=CORE_COLOR,
            edgecolor=POLYGON_EDGE,
            linewidth=REF_CORE_EDGE_LW,
            alpha=0.85,
            label="Ground-truth stairwell",
        ),
        Line2D(
            [0],
            [0],
            color=PRED_DWELLING_COLOR,
            lw=PRED_DWELLING_LW,
            linestyle=PRED_DWELLING_LINESTYLE,
            label="Predicted dwelling",
        ),
        Line2D(
            [0],
            [0],
            color=PRED_CORE_COLOR,
            lw=PRED_CORE_LW,
            linestyle=PRED_CORE_LINESTYLE,
            label="Predicted stairwell",
        ),
    ]


def add_shared_legend(fig, y_anchor=0.02):
    """Add one thesis-style legend below the subplot grid."""
    legend = fig.legend(
        handles=make_overlay_legend_handles(),
        loc="lower center",
        ncol=4,
        frameon=True,
        bbox_to_anchor=(0.5, y_anchor),
        handlelength=3.2,
        columnspacing=1.0,
        borderpad=0.45,
    )

    legend.get_frame().set_edgecolor(LEGEND_EDGE)
    legend.get_frame().set_linewidth(0.7)
    legend.get_frame().set_facecolor("white")


def load_and_plot_case(
    ax,
    bid,
    panel_key,
    gt_root,
    pred_root,
    diagnostic_csv=None,
    top_title=None,
    bottom_label=None,
):
    """
    Load one case and plot it on the provided axis.
    """
    if bid is None:
        ax.text(
            0.5,
            0.5,
            "No case selected",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(bottom_label or "")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(0.8)
        return

    gt_path = os.path.join(gt_root, f"{bid}.pickle")
    pred_path = os.path.join(pred_root, f"building_data_{bid}.pkl")

    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"GT file not found: {gt_path}")

    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")

    case_data = prepare_case_geometry(
        building_id=bid,
        gt_path=gt_path,
        pred_path=pred_path,
        manual_predicted_core_zones=MANUAL_PREDICTED_CORE_ZONES,
    )

    csv_info = get_case_info_from_csv(diagnostic_csv, bid)

    diagnosis = csv_info.get("diagnosis", panel_key)
    mean_iou = csv_info.get(
        "mean_iou_overall",
        case_data["stats"]["mean_iou_overall"],
    )
    delta_n = csv_info.get(
        "delta_n",
        case_data["stats"]["delta_n"],
    )

    plot_overlay_panel(
        ax=ax,
        case_data=case_data,
        title=top_title,
        diagnosis=diagnosis,
        mean_iou=float(mean_iou),
        delta_n=int(delta_n),
        bottom_label=bottom_label,
    )


def plot_good_match_overlay_grid(
    good_cases,
    gt_root,
    pred_root,
    diagnostic_csv=None,
    output_pdf=None,
    output_png=None,
):
    """
    Plot three successful/good-match examples in one row.
    """
    panel_items = list(good_cases.items())

    fig, axes = plt.subplots(
        1,
        3,
        figsize=FIG_GOOD_GRID,
        constrained_layout=False,
    )

    axes = np.asarray(axes).ravel()

    for ax, (panel_key, bid) in zip(axes, panel_items):
        load_and_plot_case(
            ax=ax,
            bid=bid,
            panel_key=panel_key,
            gt_root=gt_root,
            pred_root=pred_root,
            diagnostic_csv=diagnostic_csv,
            top_title=None,
            bottom_label=None,
        )

    for ax in axes[len(panel_items):]:
        ax.axis("off")

    add_shared_legend(fig, y_anchor=0.015)

    fig.subplots_adjust(
        left=0.025,
        right=0.985,
        top=0.94,
        bottom=0.22,
        wspace=0.16,
        hspace=0.0,
    )

    if output_pdf:
        fig.savefig(output_pdf, bbox_inches="tight", dpi=300)

    if output_png:
        fig.savefig(output_png, bbox_inches="tight", dpi=300)

    plt.show()


def plot_failure_overlay_grid(
    failure_cases,
    gt_root,
    pred_root,
    diagnostic_csv=None,
    output_pdf=None,
    output_png=None,
):
    """
    Plot representative failure cases.

    Supports either 4 panels as 2 x 2 or 6 panels as 2 x 3.
    """
    panel_items = list(failure_cases.items())
    n_panels = len(panel_items)

    if n_panels <= 4:
        nrows, ncols = 2, 2
        figsize = FIG_FAILURE_GRID_4
    else:
        nrows, ncols = 2, 3
        figsize = FIG_FAILURE_GRID_6

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        constrained_layout=False,
    )

    axes = np.asarray(axes).ravel()

    for ax, (category, bid) in zip(axes, panel_items):
        load_and_plot_case(
            ax=ax,
            bid=bid,
            panel_key=category,
            gt_root=gt_root,
            pred_root=pred_root,
            diagnostic_csv=diagnostic_csv,
            top_title=None,
            bottom_label=category,
        )

    for ax in axes[n_panels:]:
        ax.axis("off")

    add_shared_legend(fig, y_anchor=0.015)

    fig.subplots_adjust(
        left=0.035,
        right=0.985,
        top=0.965,
        bottom=0.18,
        wspace=0.16,
        hspace=0.34,
    )

    if output_pdf:
        fig.savefig(output_pdf, bbox_inches="tight", dpi=300)

    if output_png:
        fig.savefig(output_png, bbox_inches="tight", dpi=300)

    plt.show()


# ===============================================================
# MAIN
# ===============================================================

def main():

    # zoning_comparison_metrics/
    PROJECT_DIR = Path(__file__).resolve().parent
    REPO_DIR = PROJECT_DIR.parents[1]

    DATASET = "msd"

    GT_ROOT = (
        REPO_DIR
        / "msd_ground_truth_extraction"
        / "data"
        / "ground_truth"
    )

    PRED_ROOT = (
        REPO_DIR
        / "data"
        / "msd"
        / "msd_predicted_buildings"
        / "pkl"
    )

    OUTPUT_DIR = (
        PROJECT_DIR
        / "output"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    DIAGNOSTIC_CSV = (
        PROJECT_DIR
        / "results"
        / DATASET
        / f"diagnostic_table_with_topology_{DATASET}.csv"
    )

    # -----------------------------------------------------------
    # Figure 1: three successful examples
    # -----------------------------------------------------------
    GOOD_CASES = {
        "Good example 1": 68,
        "Good example 2": 75,
        "Good example 3": 329,
    }

    GOOD_CASES = fill_good_cases_if_needed(
        good_cases=GOOD_CASES,
        diagnostic_csv=DIAGNOSTIC_CSV,
    )

    print("Selected good-match overlay cases:")
    for title, bid in GOOD_CASES.items():
        print(f"  {title}: {bid}")

    output_good_pdf = os.path.join(
        OUTPUT_DIR,
        "successful_reference_prediction_overlays_msd.pdf",
    )

    output_good_png = os.path.join(
        OUTPUT_DIR,
        "successful_reference_prediction_overlays_msd.png",
    )

    plot_good_match_overlay_grid(
        good_cases=GOOD_CASES,
        gt_root=GT_ROOT,
        pred_root=PRED_ROOT,
        diagnostic_csv=DIAGNOSTIC_CSV,
        output_pdf=output_good_pdf,
        output_png=None,
    )

    # -----------------------------------------------------------
    # Figure 2: representative failure examples
    # -----------------------------------------------------------
    FAILURE_CASES_4 = {
        "Under-zoned": None,
        "Over-zoned": None,
        "Structural mismatch": None,
        "High area error / shifted zones": None,
    }

    FAILURE_CASES_6 = {
        "Structural mismatch 1": 1925,
        "Structural mismatch 2": 2030,
        "Structural mismatch 3": 6599,
        "Over-zoned": 176,
        "Under-zoned": 405,
        "High area error / shifted zones": 3002,
    }

    USE_SIX_FAILURE_PANELS = True

    FAILURE_CASES = FAILURE_CASES_6 if USE_SIX_FAILURE_PANELS else FAILURE_CASES_4

    FAILURE_CASES = fill_failure_cases_if_needed(
        failure_cases=FAILURE_CASES,
        diagnostic_csv=DIAGNOSTIC_CSV,
    )

    print("Selected failure overlay cases:")
    for category, bid in FAILURE_CASES.items():
        print(f"  {category}: {bid}")

    output_failure_pdf = os.path.join(
        OUTPUT_DIR,
        "failure_reference_prediction_overlays_msd.pdf",
    )

    plot_failure_overlay_grid(
        failure_cases=FAILURE_CASES,
        gt_root=GT_ROOT,
        pred_root=PRED_ROOT,
        diagnostic_csv=DIAGNOSTIC_CSV,
        output_pdf=output_failure_pdf,
        output_png=None,
    )


if __name__ == "__main__":
    main()