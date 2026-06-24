"""
Create a 4-panel thesis figure for the footprint-orientation preprocessing step.

Updated logic:
(a) Stored ground-truth polygons.
(b) Merged and cleaned external footprint.
(c) Possible core-side edge is identified from the core edge(s) closest to the outer footprint boundary.
    If more than one edge is similarly close, the final edge is selected using the dominant exterior
    façade direction of the footprint.
(d) Final oriented footprint, with the selected façade direction parallel to the x-axis and the
    selected core side placed on the lower side.

Edit only the CONFIGURATION block in main().
"""

import os
import math
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from shapely.affinity import rotate
from shapely.geometry import JOIN_STYLE, LineString, Point, Polygon, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union


# ============================================================
#                       THESIS STYLE
# ============================================================

LEGEND_EDGE_COLOR = "#bdc1c5"
POLYGON_EDGE_COLOR = "#777d84"
CORE_EDGE_COLOR = "#777d84"

DWELLING_COLORS = [
    "#DCEAF7",
    "#AFCBE3",
    "#AFCBE3",
    "#DBDBDB",
    "#BABABA",
    "#999999",
]

CORE_COLOR = "#5F666D"
FOOTPRINT_FILL_COLOR = "#F7F8F9"
GRID_COLOR = "#bdc1c5"
TEXT_COLOR = "#222222"


def apply_thesis_style() -> None:
    """Apply the thesis-wide Matplotlib styling used for publication figures."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIX Two Text", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",

        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,

        "axes.titleweight": "normal",
        "axes.labelweight": "normal",
        "axes.edgecolor": POLYGON_EDGE_COLOR,
        "axes.linewidth": 0.6,

        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "text.color": TEXT_COLOR,

        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",

        "grid.color": GRID_COLOR,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.35,

        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def dwelling_color(index: int) -> str:
    """Return a repeatable thesis-palette color for dwelling polygons."""
    return DWELLING_COLORS[index % len(DWELLING_COLORS)]


# ============================================================
#                       CONFIG TYPES
# ============================================================

@dataclass
class OrientationResult:
    raw_polygons: List[Polygon]
    apartment_polygons: List[Polygon]
    core_polygons: List[Polygon]
    merged_footprint: Polygon

    selected_core_polygon: Polygon
    selected_core_edge: Tuple[np.ndarray, np.ndarray]
    selected_core_midpoint: np.ndarray
    nearest_facade_edge: Tuple[np.ndarray, np.ndarray]
    similar_facade_edges: List[Tuple[np.ndarray, np.ndarray]]

    chosen_facade_direction_deg: float
    rotation_angle_deg: float
    flipped_180: bool
    core_centroid_y_after_rotation: float
    footprint_centroid_y_after_rotation: float
    dominant_direction_total_length: float
    selected_distance_to_boundary: float

    oriented_footprint: Polygon
    oriented_core_polygons: List[Polygon]


# ============================================================
#                       GEOMETRY HELPERS
# ============================================================

def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def geometry_to_polygons(obj: Any) -> List[Polygon]:
    """
    Convert different stored geometry formats into valid Shapely Polygons.

    Supported formats:
    - list/tuple of coordinate pairs
    - Shapely Polygon
    - Shapely MultiPolygon
    - GeoJSON-like geometry dictionary
    """
    polygons: List[Polygon] = []

    if obj is None:
        return polygons

    if isinstance(obj, BaseGeometry):
        geom = obj
    elif isinstance(obj, dict) and "type" in obj:
        try:
            geom = shape(obj)
        except Exception:
            return polygons
    else:
        try:
            geom = Polygon(obj)
        except Exception:
            return polygons

    if geom.is_empty:
        return polygons

    if not geom.is_valid:
        geom = geom.buffer(0)

    if geom.is_empty:
        return polygons

    if geom.geom_type == "Polygon":
        if geom.area > 0:
            polygons.append(geom)

    elif geom.geom_type == "MultiPolygon":
        for part in geom.geoms:
            if not part.is_valid:
                part = part.buffer(0)
            if not part.is_empty and part.area > 0:
                polygons.append(part)

    return polygons


def extract_polygons_from_floorplan_object(data: Dict[str, Any]):
    """
    Expected format:
        data["floor_plan"] = [
            {"room_type": 0, "polygon": ...},  # apartment/dwelling
            {"room_type": 1, "polygon": ...},  # stair/core
        ]

    room_type 0 = apartment/dwelling
    room_type 1 = stair/core
    """
    if not isinstance(data, dict) or "floor_plan" not in data:
        raise ValueError("Expected a dictionary with key 'floor_plan'.")

    apartment_polys: List[Polygon] = []
    core_polys: List[Polygon] = []
    all_polys: List[Polygon] = []

    for entry in data.get("floor_plan", []):
        if not isinstance(entry, dict) or "polygon" not in entry:
            continue

        parts = geometry_to_polygons(entry["polygon"])
        all_polys.extend(parts)

        if entry.get("room_type") == 0:
            apartment_polys.extend(parts)
        elif entry.get("room_type") == 1:
            core_polys.extend(parts)

    if not all_polys:
        raise ValueError("No valid polygons were found in the floor_plan object.")
    if not apartment_polys:
        raise ValueError("No apartment polygons with room_type == 0 were found.")
    if not core_polys:
        raise ValueError("No core/stair polygons with room_type == 1 were found.")

    return all_polys, apartment_polys, core_polys


def footprint_edges(footprint: Polygon) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return exterior footprint edges as numpy point pairs."""
    coords = list(footprint.exterior.coords)
    return [
        (np.array(coords[i], dtype=float), np.array(coords[i + 1], dtype=float))
        for i in range(len(coords) - 1)
    ]


def edge_length(edge: Tuple[np.ndarray, np.ndarray]) -> float:
    a, b = edge
    return float(np.linalg.norm(b - a))


def angle_of_edge(p0: np.ndarray, p1: np.ndarray) -> float:
    dx, dy = p1 - p0
    return math.degrees(math.atan2(dy, dx))


def edge_direction_deg(p0: np.ndarray, p1: np.ndarray) -> float:
    """
    Return undirected edge direction in [0, 180).
    Opposite directions along the same line are treated as equivalent.
    """
    return angle_of_edge(p0, p1) % 180.0


def angular_diff(a: float, b: float) -> float:
    """Smallest difference between two undirected directions in [0, 180)."""
    d = abs(a - b)
    return min(d, 180.0 - d)


def largest_polygon(geom: BaseGeometry) -> Polygon:
    """Return the main polygon from a Polygon or MultiPolygon."""
    if geom.geom_type == "Polygon":
        return geom
    if geom.geom_type == "MultiPolygon":
        return max(geom.geoms, key=lambda g: g.area)
    raise ValueError(f"Expected Polygon or MultiPolygon, got {geom.geom_type}")


def create_merged_footprint(all_polygons: List[Polygon]) -> Polygon:
    """
    Merge all apartment and core polygons into one external footprint.

    The positive/negative buffer closes narrow gaps and the simplify step removes
    small boundary irregularities while keeping the outline close to the original.
    """
    merged = unary_union(all_polygons)
    cleaned = (
        merged
        .buffer(0.5, join_style=JOIN_STYLE.mitre)
        .buffer(-0.4, join_style=JOIN_STYLE.mitre)
    )

    fp = largest_polygon(cleaned)
    fp = fp.simplify(0.05)

    if not fp.is_valid:
        fp = fp.buffer(0)

    return fp


# ============================================================
#             CORE EDGE AND DOMINANT FACADE LOGIC
# ============================================================

def riser_to_facade_index(
    riser_midpoint: np.ndarray,
    foot_edges: List[Tuple[np.ndarray, np.ndarray]],
) -> Optional[int]:
    """Return the index of the exterior footprint edge closest to the selected core-edge midpoint."""
    closest_idx = None
    closest_dist = 1e18
    mid_pt = Point(riser_midpoint)

    for idx, (a, b) in enumerate(foot_edges):
        dist = LineString([a, b]).distance(mid_pt)
        if dist < closest_dist:
            closest_dist = dist
            closest_idx = idx

    return closest_idx


def direction_total_length(
    direction_deg: float,
    foot_dirs: List[float],
    foot_lens: List[float],
    tol: float,
) -> float:
    """
    Calculate the total length of all footprint exterior edges with direction similar
    to direction_deg. This total length is used as the dominance measure for that
    façade direction.
    """
    total = 0.0
    for d_edge, length in zip(foot_dirs, foot_lens):
        if angular_diff(d_edge, direction_deg) <= tol:
            total += length
    return total


def collect_core_outer_edge_candidates(
    core_polys: List[Polygon],
    footprint: Polygon,
    min_edge_length: float = 0.30,
    tie_tolerance: float = 0.20,
    tol: float = 10.0,
) -> List[Dict[str, Any]]:
    """
    Collect possible access-facing core edges.

    Logic:
    1. Split each core polygon into individual edges.
    2. Exclude edges shorter than min_edge_length to avoid artefacts.
    3. For each core polygon, calculate the distance from each remaining core edge
       to the exterior boundary of the merged footprint.
    4. For each core polygon, keep all edges within tie_tolerance of the minimum
       boundary distance.
    5. Link each kept edge to the nearest exterior façade segment.
    6. For the associated façade direction, calculate the total length of all exterior
       footprint edges with similar direction. This represents the dominant façade side.
    """
    foot_edges = footprint_edges(footprint)
    foot_dirs = [edge_direction_deg(a, b) for (a, b) in foot_edges]
    foot_lens = [edge_length(edge) for edge in foot_edges]
    exterior = footprint.exterior

    all_candidates: List[Dict[str, Any]] = []

    for core_poly in core_polys:
        coords = list(core_poly.exterior.coords)

        core_edges = [
            (np.array(coords[i], dtype=float), np.array(coords[i + 1], dtype=float))
            for i in range(len(coords) - 1)
        ]

        core_candidates: List[Dict[str, Any]] = []

        for p0, p1 in core_edges:
            length = float(np.linalg.norm(p1 - p0))
            if length < min_edge_length:
                continue

            edge_line = LineString([p0, p1])
            distance_to_boundary = float(edge_line.distance(exterior))
            midpoint = (p0 + p1) / 2.0

            facade_idx = riser_to_facade_index(midpoint, foot_edges)
            if facade_idx is None:
                continue

            core_edge_dir = edge_direction_deg(p0, p1)
            facade_edge = foot_edges[facade_idx]
            facade_dir = foot_dirs[facade_idx]
            facade_len = foot_lens[facade_idx]
            parallel_diff = angular_diff(core_edge_dir, facade_dir)

            dom_len = direction_total_length(
                direction_deg=facade_dir,
                foot_dirs=foot_dirs,
                foot_lens=foot_lens,
                tol=tol,
            )

            core_candidates.append({
                "poly": core_poly,
                "riser": (p0, p1),
                "mid": midpoint,
                "edge_length": length,
                "distance_to_boundary": distance_to_boundary,
                "facade_idx": facade_idx,
                "facade_edge": facade_edge,
                "facade_dir": facade_dir,
                "facade_len": facade_len,
                "parallel_diff": parallel_diff,
                "direction_total_length": dom_len,
            })

        if not core_candidates:
            continue

        min_distance = min(c["distance_to_boundary"] for c in core_candidates)

        close_candidates = [
            c for c in core_candidates
            if c["distance_to_boundary"] <= min_distance + tie_tolerance
        ]

        all_candidates.extend(close_candidates)

    return all_candidates


def select_orientation_candidate(candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Select the final orientation reference.

    Ranking:
    1. Candidate associated with the dominant façade direction
       (largest total length of footprint exterior edges with similar orientation).
    2. Candidate whose core edge is closest to the exterior boundary.
    3. Longer core edge.
    4. Core edge most parallel to the associated façade edge.
    """
    if not candidates:
        return None

    candidates.sort(
        key=lambda c: (
            -c["direction_total_length"],
            c["distance_to_boundary"],
            -c["edge_length"],
            c["parallel_diff"],
        )
    )

    return candidates[0]


def find_similar_facade_edges_by_direction(
    direction_deg: float,
    footprint: Polygon,
    tolerance_deg: float = 10.0,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return footprint exterior edges with direction similar to the chosen façade direction."""
    similar_edges = []

    for edge in footprint_edges(footprint):
        edge_dir = edge_direction_deg(edge[0], edge[1])
        if angular_diff(edge_dir, direction_deg) <= tolerance_deg:
            similar_edges.append(edge)

    return similar_edges


# ============================================================
#                    ORIENTATION COMPUTATION
# ============================================================

def compute_orientation_result(
    footprint: Polygon,
    apartment_polys: List[Polygon],
    core_polys: List[Polygon],
    all_polys: List[Polygon],
    tol: float = 10.0,
    min_edge_length: float = 0.30,
    tie_tolerance: float = 0.20,
) -> OrientationResult:
    """
    Compute all intermediate objects for the 4-panel workflow figure.

    Logic:
    1. Select the core edge closest to the outer footprint boundary.
    2. If multiple nearby core edges exist, select the one linked to the dominant
       exterior façade direction.
    3. Rotate based on the associated façade direction.
    4. Flip by 180 degrees if the selected core polygon lies above the footprint centre.
    """
    if not core_polys:
        raise ValueError("No stair/core polygons found; orientation cannot be visualised.")

    candidates = collect_core_outer_edge_candidates(
        core_polys=core_polys,
        footprint=footprint,
        min_edge_length=min_edge_length,
        tie_tolerance=tie_tolerance,
        tol=tol,
    )

    chosen_info = select_orientation_candidate(candidates)

    if chosen_info is None:
        raise ValueError("Stair/core polygons were found, but no valid outer-side edge was detected.")

    chosen_dir = chosen_info["facade_dir"]
    rotation_needed = -chosen_dir

    # Rotate by the selected exterior façade direction, not by the core edge itself.
    oriented_fp = rotate(
        footprint,
        rotation_needed,
        origin="centroid",
        use_radians=False,
    )

    oriented_cores = [
        rotate(core, rotation_needed, origin=footprint.centroid, use_radians=False)
        for core in core_polys
    ]

    # Use the whole selected core polygon position for the flip decision.
    rot_selected_core = rotate(
        chosen_info["poly"],
        rotation_needed,
        origin=footprint.centroid,
        use_radians=False,
    )

    core_y = rot_selected_core.centroid.y
    footprint_center_y = oriented_fp.centroid.y

    flipped = False

    if core_y > footprint_center_y:
        flip_origin = oriented_fp.centroid

        oriented_fp = rotate(
            oriented_fp,
            180,
            origin=flip_origin,
            use_radians=False,
        )

        oriented_cores = [
            rotate(core, 180, origin=flip_origin, use_radians=False)
            for core in oriented_cores
        ]

        flipped = True

    similar_facade_edges = find_similar_facade_edges_by_direction(
        direction_deg=chosen_dir,
        footprint=footprint,
        tolerance_deg=tol,
    )

    return OrientationResult(
        raw_polygons=all_polys,
        apartment_polygons=apartment_polys,
        core_polygons=core_polys,
        merged_footprint=footprint,
        selected_core_polygon=chosen_info["poly"],
        selected_core_edge=chosen_info["riser"],
        selected_core_midpoint=chosen_info["mid"],
        nearest_facade_edge=chosen_info["facade_edge"],
        similar_facade_edges=similar_facade_edges,
        chosen_facade_direction_deg=chosen_dir,
        rotation_angle_deg=rotation_needed,
        flipped_180=flipped,
        core_centroid_y_after_rotation=core_y,
        footprint_centroid_y_after_rotation=footprint_center_y,
        dominant_direction_total_length=chosen_info["direction_total_length"],
        selected_distance_to_boundary=chosen_info["distance_to_boundary"],
        oriented_footprint=oriented_fp,
        oriented_core_polygons=oriented_cores,
    )


def process_case(pickle_path: str) -> OrientationResult:
    data = load_pickle(pickle_path)

    all_polys, apartment_polys, core_polys = extract_polygons_from_floorplan_object(data)
    footprint = create_merged_footprint(all_polys)

    return compute_orientation_result(
        footprint=footprint,
        apartment_polys=apartment_polys,
        core_polys=core_polys,
        all_polys=all_polys,
        tol=10.0,
        min_edge_length=0.30,
        tie_tolerance=0.20,
    )


# ============================================================
#                          PLOTTING
# ============================================================

def plot_polygon(
    ax,
    poly: Polygon,
    linewidth: float = 1.2,
    linestyle: str = "-",
    label: Optional[str] = None,
    alpha: float = 1.0,
    color: str = POLYGON_EDGE_COLOR,
    zorder: int = 3,
):
    x, y = poly.exterior.xy
    ax.plot(
        x,
        y,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        label=label,
        alpha=alpha,
        zorder=zorder,
        solid_capstyle="round",
        solid_joinstyle="round",
    )


def fill_polygon(
    ax,
    poly: Polygon,
    facecolor: str,
    alpha: float = 1.0,
    label: Optional[str] = None,
    zorder: int = 1,
):
    x, y = poly.exterior.xy
    ax.fill(
        x,
        y,
        facecolor=facecolor,
        edgecolor="none",
        alpha=alpha,
        label=label,
        zorder=zorder,
    )


def plot_edge(
    ax,
    edge,
    linewidth: float = 3.0,
    label: Optional[str] = None,
    color: str = CORE_COLOR,
    linestyle: str = "-",
    alpha: float = 1.0,
    zorder: int = 5,
):
    a, b = edge
    ax.plot(
        [a[0], b[0]],
        [a[1], b[1]],
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        label=label,
        alpha=alpha,
        zorder=zorder,
        solid_capstyle="round",
    )

def add_panel_labels_aligned(fig, axes):
    """
    Add panel labels using figure coordinates so labels in the same row
    are horizontally aligned, even if subplot aspect ratios differ.
    """
    fig.canvas.draw()

    positions = [ax.get_position() for ax in axes]

    top_row_y = min(positions[0].y0, positions[1].y0) - 0.055
    bottom_row_y = min(positions[2].y0, positions[3].y0) - 0.045

    labels = ["(a)", "(b)", "(c)", "(d)"]
    y_positions = [top_row_y, top_row_y, bottom_row_y, bottom_row_y]

    for ax, label, y in zip(axes, labels, y_positions):
        pos = ax.get_position()
        x = 0.5 * (pos.x0 + pos.x1)

        fig.text(
            x,
            y,
            label,
            ha="center",
            va="top",
            fontsize=10,
            color=TEXT_COLOR,
        )

def add_midpoint(ax, point_xy, label: Optional[str] = None):
    ax.scatter(
        [point_xy[0]],
        [point_xy[1]],
        s=34,
        marker="o",
        facecolor="white",
        edgecolor=CORE_COLOR,
        linewidth=0.9,
        label="Stairwell edge midpoint",
        zorder=12,
    )

    if label:
        ax.annotate(
            label,
            xy=(point_xy[0], point_xy[1]),
            xytext=(-45, 15),
            textcoords="offset points",
            fontsize=10,
            color=TEXT_COLOR,
            zorder=13,
        )


def thesis_legend(ax, loc: str = "best"):
    legend = ax.legend(
        loc=loc,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        facecolor="white",
        edgecolor=LEGEND_EDGE_COLOR,
        borderpad=0.35,
        handlelength=1.7,
        handletextpad=0.55,
        labelspacing=0.35,
    )

    if legend is not None:
        legend.get_frame().set_linewidth(0.6)

    return legend


def thesis_legend_ordered(ax, first_labels: List[str], loc: str = "best"):
    """Create a thesis-style legend with selected labels moved to the top."""
    handles, labels = ax.get_legend_handles_labels()
    ordered_handles = []
    ordered_labels = []

    for target in first_labels:
        for handle, label in zip(handles, labels):
            if label == target and label not in ordered_labels:
                ordered_handles.append(handle)
                ordered_labels.append(label)

    for handle, label in zip(handles, labels):
        if label not in ordered_labels:
            ordered_handles.append(handle)
            ordered_labels.append(label)

    legend = ax.legend(
        ordered_handles,
        ordered_labels,
        loc=loc,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        facecolor="white",
        edgecolor=LEGEND_EDGE_COLOR,
        borderpad=0.35,
        handlelength=1.7,
        handletextpad=0.55,
        labelspacing=0.35,
    )

    if legend is not None:
        legend.get_frame().set_linewidth(0.6)

    return legend


def add_info_box(ax, text: str):
    ax.text(
        0.02,
        0.96,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        color=TEXT_COLOR,
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            edgecolor=LEGEND_EDGE_COLOR,
            linewidth=0.6,
            alpha=0.95,
        ),
        zorder=10,
    )


def set_axis_format(ax):
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--")

    # Keep numerical tick labels, but remove axis titles for a cleaner thesis figure.
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="both", width=0.6, length=3)

    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color(POLYGON_EDGE_COLOR)


def add_panel_label_below(ax, label: str):
    """Place only the alphabetical panel label below each subplot."""
    ax.text(
        0.5,
        -0.16,
        label,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        color=TEXT_COLOR,
        clip_on=False,
    )


def set_common_limits(axes, polygons: List[Polygon], margin_ratio=0.08):
    minx = min(p.bounds[0] for p in polygons)
    miny = min(p.bounds[1] for p in polygons)
    maxx = max(p.bounds[2] for p in polygons)
    maxy = max(p.bounds[3] for p in polygons)

    dx = maxx - minx
    dy = maxy - miny
    margin = max(dx, dy) * margin_ratio

    for ax in axes:
        ax.set_xlim(minx - margin, maxx + margin)
        ax.set_ylim(miny - margin, maxy + margin)


def create_orientation_figure(result: OrientationResult, output_path: str, title_prefix: str = ""):
    apply_thesis_style()

    fig, axes = plt.subplots(2, 2, figsize=(6.14, 5.5))
    axes = axes.ravel()

    # -------------------- (a) Stored GT polygons --------------------
    ax = axes[0]

    for i, poly in enumerate(result.apartment_polygons):
        fill_polygon(
            ax,
            poly,
            facecolor=dwelling_color(i),
            alpha=1.0,
            label=f"Dwelling {i+1}" if i < len(result.apartment_polygons) else None,
        )
        plot_polygon(ax, poly, linewidth=0.9, color=POLYGON_EDGE_COLOR)

    for i, poly in enumerate(result.core_polygons):
        fill_polygon(
            ax,
            poly,
            facecolor=CORE_COLOR,
            alpha=1.0,
            label="Stairwell" if i == 0 else None,
            zorder=2,
        )
        plot_polygon(ax, poly, linewidth=1.1, color=CORE_EDGE_COLOR, zorder=4)

    ax.set_title("")
    thesis_legend(ax)
    set_axis_format(ax)
    # add_panel_label_below(ax, "(a)")

    # -------------------- (b) Merged footprint --------------------
    ax = axes[1]

    fill_polygon(ax, result.merged_footprint, facecolor=FOOTPRINT_FILL_COLOR, alpha=1.0)
    plot_polygon(
        ax,
        result.merged_footprint,
        linewidth=1.3,
        color=POLYGON_EDGE_COLOR,
        label="Merged footprint",
    )

    for i, poly in enumerate(result.core_polygons):
        plot_polygon(
            ax,
            poly,
            linewidth=1.0,
            linestyle="--",
            color=CORE_COLOR,
            label="Stairwell before orientation" if i == 0 else None,
            zorder=5,
        )

    ax.set_title("")
    thesis_legend(ax, loc="upper left")
    set_axis_format(ax)
    # add_panel_label_below(ax, "(b)")

    # -------------------- (c) Orientation reference --------------------
    ax = axes[2]

    fill_polygon(ax, result.merged_footprint, facecolor=FOOTPRINT_FILL_COLOR, alpha=1.0)
    plot_polygon(ax, result.merged_footprint, linewidth=1.1, color=POLYGON_EDGE_COLOR, label="_nolegend_")

    for i, edge in enumerate(result.similar_facade_edges):
        plot_edge(
            ax,
            edge,
            linewidth=1.8,
            color=DWELLING_COLORS[1],
            linestyle="-",
            alpha=0.75,
            label="_nolegend_" if i == 0 else None,
            zorder=4,
        )

    # The selected dominant façade is the exterior façade segment linked to the
    # selected core edge and used as the orientation reference.
    plot_edge(
        ax,
        result.nearest_facade_edge,
        linewidth=3.2,
        color=POLYGON_EDGE_COLOR,
        linestyle="-",
        label="_nolegend_",  # keep it out of the legend
        zorder=7,
    )

    plot_edge(
        ax,
        result.selected_core_edge,
        linewidth=3.0,
        color=CORE_COLOR,
        linestyle="-",
        label="_nolegend_",  # keep it out of the legend
        zorder=8,
    )

    centroid = np.array(result.merged_footprint.centroid.coords[0], dtype=float)

    ax.scatter(
        [centroid[0]],
        [centroid[1]],
        s=28,
        marker="x",
        color=CORE_COLOR,
        linewidth=0.8,
        label="Footprint centroid",   # keeps it out of the legend
        zorder=20,
    )

    # ax.annotate(
    #     "Footprint centroid",
    #     xy=(centroid[0], centroid[1]),
    #     xytext=(-6, -12),
    #     textcoords="offset points",
    #     fontsize=10,
    #     color=TEXT_COLOR,
    #     zorder=21,
    # )

    add_midpoint(ax, result.selected_core_midpoint)

    # Numerical orientation details are omitted from the panel and can be
    # described in the figure caption or methodology text.

    ax.set_title("")
    thesis_legend(ax, loc="upper left")
    set_axis_format(ax)
    # add_panel_label_below(ax, "(c)")

    # -------------------- (d) Oriented footprint --------------------
    ax = axes[3]

    fill_polygon(ax, result.oriented_footprint, facecolor=FOOTPRINT_FILL_COLOR, alpha=1.0)
    plot_polygon(
        ax,
        result.oriented_footprint,
        linewidth=1.3,
        color=POLYGON_EDGE_COLOR,
        label="Oriented footprint",
    )

    for i, poly in enumerate(result.oriented_core_polygons):
        # fill_polygon(ax, poly, facecolor=CORE_COLOR, alpha=1.0, zorder=2)
        plot_polygon(
            ax,
            poly,
            linewidth=1.0,
            linestyle="--",
            color=CORE_EDGE_COLOR,
            label="Stairwell after orientation" if i == 0 else None,
            zorder=5,
        )

    # fp_centroid = result.oriented_footprint.centroid
    # ax.axhline(
    #     fp_centroid.y,
    #     color=LEGEND_EDGE_COLOR,
    #     linestyle="--",
    #     linewidth=0.9,
    #     alpha=1.0,
    #     label="Footprint centroid y",
    #     zorder=3,
    # )

    # Numerical rotation details are omitted from the panel and can be
    # described in the figure caption or methodology text.

    ax.set_title("")
    thesis_legend(ax, loc="upper right")
    set_axis_format(ax)
    # add_panel_label_below(ax, "(d)")

    set_common_limits(
        axes[:3],
        result.raw_polygons + [result.merged_footprint],
        margin_ratio=0.10,
    )

    minx, miny, maxx, maxy = result.oriented_footprint.bounds
    margin = max(maxx - minx, maxy - miny) * 0.10

    axes[3].set_xlim(minx - margin, maxx + margin)
    axes[3].set_ylim(miny - margin, maxy + margin)

    if title_prefix:
        fig.suptitle(title_prefix, fontsize=10, fontweight="normal", y=0.995)
        fig.tight_layout(rect=[0, 0.02, 1, 0.97], pad=0.8, w_pad=1.0, h_pad=2.0)
    else:
        fig.tight_layout(pad=0.8, w_pad=1.0, h_pad=2.0)

        add_panel_labels_aligned(fig, axes)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print(f"Saved orientation figure to: {output_path}")


# ============================================================
#                            MAIN
# ============================================================

def main():
    # ========================================================
    # CONFIGURATION: edit these only
    # ========================================================

    # If this script is inside:
    # gml_footprint_replacement/figures/scripts/
    PROJECT_DIR = Path(__file__).resolve().parents[1]
    REPO_DIR = PROJECT_DIR.parent

    BUILDING_ID = 75

    # Ground-truth pickle folder
    DATAPATH = REPO_DIR / "msd_ground_truth_extraction" / "data" / "ground_truth"

    # Output folder for this figure
    OUTPUT_DIR = PROJECT_DIR / "figures" / "output" 

    # ========================================================

    pickle_path = DATAPATH / f"{BUILDING_ID}.pickle"
    output_path = OUTPUT_DIR / f"orientation_workflow_{BUILDING_ID}.pdf"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    result = process_case(pickle_path)

    create_orientation_figure(
        result=result,
        output_path=output_path,
        title_prefix="",
    )


if __name__ == "__main__":
    main()
