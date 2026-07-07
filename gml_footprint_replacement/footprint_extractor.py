import os
import json
import math
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from shapely.geometry import Polygon, mapping, LineString, Point, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.affinity import rotate
from shapely.geometry import JOIN_STYLE


# ============================================================
#                     GEOMETRY UTILITIES
# ============================================================

def load_floorplan_pickle(path: str):
    """Load the stored floor-plan pickle."""
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


def get_all_polygons(floorplan: Dict[str, Any]) -> List[Polygon]:
    """Return all valid polygons from the stored floor_plan entries."""
    polys: List[Polygon] = []

    for entry in floorplan.get("floor_plan", []):
        if not isinstance(entry, dict) or "polygon" not in entry:
            continue

        polys.extend(geometry_to_polygons(entry["polygon"]))

    return polys


def get_all_stair_polygons(floorplan: Dict[str, Any]) -> List[Polygon]:
    """
    Return all stair/core polygons based on room_type == 1.

    This handles Polygon, MultiPolygon, coordinate-list, and GeoJSON-like storage.
    """
    stair_polys: List[Polygon] = []

    for entry in floorplan.get("floor_plan", []):
        if not isinstance(entry, dict):
            continue

        if entry.get("room_type") == 1 and "polygon" in entry:
            stair_polys.extend(geometry_to_polygons(entry["polygon"]))

    return stair_polys


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


def riser_to_facade_index(
    riser_midpoint: np.ndarray,
    foot_edges: List[Tuple[np.ndarray, np.ndarray]],
) -> Optional[int]:
    """
    Return the index of the exterior footprint edge closest to the selected core-edge midpoint.
    """
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
    direction: float,
    foot_dirs: List[float],
    foot_lens: List[float],
    tol: float,
) -> float:
    """
    Sum the total footprint-edge length belonging to a similar façade direction.
    This represents how dominant a façade direction is in the footprint.
    """
    total = 0.0

    for d_edge, L_edge in zip(foot_dirs, foot_lens):
        if angular_diff(d_edge, direction) <= tol:
            total += L_edge

    return total


# ============================================================
#       CORE EDGE CLOSEST TO OUTER FOOTPRINT BOUNDARY
# ============================================================

def collect_core_outer_edge_candidates(
    stair_polys: List[Polygon],
    footprint: Polygon,
    min_edge_length: float = 0.30,
    tie_tolerance: float = 0.20,
    tol: float = 10.0,
) -> List[Dict[str, Any]]:
    """
    Collect possible access-facing core edges.

    Logic:
    1. Split each core polygon into edges.
    2. Remove very short edges.
    3. Calculate each core edge's distance to the exterior boundary of the merged footprint.
    4. Keep edges whose distance is within tie_tolerance of the minimum distance
       for that core polygon.
    5. Link each kept core edge to the nearest exterior façade segment.
    6. Store the total length of footprint edges with similar façade direction
       to identify the dominant façade side.
    """
    foot_edges = footprint_edges(footprint)
    foot_dirs = [edge_direction_deg(a, b) for (a, b) in foot_edges]
    foot_lens = [edge_length(edge) for edge in foot_edges]
    exterior = footprint.exterior

    all_candidates: List[Dict[str, Any]] = []

    for stair_poly in stair_polys:
        coords = list(stair_poly.exterior.coords)

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
                direction=facade_dir,
                foot_dirs=foot_dirs,
                foot_lens=foot_lens,
                tol=tol,
            )

            core_candidates.append({
                "poly": stair_poly,
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


def select_orientation_candidate(
    candidates: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Select the final orientation reference.

    Ranking:
    1. Candidate associated with the dominant façade direction
       (largest total length of footprint edges with similar orientation).
    2. Closest core edge to the footprint exterior.
    3. Longest core edge.
    4. Most parallel core edge to the associated façade.
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


# ============================================================
#                    ORIENTATION LOGIC
# ============================================================

def orient_footprint_by_stairs(
    footprint: Polygon,
    stair_polys: List[Polygon],
    tol: float = 10.0,
    min_edge_length: float = 0.30,
    tie_tolerance: float = 0.20,
) -> Polygon:
    """
    Orient the footprint using the stair/core geometry as preprocessing reference.

    Logic:
    1. Identify core edges located closest to the outer footprint boundary.
    2. If more than one edge is close, select the one associated with the dominant
       exterior façade direction.
    3. Rotate the footprint based on the selected exterior façade direction, so the
       access-facing façade becomes parallel to the x-axis.
    4. Flip by 180 degrees if the selected core polygon lies above the footprint centre.

    The stair/core geometry is used only for preprocessing orientation.
    It is not passed to the MZA as an internal thermal zone.
    """
    if not stair_polys:
        print("No stairs/core polygons found; orientation skipped.")
        return footprint

    candidates = collect_core_outer_edge_candidates(
        stair_polys=stair_polys,
        footprint=footprint,
        min_edge_length=min_edge_length,
        tie_tolerance=tie_tolerance,
        tol=tol,
    )

    chosen_info = select_orientation_candidate(candidates)

    if chosen_info is None:
        print("Stairs/core polygons found, but no valid outer-side edge was detected; orientation skipped.")
        return footprint

    chosen_dir = chosen_info["facade_dir"]

    print(
        f"Selected core edge closest to outer boundary; "
        f"associated facade direction ~{chosen_dir:.1f} deg."
    )
    print(
        f"   distance to boundary = {chosen_info['distance_to_boundary']:.3f} m, "
        f"dominant-direction length = {chosen_info['direction_total_length']:.3f} m"
    )

    rotation_needed = -chosen_dir

    print(f"Rotating footprint based on facade direction: {rotation_needed:.2f} deg")

    rot_fp = rotate(
        footprint,
        rotation_needed,
        origin="centroid",
        use_radians=False,
    )

    rot_core = rotate(
        chosen_info["poly"],
        rotation_needed,
        origin=footprint.centroid,
        use_radians=False,
    )

    core_y = rot_core.centroid.y
    footprint_center_y = rot_fp.centroid.y

    print(f"   core centroid y = {core_y:.2f}")
    print(f"   footprint centroid y = {footprint_center_y:.2f}")

    if core_y > footprint_center_y:
        print("Core polygon above footprint centre; flipping 180 deg")

        rot_fp = rotate(
            rot_fp,
            180,
            origin="centroid",
            use_radians=False,
        )

    return rot_fp


# ============================================================
#               MAIN FOOTPRINT EXTRACTION
# ============================================================

def extract_footprint(building_id: int, datapath: str) -> Polygon:
    """
    Load a stored floor-plan pickle and generate an oriented external footprint.

    Expected pickle format:
        floorplan["floor_plan"] = [
            {"room_type": 0, "polygon": ...},  # apartment/dwelling
            {"room_type": 1, "polygon": ...},  # stair/core
        ]

    room_type 0 = apartment/dwelling
    room_type 1 = stair/core
    """
    graph_path = os.path.join(datapath, f"{building_id}.pickle")

    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Missing: {graph_path}")

    fp_data = load_floorplan_pickle(graph_path)

    polys = get_all_polygons(fp_data)

    if not polys:
        raise ValueError(f"No valid polygons found in {graph_path}")

    merged = unary_union(polys)

    cleaned = (
        merged
        .buffer(0.5, join_style=JOIN_STYLE.mitre)
        .buffer(-0.4, join_style=JOIN_STYLE.mitre)
    )

    footprint = largest_polygon(cleaned)
    footprint = footprint.simplify(0.05)

    if not footprint.is_valid:
        footprint = footprint.buffer(0)

    stair_polys = get_all_stair_polygons(fp_data)

    footprint = orient_footprint_by_stairs(
        footprint=footprint,
        stair_polys=stair_polys,
        tol=10.0,
        min_edge_length=0.30,
        tie_tolerance=0.20,
    )

    return footprint


# ============================================================
#                      GEOJSON EXPORT
# ============================================================

def create_footprint_geojson(
    building_id: int,
    datapath: str,
    output_geojson: str,
    show_plot: bool = False,
) -> str:
    """
    Create an oriented external footprint and save it as GeoJSON.
    """
    fp = extract_footprint(building_id, datapath)

    geo = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {
                "building_id": building_id,
                "description": "Oriented external footprint derived from apartment and stair/core polygons"
            },
            "geometry": mapping(fp)
        }]
    }

    os.makedirs(os.path.dirname(output_geojson), exist_ok=True)

    with open(output_geojson, "w", encoding="utf-8") as f:
        json.dump(geo, f, indent=2)

    print(f"Saved GeoJSON: {output_geojson}")

    if not show_plot:
        return output_geojson

    try:
        import matplotlib.pyplot as plt

        x, y = fp.exterior.xy
        plt.figure(figsize=(7, 7))
        plt.plot(x, y, "-o", markersize=3)
        plt.title(f"Footprint Preview: Building {building_id}")
        plt.axis("equal")
        plt.grid(True)
        plt.show()

    except Exception as e:
        print("Could not plot footprint:", e)

    return output_geojson


# ============================================================
#                       OPTIONAL TEST
# ============================================================

if __name__ == "__main__":
    # Edit these paths for a quick standalone test.
    DATAPATH = r"C:\WF\Thomas Sharon\Floorplan_Dataset\rd_pickle"
    BUILDING_ID = 3
    OUTPUT_GEOJSON = r"C:\WF\Thomas Sharon\Floorplan_Dataset\gml_msd_trial\footprint\footprint_75.geojson"

    create_footprint_geojson(
        building_id=BUILDING_ID,
        datapath=DATAPATH,
        output_geojson=OUTPUT_GEOJSON,
        show_plot=True,
    )
