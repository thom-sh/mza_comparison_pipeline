import os
import json
import math
import pickle
import numpy as np
from shapely.geometry import Polygon, mapping, LineString, Point
from shapely.ops import unary_union
from shapely.affinity import rotate
from shapely.geometry import JOIN_STYLE


# ============================================================
#                     GEOMETRY UTILITIES
# ============================================================

def load_floorplan_pickle(path):
    """Load your custom pickle format."""
    with open(path, "rb") as f:
        return pickle.load(f)


def get_all_stair_polygons(floorplan):
    """Return list of stair polygons based on room_type == 1."""
    stair_polys = []
    for entry in floorplan["floor_plan"]:
        if entry["room_type"] == 1:      # stair
            poly = Polygon(entry["polygon"])
            if poly.is_valid and not poly.is_empty:
                stair_polys.append(poly)
    return stair_polys


def footprint_edges(footprint):
    coords = list(footprint.exterior.coords)
    edges = []
    for i in range(len(coords) - 1):
        e = (np.array(coords[i]), np.array(coords[i + 1]))
        edges.append(e)
    return edges


def edge_length(edge):
    a, b = edge
    return np.linalg.norm(b - a)


def angle_of_edge(p0, p1):
    dx, dy = p1 - p0
    return math.degrees(math.atan2(dy, dx))


def edge_direction_deg(p0, p1):
    dx, dy = p1 - p0
    ang = math.degrees(math.atan2(dy, dx)) % 180.0
    return ang


def angular_diff(a, b):
    d = abs(a - b)
    return min(d, 180.0 - d)


def riser_to_facade_index(riser_midpoint, foot_edges):
    closest_idx = None
    closest_dist = 1e18
    mid_pt = Point(riser_midpoint)

    for idx, (a, b) in enumerate(foot_edges):
        dist = LineString([a, b]).distance(mid_pt)
        if dist < closest_dist:
            closest_dist = dist
            closest_idx = idx
    return closest_idx


def find_best_outward_edge(stair_poly, footprint):
    centroid = np.array(footprint.centroid.coords[0])
    exterior = footprint.exterior

    coords = list(stair_poly.exterior.coords)
    edges = [
        (np.array(coords[i]), np.array(coords[i+1]))
        for i in range(len(coords)-1)
    ]

    best_edge = None
    best_score = -1e18

    for p0, p1 in edges:
        mid = (p0 + p1) / 2
        mid_pt = Point(mid)

        d_centroid = np.linalg.norm(mid - centroid)
        d_boundary = mid_pt.distance(exterior)

        score = d_centroid - d_boundary
        if score > best_score:
            best_score = score
            best_edge = (p0, p1)

    return best_edge


# ============================================================
#    ORIENTATION LOGIC (UNCHANGED, BUT USING NEW FORMAT)
# ============================================================

# def orient_footprint_by_stairs(footprint, stair_polys, tol=10.0):

#     if not stair_polys:
#         print("⚠️ No stairs found — orientation skipped.")
#         return footprint

#     foot_edges = footprint_edges(footprint)
#     foot_dirs  = [edge_direction_deg(a, b) for (a, b) in foot_edges]
#     foot_lens  = [edge_length(e) for e in foot_edges]

#     stair_infos = []
#     for stair_poly in stair_polys:
#         best_edge = find_best_outward_edge(stair_poly, footprint)
#         if best_edge is None:
#             continue

#         p0, p1 = best_edge
#         mid = (p0 + p1) / 2
#         facade_idx = riser_to_facade_index(mid, foot_edges)
#         facade_dir = foot_dirs[facade_idx]

#         stair_infos.append({
#             "riser": (p0, p1),
#             "mid": mid,
#             "facade_idx": facade_idx,
#             "facade_dir": facade_dir,
#         })

#     if not stair_infos:
#         print("⚠️ Stairs found but no valid risers — orientation skipped.")
#         return footprint

#     # cluster directions
#     unique_dirs = []
#     for info in stair_infos:
#         d = info["facade_dir"]
#         if not any(angular_diff(d, u) < tol for u in unique_dirs):
#             unique_dirs.append(d)

#     if len(unique_dirs) == 1:
#         chosen_dir = unique_dirs[0]
#         chosen_riser = stair_infos[0]["riser"]
#         print(f"🎯 All stairs aligned at ~{chosen_dir:.1f}°.")

#     else:
#         print("🔍 Stairs on different façades → selecting dominant façade")

#         dir_to_len = {}
#         for d0 in unique_dirs:
#             total = 0.0
#             for (d_edge, L) in zip(foot_dirs, foot_lens):
#                 if angular_diff(d_edge, d0) < tol:
#                     total += L
#             dir_to_len[d0] = total

#         chosen_dir = max(dir_to_len, key=dir_to_len.get)
#         print(f"🏗 Dominant façade = {chosen_dir:.1f}°")

#         # pick riser aligned with dominant direction
#         best_info = None
#         best_diff = 1e18
#         for info in stair_infos:
#             diff = angular_diff(info["facade_dir"], chosen_dir)
#             if diff < best_diff:
#                 best_diff = diff
#                 best_info = info

#         chosen_riser = best_info["riser"]

#     # rotate chosen riser parallel to x-axis
#     p0, p1 = chosen_riser
#     angle = angle_of_edge(p0, p1)
#     rotation_needed = -angle
#     print(f"🔄 Rotating footprint {rotation_needed:.2f}°")

#     rot_fp = rotate(footprint, rotation_needed, origin='centroid', use_radians=False)

#     # ensure riser is at bottom
#     line_rot = rotate(LineString([p0, p1]), rotation_needed, origin=footprint.centroid, use_radians=False)
#     mid_rot = line_rot.centroid

#     minx, miny, maxx, maxy = rot_fp.bounds
#     center_y = 0.5 * (miny + maxy)

#     if mid_rot.y > center_y:
#         print("↻ Stair façade above center → flipping 180°")
#         rot_fp = rotate(rot_fp, 180, origin='centroid', use_radians=False)

#     return rot_fp

def orient_footprint_by_stairs(footprint, stair_polys, tol=10.0):

    if not stair_polys:
        print("⚠️ No stairs found — orientation skipped.")
        return footprint

    foot_edges = footprint_edges(footprint)
    foot_dirs  = [edge_direction_deg(a, b) for (a, b) in foot_edges]
    foot_lens  = [edge_length(e) for e in foot_edges]

    stair_infos = []

    # 1. Identify outward-facing stair/core edge
    # 2. Link it to the nearest exterior façade edge
    for stair_poly in stair_polys:
        best_edge = find_best_outward_edge(stair_poly, footprint)
        if best_edge is None:
            continue

        p0, p1 = best_edge
        mid = (p0 + p1) / 2

        facade_idx = riser_to_facade_index(mid, foot_edges)
        facade_dir = foot_dirs[facade_idx]

        stair_infos.append({
            "riser": (p0, p1),
            "mid": mid,
            "facade_idx": facade_idx,
            "facade_dir": facade_dir,
        })

    if not stair_infos:
        print("⚠️ Stairs found but no valid risers — orientation skipped.")
        return footprint

    # Cluster façade directions associated with stair/core edges
    unique_dirs = []
    for info in stair_infos:
        d = info["facade_dir"]
        if not any(angular_diff(d, u) < tol for u in unique_dirs):
            unique_dirs.append(d)

    # Select the façade direction to use for rotation
    if len(unique_dirs) == 1:
        chosen_dir = unique_dirs[0]
        chosen_info = stair_infos[0]
        print(f"🎯 Stair/core side associated with façade direction ~{chosen_dir:.1f}°.")

    else:
        print("🔍 Stairs/cores associated with different façades → selecting dominant façade")

        dir_to_len = {}
        for d0 in unique_dirs:
            total = 0.0
            for d_edge, L in zip(foot_dirs, foot_lens):
                if angular_diff(d_edge, d0) < tol:
                    total += L
            dir_to_len[d0] = total

        chosen_dir = max(dir_to_len, key=dir_to_len.get)
        print(f"🏗 Dominant façade direction = {chosen_dir:.1f}°")

        chosen_info = min(
            stair_infos,
            key=lambda info: angular_diff(info["facade_dir"], chosen_dir)
        )

    # ------------------------------------------------------------
    # IMPORTANT CHANGE:
    # Rotate based on the chosen façade direction, not the stair/core edge
    # ------------------------------------------------------------
    rotation_needed = -chosen_dir
    print(f"🔄 Rotating footprint based on façade direction: {rotation_needed:.2f}°")

    rot_fp = rotate(
        footprint,
        rotation_needed,
        origin="centroid",
        use_radians=False
    )

    # Rotate the selected stair/core midpoint for side-position check
    mid = chosen_info["mid"]
    mid_rot = rotate(
        Point(mid),
        rotation_needed,
        origin=footprint.centroid,
        use_radians=False
    )

    # Ensure the stair/core side is consistently placed at the bottom
    minx, miny, maxx, maxy = rot_fp.bounds
    center_y = 0.5 * (miny + maxy)

    if mid_rot.y > center_y:
        print("↻ Stair/core façade above center → flipping 180°")
        rot_fp = rotate(
            rot_fp,
            180,
            origin="centroid",
            use_radians=False
        )

    return rot_fp

# ============================================================
#               MAIN FOOTPRINT EXTRACTION
# ============================================================

def extract_footprint(building_id, datapath):
    """
    Loads your new-style pickle and generates an oriented footprint.
    """
    graph_path = os.path.join(datapath, f"{building_id}.pickle")
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"❌ Missing: {graph_path}")

    fp_data = load_floorplan_pickle(graph_path)

    # collect ALL polygons (both stair + apartment)
    polys = []
    for entry in fp_data["floor_plan"]:
        poly = Polygon(entry["polygon"])
        if poly.is_valid and not poly.is_empty:
            polys.append(poly)

    merged = unary_union(polys).buffer(0.5, join_style=JOIN_STYLE.mitre).buffer(-0.4, join_style=JOIN_STYLE.mitre)
    if merged.geom_type == "MultiPolygon":
        merged = max(merged.geoms, key=lambda g: g.area)

    footprint = merged.simplify(0.05)

    stair_polys = get_all_stair_polygons(fp_data)
    footprint = orient_footprint_by_stairs(footprint, stair_polys)

    return footprint


# ============================================================
#                      GEOJSON EXPORT
# ============================================================

def create_footprint_geojson(building_id, datapath, output_geojson):
    fp = extract_footprint(building_id, datapath)

    geo = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {
                "building_id": building_id,
                "description": "Oriented footprint (new pickle format)"
            },
            "geometry": mapping(fp)
        }]
    }

    os.makedirs(os.path.dirname(output_geojson), exist_ok=True)
    with open(output_geojson, "w") as f:
        json.dump(geo, f, indent=2)

    print(f"📁 Saved GeoJSON: {output_geojson}")

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
        print("⚠️ Could not plot footprint:", e)

    return output_geojson
