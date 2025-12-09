import os
import json
import math
import numpy as np
from shapely.geometry import Polygon, mapping, LineString, Point
from shapely.ops import unary_union
from shapely.affinity import rotate

# Add project root to Python path for utils import
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import load_pickle
from constants import ROOM_NAMES   # ROOM_NAMES defines the room_type indices


# ============================================================
#              GEOMETRY UTILITIES
# ============================================================

def get_all_stair_polygons(G):
    """Return list of stair polygons based on room_type == 'Stairs'."""
    STAIR_TYPE = ROOM_NAMES.index("Stairs")
    stair_polys = []
    for _, d in G.nodes(data=True):
        if d.get("room_type") == STAIR_TYPE:
            geom = d.get("geometry")
            if geom:
                try:
                    poly = Polygon(geom)
                    if poly.is_valid and not poly.is_empty:
                        stair_polys.append(poly)
                except Exception:
                    pass
    return stair_polys


def footprint_edges(footprint):
    """Return list of edges (as (p0, p1) arrays) for the footprint exterior."""
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
    """Direction of edge in [0, 180) degrees (ignores sign/orientation)."""
    dx, dy = p1 - p0
    ang = math.degrees(math.atan2(dy, dx)) % 180.0
    return ang


def angular_diff(a, b):
    """Smallest difference between two directions in [0, 180)."""
    d = abs(a - b)
    return min(d, 180.0 - d)


def riser_to_facade_index(riser_midpoint, foot_edges):
    """Return index of the façade (edge) closest to the riser midpoint."""
    closest_idx = None
    closest_dist = 1e18
    mid_pt = Point(riser_midpoint)

    for idx, (a, b) in enumerate(foot_edges):
        line = LineString([a, b])
        dist = line.distance(mid_pt)
        if dist < closest_dist:
            closest_dist = dist
            closest_idx = idx
    return closest_idx


def find_best_outward_edge(stair_poly, footprint):
    """
    Identify the true outward-facing riser edge of a stair polygon.
    Logic:
      1. For each stair edge, shoot rays along its two normals.
      2. If either ray exits the footprint, edge is 'outward-facing'.
      3. Among outward edges, pick the longest one.
      4. If tie: pick the edge whose direction is most aligned with the
         nearest facade direction.
    """
    from shapely.geometry import LineString, Point
    import numpy as np
    import math

    # get stair edges
    coords = list(stair_poly.exterior.coords)
    edges = [(np.array(coords[i]), np.array(coords[i+1]))
             for i in range(len(coords) - 1)]

    outward_edges = []

    # loop edges
    for p0, p1 in edges:
        mid = (p0 + p1) / 2
        mid_pt = Point(mid)

        # edge direction
        v = p1 - p0
        # two normals
        n1 = np.array([-v[1],  v[0]])
        n2 = np.array([ v[1], -v[0]])

        normals = [n1, n2]
        is_outward = False

        for n in normals:
            if np.linalg.norm(n) < 1e-9:
                continue
            n_dir = n / np.linalg.norm(n)

            # shoot ray outward 3 meters
            ray_end = mid + n_dir * 3.0
            ray = LineString([mid, ray_end])

            # outward if ray does NOT lie fully inside building
            if not footprint.contains(ray):
                is_outward = True
                break

        if is_outward:
            length = np.linalg.norm(p1 - p0)
            outward_edges.append(((p0, p1), length))

    if not outward_edges:
        return None  # stairs fully internal? rare but possible

    # STEP 3: pick longest outward edge
    outward_edges.sort(key=lambda x: -x[1])  # descending length
    best_edge, best_len = outward_edges[0]

    return best_edge


# ============================================================
#   ORIENTATION LOGIC (ALL YOUR BUILDING RULES)
# ============================================================

def orient_footprint_by_stairs(footprint, stair_polys,
                               dir_tolerance_deg=10.0):

    if not stair_polys:
        print("⚠️ No stairs found — skipping orientation.")
        return footprint

    # All footprint façade edges
    foot_edges = footprint_edges(footprint)
    foot_dirs  = [edge_direction_deg(a, b) for (a, b) in foot_edges]
    foot_lens  = [edge_length(e) for e in foot_edges]

    # Step 1: For each stair polygon -> outward-facing edge & its façade
    stair_infos = []
    for stair_poly in stair_polys:
        best_edge = find_best_outward_edge(stair_poly, footprint)
        if best_edge is None:
            continue

        p0, p1 = best_edge
        midpoint = (p0 + p1) / 2

        facade_idx = riser_to_facade_index(midpoint, foot_edges)
        facade_dir = foot_dirs[facade_idx]

        stair_infos.append({
            "riser": (p0, p1),
            "mid": midpoint,
            "facade_idx": facade_idx,
            "facade_dir": facade_dir,
        })

    if not stair_infos:
        print("⚠️ Could not derive any stair riser info — skipping orientation.")
        return footprint

    # Step 2: Choose dominant façade direction
    unique_dirs = []
    for info in stair_infos:
        d = info["facade_dir"]
        if not any(angular_diff(d, ud) < dir_tolerance_deg for ud in unique_dirs):
            unique_dirs.append(d)

    if len(unique_dirs) == 1:
        chosen_riser = stair_infos[0]["riser"]
        facade_dir_chosen = unique_dirs[0]
    else:
        # sum façade lengths for clustering
        dir_to_total_len = {}
        for stair_dir in unique_dirs:
            total = 0.0
            for (d_edge, L_edge) in zip(foot_dirs, foot_lens):
                if angular_diff(d_edge, stair_dir) < dir_tolerance_deg:
                    total += L_edge
            dir_to_total_len[stair_dir] = total

        facade_dir_chosen = max(dir_to_total_len, key=dir_to_total_len.get)

        best_info = min(
            stair_infos,
            key=lambda info: angular_diff(info["facade_dir"], facade_dir_chosen)
        )
        chosen_riser = best_info["riser"]

    # Debug before rotation
    # plot_orientation_debug(
    #     footprint,
    #     stair_polys,
    #     foot_edges,
    #     stair_infos,
    #     chosen_riser=chosen_riser,
    #     title="Before Rotation: Stair Orientation Debug"
    # )

    # ----------------------------------------------------
    # 🔥 CORRECTED GEOMETRIC ROTATION + OUTWARD NORMAL LOGIC
    # ----------------------------------------------------

    p0, p1 = chosen_riser

    # 1) Compute rotation to make riser horizontal
    v = p1 - p0
    angle = math.degrees(math.atan2(v[1], v[0]))
    rotation_needed = -angle

    # 2) Rotate footprint
    rot_fp = rotate(footprint, rotation_needed,
                    origin='centroid', use_radians=False)

    # 3) Rotate riser endpoints
    p0_rot = np.array(rotate(Point(p0), rotation_needed,
                             origin=footprint.centroid,
                             use_radians=False).coords[0])
    p1_rot = np.array(rotate(Point(p1), rotation_needed,
                             origin=footprint.centroid,
                             use_radians=False).coords[0])

    # 4) Compute normals in *rotated* space
    v_rot = p1_rot - p0_rot
    n1 = np.array([-v_rot[1],  v_rot[0]])
    n2 = np.array([ v_rot[1], -v_rot[0]])

    # Normalize
    normals = []
    for n in (n1, n2):
        if np.linalg.norm(n) > 1e-9:
            normals.append(n / np.linalg.norm(n))

    # 5) Determine outward normal using ray test on *rotated* footprint
    midpoint_rot = (p0_rot + p1_rot) / 2
    midpoint_rot_pt = Point(midpoint_rot)
    outward_normal = None

    for n in normals:
        ray_end = midpoint_rot + n * 3.0  # 3m ray
        ray = LineString([midpoint_rot, ray_end])
        if not rot_fp.contains(ray):
            outward_normal = n
            break

    if outward_normal is None:
        print("⚠️ Warning: Could not determine outward normal. Skipping flip.")
        final_fp = rot_fp
    else:
        # 6) Flip only if outward normal points upward (positive y)
        if outward_normal[1] > 0:
            final_fp = rotate(rot_fp, 180,
                              origin='centroid', use_radians=False)
        else:
            final_fp = rot_fp

    # Debug after rotation
    # plot_orientation_debug(
    #     final_fp,
    #     stair_polys,
    #     footprint_edges(final_fp),
    #     stair_infos,
    #     chosen_riser=chosen_riser,
    #     title="After Rotation: Stair Orientation Debug"
    # )

    return final_fp

import matplotlib.pyplot as plt
from shapely.geometry import LineString
import numpy as np

def plot_orientation_debug(
    footprint,
    stair_polys,
    foot_edges,
    stair_infos,
    chosen_riser=None,
    title="Footprint Orientation Debug"
):
    plt.figure(figsize=(10, 10))

    # -------------------------------
    # Plot footprint outline
    # -------------------------------
    fx, fy = footprint.exterior.xy
    plt.plot(fx, fy, 'k-', linewidth=2, label="Footprint")

    # -------------------------------
    # Plot stair polygons
    # -------------------------------
    for sp in stair_polys:
        sx, sy = sp.exterior.xy
        plt.plot(sx, sy, 'gray', linewidth=1.5, label="Stairs")

    # -------------------------------
    # Plot stair riser edges + midpoints
    # -------------------------------
    for info in stair_infos:
        p0, p1 = info["riser"]
        mid = info["mid"]

        # Riser line
        plt.plot([p0[0], p1[0]], [p0[1], p1[1]], 'r-', linewidth=3)

        # Midpoint
        plt.scatter(mid[0], mid[1], color='red', s=40)

    # -------------------------------
    # Plot façade edges (green)
    # -------------------------------
    for (a, b) in foot_edges:
        plt.plot([a[0], b[0]], [a[1], b[1]], color='green', alpha=0.5)

    # -------------------------------
    # Highlight chosen riser (yellow)
    # -------------------------------
    if chosen_riser is not None:
        p0, p1 = chosen_riser
        plt.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            color='yellow',
            linewidth=5,
            label="Chosen Riser Edge"
        )

    # -------------------------------
    # Pretty plot styling
    # -------------------------------
    plt.title(title)
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc="upper right")
    plt.show()


# ============================================================
#             FOOTPRINT EXTRACTION + ORIENTATION
# ============================================================

def extract_footprint(building_id, datapath):
    """
    Extract a unified building footprint polygon from the Swiss dataset graph_out,
    and orient it so that the outer-facing stair side lies along the x-axis and
    appears on the bottom part of the building in the plot.

    Returns:
        shapely Polygon (oriented footprint)
    """
    graph_path = os.path.join(datapath, "graph_out", f"{building_id}.pickle")
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"❌ File not found: {graph_path}")

    G = load_pickle(graph_path)

    # Collect polygons (skip balconies by room_type index)
    BALCONY_TYPE = ROOM_NAMES.index("Balcony")
    room_polys = []
    for _, d in G.nodes(data=True):
        if d.get("room_type") == BALCONY_TYPE:
            continue
        geom = d.get("geometry")
        if geom:
            try:
                poly = Polygon(geom)
                if poly.is_valid and not poly.is_empty:
                    room_polys.append(poly)
            except Exception:
                pass

    if not room_polys:
        raise ValueError(f"❌ No valid polygons found for building ID {building_id}")

    # Merge → smooth
    merged = unary_union(room_polys)
    footprint = merged.buffer(0.5).buffer(-0.4)
    if footprint.geom_type == "MultiPolygon":
        footprint = max(footprint.geoms, key=lambda g: g.area)
    footprint = footprint.simplify(0.05, preserve_topology=True)

    # Orientation by stairs
    stair_polys = get_all_stair_polygons(G)
    footprint = orient_footprint_by_stairs(footprint, stair_polys)

    print(f"✅ Final oriented footprint for ID {building_id}.")
    return footprint


# ============================================================
#                     GEOJSON EXPORT
# ============================================================

def create_footprint_geojson(building_id, datapath, output_geojson):
    """
    Extracts a building footprint (with stair-based orientation) and saves it as a GeoJSON file.
    """
    footprint = extract_footprint(building_id, datapath)

    geojson_dict = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "building_id": building_id,
                    "source": "graph_out",
                    "description": "Stair-oriented building footprint",
                },
                "geometry": mapping(footprint),
            }
        ],
    }

    os.makedirs(os.path.dirname(output_geojson), exist_ok=True)
    with open(output_geojson, "w", encoding="utf-8") as f:
        json.dump(geojson_dict, f, indent=2)

    print(f"📁 Saved oriented footprint GeoJSON for building {building_id}:\n{output_geojson}")

    # Optional: quick visual preview of the footprint
    try:
        import matplotlib.pyplot as plt
        x, y = footprint.exterior.xy
        plt.figure(figsize=(6, 6))
        plt.plot(x, y, "-o", markersize=3)
        plt.title(f"Footprint Preview: Building {building_id}")
        plt.axis("equal")
        plt.show()
    except Exception as e:
        print("Plotting failed:", e)

    return output_geojson
