# ===============================================================
#   DATASET LOADERS FOR SWISS GT + PREDICTED BUILDING FOOTPRINTS
# ===============================================================

import pickle
import networkx as nx
import numpy as np

from shapely.geometry import Polygon
from shapely.ops import unary_union

from utils import load_pickle       # you already have this
from constants1 import ROOM_NAMES   # list of all room type names


# ---------------------------------------------------------------
# Utility: return largest polygon (handles MultiPolygon)
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
# LOAD GROUND TRUTH (SWISS) FOOTPRINT
# ---------------------------------------------------------------
def load_gt_rooms_and_footprint(gt_path):
    """
    Load Swiss GT graph, remove balconies, extract room polygons,
    and compute a cleaned footprint.
    Returns:
        footprint (Polygon)
        rooms (list[Polygon])
        room_types (list[int])
    """
    G = load_pickle(gt_path)

    # Map room names → index
    NAME_TO_IDX = {n: i for i, n in enumerate(ROOM_NAMES)}
    BALCONY_IDX = NAME_TO_IDX.get("Balcony", None)

    # Remove balconies
    if BALCONY_IDX is not None:
        balcony_nodes = [
            n for n, d in G.nodes(data=True)
            if d.get("room_type") == BALCONY_IDX
        ]
        G.remove_nodes_from(balcony_nodes)

    rooms = []
    room_types = []

    # Extract room polygons
    for _, d in G.nodes(data=True):
        geom = d.get("geometry")
        if geom is None:
            continue
        try:
            poly = Polygon(geom)
            if poly.is_valid and not poly.is_empty:
                rooms.append(poly)
                room_types.append(d.get("room_type"))
        except:
            pass

    # Merge all rooms → building footprint
    merged = unary_union(rooms)

    # Smooth and clean
    fp = merged.buffer(0.5).buffer(-0.4)
    fp = fp.simplify(0.05, preserve_topology=True)

    fp = largest_poly(fp)

    return fp, rooms, room_types


# ---------------------------------------------------------------
# LOAD PREDICTED ZONES (FROM GML CONVERSION)
# ---------------------------------------------------------------
def load_predicted_zone_polygons(pred_path):
    """
    Load predicted building pickle file and extract zone polygons.
    Returns:
        list of shapely Polygons (one per predicted zone)
    """
    with open(pred_path, "rb") as fh:
        bldg = pickle.load(fh)[0]

    # Select Storey_1 (only one storey is typically present)
    storey1 = next(
        s for s in bldg["polygons"]["storeys"]
        if s["storey_name"] == "Storey_1"
    )

    zone_polys = []
    for zone in storey1["zones"]:
        polys = []

        # floors = list of tuples: (Polygon, height)
        for f in zone.get("floors", []):
            if isinstance(f, tuple) and isinstance(f[0], Polygon):
                polys.append(f[0])

        if polys:
            zone_polys.append(unary_union(polys))

    return zone_polys


# ---------------------------------------------------------------
# COMPUTE PREDICTED FOOTPRINT
# ---------------------------------------------------------------
def compute_predicted_footprint(pred_path):
    """
    Compute building footprint as union of all predicted zones.
    """
    zones = load_predicted_zone_polygons(pred_path)
    if len(zones) == 0:
        return None

    merged = unary_union(zones)
    fp = largest_poly(merged)
    return fp
