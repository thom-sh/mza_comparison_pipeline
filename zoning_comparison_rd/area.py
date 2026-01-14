# ===========================================================
#  SWISS GT FOOTPRINT (Balconies removed, entrance edges removed)
# ===========================================================

import os
from shapely.geometry import Polygon
from shapely.ops import unary_union
import networkx as nx

import sys
sys.path.append(r"C:\Sharon\msd_copy\floorplan_apartment")

from utils import load_pickle
from constants1 import ROOM_NAMES


# ----------------------
# CONFIG
# ----------------------
ID = 68
datapath_gt = r"N:\9_SF-Public\Austausch\Thomas Sharon\Master_Thesis_Sharon\Floorplan_Dataset\archive\modified-swiss-dwellings-v2\train"
GT_GRAPH = os.path.join(datapath_gt, "graph_out", f"{ID}.pickle")

# ----------------------
# 1. LOAD GT GRAPH
# ----------------------
G = load_pickle(GT_GRAPH)

NAME_TO_IDX = {name: i for i, name in enumerate(ROOM_NAMES)}
BALCONY = NAME_TO_IDX["Balcony"]

# Remove balconies
bal_nodes = [n for n, d in G.nodes(data=True) if d["room_type"] == BALCONY]
G.remove_nodes_from(bal_nodes)

# Remove entrance edges (important!)
H = G.copy()
entrance_edges = [(u, v) for u, v, d in H.edges(data=True) if d.get("connectivity") == "entrance"]
H.remove_edges_from(entrance_edges)

# ----------------------
# 2. COLLECT ROOM POLYGONS
# ----------------------
room_polys = []
for _, d in H.nodes(data=True):
    geom = d.get("geometry")
    if geom:
        try:
            poly = Polygon(geom)
            if poly.is_valid and not poly.is_empty:
                room_polys.append(poly)
        except:
            pass

# ----------------------
# 3. MERGE FOOTPRINT
# ----------------------
merged = unary_union(room_polys)

# Buffer cleaning (same as main pipeline)
footprint = merged.buffer(0.5).buffer(-0.4)

# Keep largest piece if needed
if footprint.geom_type == "MultiPolygon":
    footprint = max(footprint.geoms, key=lambda g: g.area)

# Optional clean
footprint = footprint.simplify(0.05, preserve_topology=True)

# ----------------------
# 4. AREA
# ----------------------
area = footprint.area
print(f"🏠 Swiss GT FOOTPRINT AREA (correct) = {area:.2f} m²")
