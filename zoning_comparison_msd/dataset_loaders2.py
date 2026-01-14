# =====================================================================
#   DATASET LOADERS — SAFE + CLEAN VERSION
#   Extract Apartments, Stairs, and GT Footprint from Swiss Graph
# =====================================================================

import pickle
import networkx as nx
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.errors import ShapelyError

from utils import load_pickle
from constants1 import ROOM_NAMES


# ---------------------------------------------------------------
# Utility: largest polygon
# ---------------------------------------------------------------
def largest_poly(g):
    if g is None:
        return None
    if g.geom_type == "Polygon":
        return g
    if g.geom_type == "MultiPolygon" and len(g.geoms) > 0:
        return max(g.geoms, key=lambda p: p.area)
    return g


# ---------------------------------------------------------------
# Validate + create polygon safely
# ---------------------------------------------------------------
def safe_poly(geom):
    try:
        p = Polygon(geom)
        if p.is_valid and not p.is_empty:
            return p
    except ShapelyError:
        pass
    except Exception:
        pass
    return None


# ---------------------------------------------------------------
#  MAIN GT LOADER (bulletproof)
# ---------------------------------------------------------------
def extract_gt_apartments(gt_path):
    print(f"\n[GT-LOADER] Loading: {gt_path}")

    # Load graph
    try:
        G = load_pickle(gt_path)
    except Exception as e:
        print(f"[ERROR] Cannot load GT pickle: {e}")
        return [], [], None

    print(f"[GT-LOADER] Nodes: {len(G.nodes())}, Edges: {len(G.edges())}")

    # -------------------------------
    # Room types from constants
    # -------------------------------
    NAME_TO_IDX = {n: i for i, n in enumerate(ROOM_NAMES)}

    def safe_idx(name):
        return NAME_TO_IDX[name] if name in NAME_TO_IDX else None

    PRIVATE_TYPES = {
        safe_idx("Bedroom"), safe_idx("Livingroom"), safe_idx("Kitchen"),
        safe_idx("Dining"), safe_idx("Bathroom"), safe_idx("Storeroom")
    } - {None}

    STAIRS_TYPES = {safe_idx("Stairs")} - {None}
    BALCONY_TYPES = {safe_idx("Balcony")} - {None}

    # -----------------------------------------------------------
    # Remove balconies completely
    # -----------------------------------------------------------
    balconies = [
        n for n, d in G.nodes(data=True)
        if d.get("room_type") in BALCONY_TYPES
    ]
    if balconies:
        G.remove_nodes_from(balconies)
        print(f"[GT-LOADER] Removed balconies: {len(balconies)}")

    # -----------------------------------------------------------
    # Build apartment clusters (remove entrance edges)
    # -----------------------------------------------------------
    H = G.copy()
    to_remove = [
        (u, v) for u, v, d in H.edges(data=True)
        if d.get("connectivity") == "entrance"
    ]
    H.remove_edges_from(to_remove)

    print(f"[GT-LOADER] Entrance edges removed: {len(to_remove)}")

    components = list(nx.connected_components(H))
    print(f"[GT-LOADER] Connected components: {len(components)}")

    # Filter components containing private rooms
    apartments_nodes = []
    for comp in components:
        types = [G.nodes[n].get("room_type") for n in comp]
        if any(t in PRIVATE_TYPES for t in types):
            apartments_nodes.append(comp)

    print(f"[GT-LOADER] Apartment components: {len(apartments_nodes)}")

    # -----------------------------------------------------------
    # Convert node sets → merged apartment polygons
    # -----------------------------------------------------------
    apartments = []

    for comp in apartments_nodes:
        polys = []
        for n in comp:
            geom = G.nodes[n].get("geometry")
            p = safe_poly(geom)
            if p:
                polys.append(p)

        if polys:
            merged = unary_union(polys)
            merged = merged.buffer(0.25).buffer(-0.25)
            apartments.append(largest_poly(merged))

    print(f"[GT-LOADER] Apartments extracted: {len(apartments)}")

    # -----------------------------------------------------------
    # Extract stair polygons
    # -----------------------------------------------------------
    stairs_polys = []
    for n, d in G.nodes(data=True):
        if d.get("room_type") in STAIRS_TYPES:
            p = safe_poly(d.get("geometry"))
            if p:
                stairs_polys.append(p)

    print(f"[GT-LOADER] Stair polygons: {len(stairs_polys)}")

    # -----------------------------------------------------------
    # Extract footprint from ALL valid room polygons
    # -----------------------------------------------------------
    all_polys = []
    for n, d in G.nodes(data=True):
        p = safe_poly(d.get("geometry"))
        if p:
            all_polys.append(p)

    if not all_polys:
        print("[ERROR] No valid room polygons — cannot compute footprint.")
        return apartments, stairs_polys, None

    fp = unary_union(all_polys)
    fp = fp.buffer(0.5).buffer(-0.4)
    fp = largest_poly(fp)

    if fp is None:
        print("[ERROR] GT footprint generation failed.")
        return apartments, stairs_polys, None

    print(f"[GT-LOADER] Footprint OK. Area = {fp.area:.2f}")

    # -----------------------------------------------------------
    # Return everything
    # -----------------------------------------------------------
    return apartments, stairs_polys, fp
