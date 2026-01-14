# %%
# ===========================================================
#   FULL SIDE-BY-SIDE DISPLAY: SWISS GT vs PREDICTED ZONING
# ===========================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from shapely.geometry import Polygon
from shapely.ops import unary_union
import pickle

# Your utilities
import sys
sys.path.append(r"C:\Sharon\msd_copy\floorplan_apartment\utils_apt.py")
from utils import load_pickle
from constants1 import ROOM_NAMES


# ===========================================================
#                      CONFIGURATION
# ===========================================================
IDS = [75]   # change as needed

for ID in IDS:
# --- SWISS GT PATHS ---
    datapath = r"N:\9_SF-Public\Austausch\Thomas Sharon\Master_Thesis_Sharon\Floorplan_Dataset\archive\modified-swiss-dwellings-v2\train"
    p = {
        "struct_in": os.path.join(datapath, "struct_in"),
        "graph_out": os.path.join(datapath, "graph_out"),
    }

    # --- PREDICTED PATH ---
    PRED_PATH = fr"N:\9_SF-Public\Austausch\Thomas Sharon\Master_Thesis_Sharon\Floorplan_Dataset\gml_msd\building_data\building_data_{ID}.pkl"


    # ===========================================================
    #                 LOAD SWISS GROUND TRUTH
    # ===========================================================
    stack = np.load(os.path.join(p["struct_in"], f"{ID}.npy"))
    G = load_pickle(os.path.join(p["graph_out"], f"{ID}.pickle"))
    print(f"✅ Loaded Swiss GT for ID {ID}: {len(G.nodes)} rooms, {len(G.edges)} edges")

    # Room type maps
    NAME_TO_IDX = {name: i for i, name in enumerate(ROOM_NAMES)}
    PRIVATE_TYPES = {NAME_TO_IDX[n] for n in ["Bedroom","Livingroom","Kitchen","Dining","Bathroom","Storeroom"] if n in NAME_TO_IDX}
    STAIRS_TYPES  = {NAME_TO_IDX["Stairs"]} if "Stairs" in NAME_TO_IDX else set()
    BALCONY_TYPES = {NAME_TO_IDX["Balcony"]} if "Balcony" in NAME_TO_IDX else set()

    # Remove balconies
    balcony_nodes = [n for n, d in G.nodes(data=True) if d.get("room_type") in BALCONY_TYPES]
    G.remove_nodes_from(balcony_nodes)

    # Remove entrance edges
    H = G.copy()
    entrance_edges = [(u, v) for u, v, d in H.edges(data=True) if d.get("connectivity") == "entrance"]
    H.remove_edges_from(entrance_edges)

    # Connected components = apartments
    components = list(nx.connected_components(H))
    apartments = []
    for comp in components:
        types = [G.nodes[n].get("room_type") for n in comp]
        if any(t in PRIVATE_TYPES for t in types):
            apartments.append(comp)

    # Valid room polygons
    room_polys = []
    for _, d in G.nodes(data=True):
        geom = d.get("geometry")
        if geom:
            try:
                poly = Polygon(geom)
                if poly.is_valid and not poly.is_empty:
                    room_polys.append(poly)
            except:
                pass

    # Building footprint
    merged = unary_union(room_polys)
    footprint = merged.buffer(0.5).buffer(-0.4)
    if footprint.geom_type == "MultiPolygon":
        footprint = max(footprint.geoms, key=lambda g: g.area)
    footprint = footprint.simplify(0.05, preserve_topology=True)


    # ===========================================================
    #                    LOAD PREDICTED PICKLE
    # ===========================================================
    with open(PRED_PATH, "rb") as fh:
        buildings = pickle.load(fh)

    bldg = buildings[0]
    storeys = bldg["polygons"]["storeys"]

    # Find Storey_1
    storey = next((s for s in storeys if s.get("storey_name") == "Storey_1"), None)
    zones_pred = storey.get("zones", []) if storey else []
    print(f"🔧 Predicted zones found: {len(zones_pred)}")


    # Predictor helper
    def extract_xy(coords):
        try:
            return zip(*[(c[0], c[1]) for c in coords])
        except:
            return [], []


    # ===========================================================
    #                   SET UP SIDE-BY-SIDE PLOT
    # ===========================================================
    fig, axs = plt.subplots(1, 2, figsize=(22, 10))

    # -----------------------------------------------------------
    # LEFT → Swiss GT
    # -----------------------------------------------------------
    ax = axs[0]
    ax.set_title(f"Swiss Ground Truth — Building {ID}", fontsize=16)
    ax.axis("equal"); ax.axis("off")

    colors = plt.cm.get_cmap("tab10", len(apartments))

    # Apartments colored
    for i, comp in enumerate(apartments):
        for n in comp:
            geom = G.nodes[n].get("geometry")
            rtype = G.nodes[n].get("room_type")
            if geom:
                try:
                    poly = Polygon(geom)
                    if poly.is_valid and not poly.is_empty:
                        if rtype not in STAIRS_TYPES:
                            x, y = poly.exterior.xy
                            ax.fill(x, y, color=colors(i)[:3], alpha=0.55, edgecolor="black")
                except:
                    pass

    # Stairs in black
    for n, d in G.nodes(data=True):
        if d.get("room_type") in STAIRS_TYPES:
            poly = Polygon(d["geometry"])
            x, y = poly.exterior.xy
            ax.fill(x, y, color='black')


    # -----------------------------------------------------------
    # RIGHT → Predicted Zones (from Storey_1)
    # -----------------------------------------------------------
    ax = axs[1]
    ax.set_title(f"Predicted Zoning – Storey_1 — Building {ID}", fontsize=16)
    ax.axis("equal"); ax.axis("off")

    zone_colors = plt.cm.get_cmap("tab20", len(zones_pred))

    for i, zone in enumerate(zones_pred):
        zone_name = zone.get("name", f"Zone_{i}")

        for f in zone.get("floors", []):
            if isinstance(f, tuple) and isinstance(f[0], Polygon):
                poly = f[0]
                if poly and not poly.is_empty:
                    x, y = extract_xy(poly.exterior.coords)
                    ax.fill(x, y, alpha=0.55, facecolor=zone_colors(i)[:3], edgecolor="black")
                    cx, cy = poly.centroid.x, poly.centroid.y
                    ax.text(cx, cy, zone_name.split("_")[-1], ha='center', va='center', fontsize=8)


    # -----------------------------------------------------------
    # FINAL SHOW
    # -----------------------------------------------------------
    plt.tight_layout()
    plt.show()
