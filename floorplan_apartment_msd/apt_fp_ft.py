# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from shapely.geometry import Polygon
from shapely.ops import unary_union

import sys
sys.path.append(r"C:\Sharon\msd_copy\floorplan_apartment\utils_apt.py")

from utils_apt import load_pickle
from constants1 import ROOM_NAMES, CMAP_ROOMTYPE

# === PATH SETUP ===
datapath = r"C:\WF\Thomas Sharon\Floorplan_Dataset\archive\modified-swiss-dwellings-v2\train"
p = {
    "struct_in": os.path.join(datapath, "struct_in"),
    "graph_out": os.path.join(datapath, "graph_out"),
}

# === CHOOSE SAMPLE ID ===
# i = [1588, 1602, 1663, 1686, 1939, 1943, 1956, 1972, 1996, 2075, 2097, 2244, 2258, 2389, 2538, 2542, 2751, 2894, 3451, 3594, 5443] # change as needed
i = [8562]

for ID in i:
    # === LOAD FILES ===
    stack = np.load(os.path.join(p["struct_in"], f"{ID}.npy"))
    G = load_pickle(os.path.join(p["graph_out"], f"{ID}.pickle"))
    print(f"Loaded graph_out for ID {ID}: {len(G.nodes)} rooms, {len(G.edges)} edges")

    # === ROOM TYPE FILTERS ===
    NAME_TO_IDX = {name: i for i, name in enumerate(ROOM_NAMES)}
    PRIVATE_NAMES = ["Bedroom", "Livingroom", "Kitchen", "Dining", "Bathroom"]
    STAIRS_NAMES = ["Stairs"]
    AUXILIARY_NAMES = ["Balcony", "Storeroom"]

    PRIVATE_TYPES = {NAME_TO_IDX[n] for n in PRIVATE_NAMES if n in NAME_TO_IDX}
    STAIRS_TYPES = {NAME_TO_IDX[n] for n in STAIRS_NAMES if n in NAME_TO_IDX}
    AUXILIARY_TYPES = {NAME_TO_IDX[n] for n in AUXILIARY_NAMES if n in NAME_TO_IDX}

    # === REMOVE AUXILIARY ROOMS ===
    auxiliary_nodes = [n for n, d in G.nodes(data=True) if d.get("room_type") in AUXILIARY_TYPES]
    G.remove_nodes_from(auxiliary_nodes)
    print(f"🧹 Removed {len(auxiliary_nodes)} balconies and storerooms.")

    # === REMOVE ENTRANCE EDGES (TO SPLIT APARTMENTS) ===
    H = G.copy()
    entrance_edges = [(u, v) for u, v, d in H.edges(data=True) if d.get("connectivity") == "entrance"]
    H.remove_edges_from(entrance_edges)

    # === FIND CONNECTED COMPONENTS ===
    components = list(nx.connected_components(H))
    apartments = []
    for comp in components:
        types = [G.nodes[n].get("room_type") for n in comp if "room_type" in G.nodes[n]]
        if any((t in PRIVATE_TYPES) for t in types):
            apartments.append(comp)

    print(f"\n🏠 Detected {len(apartments)} apartment unit(s).\n")

    # === COLLECT VALID POLYGONS (for footprint) ===
    room_polys = []
    for _, d in G.nodes(data=True):
        if d.get("room_type") in AUXILIARY_TYPES:
            continue
        geom = d.get("geometry")
        if geom:
            try:
                poly = Polygon(geom)
                if poly.is_valid and not poly.is_empty:
                    room_polys.append(poly)
            except Exception:
                pass

    # === MERGE ALL POLYGONS TO GET BUILDING FOOTPRINT ===
    merged = unary_union(room_polys)
    outer_buffer = 0.5
    inner_buffer = -0.4
    footprint = merged.buffer(outer_buffer).buffer(inner_buffer)
    if footprint.geom_type == "MultiPolygon":
        footprint = max(footprint.geoms, key=lambda g: g.area)
    footprint = footprint.simplify(0.05, preserve_topology=True)

    # === CREATE UNIFIED APARTMENT OUTLINES ===
    apartment_polygons = []
    for comp in apartments:
        apt_polys = []
        for n in comp:
            geom = G.nodes[n].get("geometry")
            if geom:
                try:
                    poly = Polygon(geom)
                    if poly.is_valid and not poly.is_empty:
                        apt_polys.append(poly)
                except Exception:
                    pass
        if apt_polys:
            merged = unary_union(apt_polys)
            merged_buffered = merged.buffer(0.2).buffer(-0.2)
            if merged_buffered.geom_type == "MultiPolygon":
                merged_buffered = unary_union(merged_buffered)
            apartment_polygons.append(merged_buffered)

    # === PLOTS ===
    fig, axs = plt.subplots(1, 4, figsize=(30, 8))

    # ---------------------------------------------------
    # 1️⃣ APARTMENT IDENTIFICATION
    # ---------------------------------------------------
    ax = axs[0]
    ax.set_title("Apartment Identification (Stairs in Black)", fontsize=15)
    ax.axis("equal"); ax.axis("off")

    colors = plt.cm.get_cmap("tab10", len(apartments))
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
                            ax.fill(x, y, color=colors(i)[:3], alpha=0.55, edgecolor="black", linewidth=0.6)
                except Exception:
                    pass

    # Draw stairs
    for n, d in G.nodes(data=True):
        if d.get("room_type") in STAIRS_TYPES:
            geom = d.get("geometry")
            if geom:
                poly = Polygon(geom)
                if poly.is_valid:
                    x, y = poly.exterior.xy
                    ax.fill(x, y, color="black", alpha=0.9)
                    ax.plot(x, y, color="white", linewidth=1.2)

    # ---------------------------------------------------
    # 2️⃣ UNIFIED APARTMENT OUTLINES ONLY
    # ---------------------------------------------------
    ax = axs[1]
    ax.set_title("Unified Apartment Outlines Only", fontsize=15)
    ax.axis("equal"); ax.axis("off")

    for merged_buffered in apartment_polygons:
        if merged_buffered.geom_type == "Polygon":
            x, y = merged_buffered.exterior.xy
            ax.plot(x, y, "r-", lw=2.5)
        elif merged_buffered.geom_type == "MultiPolygon":
            for part in merged_buffered.geoms:
                x, y = part.exterior.xy
                ax.plot(x, y, "r-", lw=2.5)

    # Draw stairs
    for n, d in G.nodes(data=True):
        if d.get("room_type") in STAIRS_TYPES:
            geom = d.get("geometry")
            if geom:
                poly = Polygon(geom)
                if poly.is_valid:
                    x, y = poly.exterior.xy
                    ax.fill(x, y, color="black", alpha=0.9)
                    ax.plot(x, y, color="white", linewidth=1.2)

    # ---------------------------------------------------
    # 3️⃣ BUILDING FOOTPRINT ONLY
    # ---------------------------------------------------
    ax = axs[2]
    ax.set_title("Building Footprint (Unified Boundary)", fontsize=15)
    ax.axis("equal"); ax.axis("off")

    x, y = footprint.exterior.xy
    ax.fill(x, y, fc="#ffb3b3", ec="black", lw=1.8, alpha=0.9)

    # ---------------------------------------------------
    # 4️⃣ COMBINED VIEW: FOOTPRINT + APARTMENT OUTLINES
    # ---------------------------------------------------
    ax = axs[3]
    ax.set_title("Building Footprint + Apartment Outlines", fontsize=15)
    ax.axis("equal"); ax.axis("off")

    # Draw footprint first (light pink)
    x, y = footprint.exterior.xy
    ax.fill(x, y, fc="#ffe6e6", ec="black", lw=1.0, alpha=0.8)

    # Then overlay apartment outlines
    for merged_buffered in apartment_polygons:
        if merged_buffered.geom_type == "Polygon":
            x, y = merged_buffered.exterior.xy
            ax.plot(x, y, "r-", lw=2.5)
        elif merged_buffered.geom_type == "MultiPolygon":
            for part in merged_buffered.geoms:
                x, y = part.exterior.xy
                ax.plot(x, y, "r-", lw=2.5)

    # Stairs overlay
    for n, d in G.nodes(data=True):
        if d.get("room_type") in STAIRS_TYPES:
            geom = d.get("geometry")
            if geom:
                poly = Polygon(geom)
                if poly.is_valid:
                    x, y = poly.exterior.xy
                    ax.fill(x, y, color="black", alpha=0.9)
                    ax.plot(x, y, color="white", linewidth=1.2)

    # ---------------------------------------------------
    # FINAL LAYOUT
    # ---------------------------------------------------
    plt.suptitle(f"Apartment Identification + Footprint + Combined View — Building ID {ID}", fontsize=17)
    plt.tight_layout()
    plt.show()
