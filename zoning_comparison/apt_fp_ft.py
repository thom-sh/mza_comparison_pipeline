# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from shapely.geometry import Polygon
from shapely.ops import unary_union
from utils import load_pickle
from constants1 import ROOM_NAMES

# === PATH SETUP ===
datapath = r"N:\9_SF-Public\Austausch\Thomas Sharon\Master_Thesis_Sharon\Floorplan_Dataset\archive\modified-swiss-dwellings-v2\train"
p = {
    "graph_out": os.path.join(datapath, "graph_out"),
}

# === CHOOSE SAMPLE ID ===
ID = 553  # change as needed

# === LOAD GRAPH ===
G = load_pickle(os.path.join(p["graph_out"], f"{ID}.pickle"))
print(f"✅ Loaded graph_out for ID {ID}: {len(G.nodes)} rooms, {len(G.edges)} edges")

# === ROOM TYPE FILTERS ===
NAME_TO_IDX = {name: i for i, name in enumerate(ROOM_NAMES)}
PRIVATE_NAMES = ["Bedroom", "Livingroom", "Kitchen", "Dining", "Bathroom", "Storeroom"]
STAIRS_NAMES = ["Stairs"]
BALCONY_NAMES = ["Balcony"]

PRIVATE_TYPES = {NAME_TO_IDX[n] for n in PRIVATE_NAMES if n in NAME_TO_IDX}
STAIRS_TYPES = {NAME_TO_IDX[n] for n in STAIRS_NAMES if n in NAME_TO_IDX}
BALCONY_TYPES = {NAME_TO_IDX[n] for n in BALCONY_NAMES if n in NAME_TO_IDX}

# === REMOVE BALCONIES ===
balcony_nodes = [n for n, d in G.nodes(data=True) if d.get("room_type") in BALCONY_TYPES]
G.remove_nodes_from(balcony_nodes)
print(f"🧹 Removed {len(balcony_nodes)} balconies.")

# === REMOVE ENTRANCE EDGES (TO SPLIT APARTMENTS) ===
H = G.copy()
entrance_edges = [(u, v) for u, v, d in H.edges(data=True) if d.get("connectivity") == "entrance"]
H.remove_edges_from(entrance_edges)

# === FIND CONNECTED COMPONENTS (apartments) ===
components = list(nx.connected_components(H))
apartments = []
for comp in components:
    types = [G.nodes[n].get("room_type") for n in comp if "room_type" in G.nodes[n]]
    if any((t in PRIVATE_TYPES) for t in types):
        apartments.append(comp)

print(f"\n🏠 Detected {len(apartments)} apartment unit(s).\n")

# === COLLECT VALID ROOM POLYGONS (for footprint) ===
room_polys = []
for _, d in G.nodes(data=True):
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
footprint = merged.buffer(0.5).buffer(-0.4)
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

# === PLOTS (2 PANELS) ===
fig, axs = plt.subplots(1, 2, figsize=(16, 8))

# ---------------------------------------------------
# 1️⃣ BUILDING FOOTPRINT
# ---------------------------------------------------
ax = axs[0]
ax.set_title("Building Footprint", fontsize=15)
ax.axis("equal"); ax.axis("off")

x, y = footprint.exterior.xy
ax.fill(x, y, fc="#ffb3b3", ec="black", lw=1.8, alpha=0.9)

# ---------------------------------------------------
# 2️⃣ APARTMENTS + STAIRCASES (no footprint)
# ---------------------------------------------------
ax = axs[1]
ax.set_title("Apartments and Staircases", fontsize=15)
ax.axis("equal"); ax.axis("off")

colors = plt.cm.get_cmap("tab10", len(apartments))
for i, merged_buffered in enumerate(apartment_polygons):
    if merged_buffered.geom_type == "Polygon":
        x, y = merged_buffered.exterior.xy
        ax.fill(x, y, color=colors(i)[:3], alpha=0.6, ec="black", lw=0.8)
    elif merged_buffered.geom_type == "MultiPolygon":
        for part in merged_buffered.geoms:
            x, y = part.exterior.xy
            ax.fill(x, y, color=colors(i)[:3], alpha=0.6, ec="black", lw=0.8)

# Draw staircases in black
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
plt.suptitle(f"Building Footprint and Apartment Layout — ID {ID}", fontsize=17)
plt.tight_layout()
plt.show()
