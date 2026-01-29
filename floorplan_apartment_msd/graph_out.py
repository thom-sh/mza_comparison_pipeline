import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Patch
from utils_apt import load_pickle
from constants1 import CMAP_ROOMTYPE, COLORS_ROOMTYPE, ROOM_NAMES

# --- paths ---
datapath = r"N:\9_SF-Public\Austausch\Thomas Sharon\Master_Thesis_Sharon\Floorplan_Dataset\archive\modified-swiss-dwellings-v2\train"
graph_path = os.path.join(datapath, "graph_out")

# --- choose sample ID ---
id = 8562  # change this as needed

# --- load graph_out (world-coordinate data) ---
G = load_pickle(os.path.join(graph_path, f"{id}.pickle"))

# --- remove balconies ---
balcony_nodes = [n for n, d in G.nodes(data=True) if d.get("room_type") == 8]
for b in balcony_nodes:
    G.remove_node(b)

# --- remove balconies ---
storeroom_nodes = [p for p, e in G.nodes(data=True) if e.get("room_type") == 6]
for c in storeroom_nodes:
    G.remove_node(c)

print(f"Removed {len(balcony_nodes)} balconies from graph.")
print(f"Removed {len(storeroom_nodes)} storerooms from graph.")

# --- extract original world-space polygons and centroids ---
pos_world = {}
polys_world = {}
for n, d in G.nodes(data=True):
    poly_xy = np.asarray(d["geometry"])   # original world coordinates
    polys_world[n] = poly_xy
    c_xy = np.asarray(d["centroid"])
    pos_world[n] = (float(c_xy[0]), float(c_xy[1]))

# --- plot in world coordinates ---
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_title(f"Graph_out (World Coordinates, No Balconies, ID {id})", fontsize=16)
ax.axis("equal")
ax.axis("off")

# --- draw room polygons ---
for n, d in G.nodes(data=True):
    if n not in polys_world:
        continue
    room_type = d.get("room_type", "Bedroom")
    color = np.array(CMAP_ROOMTYPE(room_type))[:3]
    uv = polys_world[n]
    ax.fill(uv[:, 0], uv[:, 1], color=color, alpha=0.5, edgecolor="black", linewidth=0.6)

# --- draw edges and centroid nodes ---
valid_edges = [(u_, v_) for u_, v_ in G.edges() if u_ in pos_world and v_ in pos_world]
nx.draw_networkx_edges(G, pos_world, edgelist=valid_edges, ax=ax, edge_color="cyan", width=2, alpha=0.7)

node_colors = [np.array(CMAP_ROOMTYPE(G.nodes[n].get("room_type", "Bedroom")))[:3] for n in G.nodes()]
nx.draw_networkx_nodes(G, pos_world, ax=ax, node_color=node_colors, node_size=70, edgecolors="black", linewidths=0.6)

# --- add legend ---
legend_elements = [
    Patch(facecolor=COLORS_ROOMTYPE[i], edgecolor="black", label=ROOM_NAMES[i])
    for i in range(len(ROOM_NAMES))
]
ax.legend(
    handles=legend_elements,
    title="Room Types",
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    frameon=True,
    fontsize=10,
    title_fontsize=12
)

plt.tight_layout()
plt.show()
