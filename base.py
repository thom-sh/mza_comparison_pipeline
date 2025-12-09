import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from shapely.geometry import Polygon
from utils import load_pickle
from constants1 import CMAP_ROOMTYPE, ROOM_NAMES, COLORS_ROOMTYPE

# --- paths ---
datapath = r"N:\9_SF-Public\Austausch\Thomas Sharon\Master_Thesis_Sharon\Floorplan_Dataset\archive\modified-swiss-dwellings-v2\train"
p = {
    "struct_in": os.path.join(datapath, "struct_in"),
    "graph_out": os.path.join(datapath, "graph_out"),
}

# --- choose sample ID ---
id = 75  # change this as needed

# --- load struct_in (with x/y grids) and graph_out ---
stack = np.load(os.path.join(p["struct_in"], f"{id}.npy"))
struct = stack[..., 0]
X = stack[..., 1].astype(np.float64)
Y = stack[..., 2].astype(np.float64)
G = load_pickle(os.path.join(p["graph_out"], f"{id}.pickle"))

h, w = struct.shape

# --- compute affine transform (world → pixel) ---
uu, vv = np.meshgrid(np.arange(w), np.arange(h))  # uu: columns, vv: rows
xs = X.reshape(-1)
ys = Y.reshape(-1)
ones = np.ones_like(xs)
A = np.stack([xs, ys, ones], axis=1)
u = uu.reshape(-1).astype(float)
v = vv.reshape(-1).astype(float)

Mu, *_ = np.linalg.lstsq(A, u, rcond=None)
Mv, *_ = np.linalg.lstsq(A, v, rcond=None)
M = np.stack([Mu, Mv], axis=1)  # [3,2]

def world_to_pixel(points_xy):
    pts = np.hstack([points_xy, np.ones((points_xy.shape[0], 1))])
    return pts @ M

# --- transform polygons + centroids ---
pos, polys_px = {}, {}
for n, d in G.nodes(data=True):
    poly_xy = np.asarray(d["geometry"])
    uv = world_to_pixel(poly_xy)
    polys_px[n] = uv
    c_xy = np.asarray(d["centroid"])
    c_uv = world_to_pixel(c_xy.reshape(1, 2))[0]
    pos[n] = (float(c_uv[0]), float(c_uv[1]))

# --- plot overlay ---
fig, ax = plt.subplots(figsize=(12, 10))
ax.imshow(struct, cmap="gray", alpha=0.5)
ax.set_title(f"Graph_out over Struct_in (Affine-aligned, ID {id})", fontsize=16)
ax.axis("off")

# --- draw room polygons (all room types) ---
for n, d in G.nodes(data=True):
    room_type = d.get("room_type", "Bedroom")
    uv = polys_px[n]
    color = np.array(CMAP_ROOMTYPE(room_type))[:3]
    ax.fill(uv[:, 0], uv[:, 1], color=color, alpha=0.45, edgecolor="black", linewidth=0.6)

# --- draw edges + centroid nodes ---
valid_edges = [(u_, v_) for u_, v_ in G.edges() if u_ in pos and v_ in pos]
nx.draw_networkx_edges(G, pos, edgelist=valid_edges, ax=ax, edge_color="cyan", width=2, alpha=0.7)

node_colors = [np.array(CMAP_ROOMTYPE(G.nodes[n].get("room_type", "Bedroom")))[:3] for n in G.nodes()]
nx.draw_networkx_nodes(
    G, pos, ax=ax,
    node_color=node_colors,
    node_size=70,
    edgecolors="black",
    linewidths=0.6
)

# --- add legend on the side ---
from matplotlib.patches import Patch

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
