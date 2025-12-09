import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors 
import networkx as nx
from matplotlib.patches import Patch
from utils import load_pickle
from constants1 import CMAP_ROOMTYPE, COLORS_ROOMTYPE, ROOM_NAMES

# === PATH SETUP ===
datapath = r"N:\9_SF-Public\Austausch\Thomas Sharon\Master_Thesis_Sharon\Floorplan_Dataset\archive\modified-swiss-dwellings-v2\train"
struct_path = os.path.join(datapath, "struct_in")

# === Choose sample ID ===
id = 75  # change as needed

# === Load struct_in file ===
stack = np.load(os.path.join(struct_path, f"{id}.npy"))

# The struct_in .npy has multiple layers — first one is the structure mask
struct = stack[..., 0]

# === Display ===
plt.figure(figsize=(8, 8))
plt.imshow(struct, cmap="gray")
plt.title(f"Struct_in — ID {id}", fontsize=16)
plt.axis("off")
plt.tight_layout()
plt.show()

# === PATH SETUP ===
datapath = r"N:\9_SF-Public\Austausch\Thomas Sharon\Master_Thesis_Sharon\Floorplan_Dataset\archive\modified-swiss-dwellings-v2\train"
full_path = os.path.join(datapath, "full_out")

# === Choose sample ID ===
id = 75  # change as needed

# === Load full_out file ===
full_out = np.load(os.path.join(full_path, f"{id}.npy"), allow_pickle=True)
print("Full_out shape:", full_out.shape)

# --- If it's 3D (e.g., has multiple channels) use only the first layer ---
if full_out.ndim == 3 and full_out.shape[-1] > 1:
    full_out = full_out[..., 0]

# --- Convert to integer if necessary ---
full_out = full_out.astype(int)

# --- Handle out-of-range values safely ---
mask = (full_out >= 0) & (full_out < len(COLORS_ROOMTYPE))
colored_img = np.zeros((*full_out.shape, 3))
colored_img[:] = 1.0  # make background white

for i, color in enumerate(COLORS_ROOMTYPE):
    rgb = np.array(mcolors.to_rgb(color))  # ✅ FIXED: use mcolors.to_rgb
    colored_img[full_out == i] = rgb

# --- Display ---
plt.figure(figsize=(8, 8))
plt.imshow(colored_img)
plt.title(f"Full_out (Room-type colors) — ID {id}", fontsize=16)
plt.axis("off")
plt.tight_layout()
plt.show()

import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Patch
from utils import load_pickle
from constants1 import CMAP_ROOMTYPE, COLORS_ROOMTYPE, ROOM_NAMES

# === PATH SETUP ===
datapath = r"N:\9_SF-Public\Austausch\Thomas Sharon\Master_Thesis_Sharon\Floorplan_Dataset\archive\modified-swiss-dwellings-v2\train"
graph_path = os.path.join(datapath, "graph_out")

# === Choose sample ID ===
id = 75  # change as needed

# === Load the graph_out file ===
G = load_pickle(os.path.join(graph_path, f"{id}.pickle"))
print(f"Loaded graph_out for ID {id}")
print(f"Number of nodes: {len(G.nodes())}, edges: {len(G.edges())}")

# === Build positions and polygons from stored geometry ===
pos, polys = {}, {}
for n, d in G.nodes(data=True):
    if "geometry" not in d or "centroid" not in d:
        continue
    poly = np.asarray(d["geometry"])
    centroid = np.asarray(d["centroid"])
    polys[n] = poly
    pos[n] = (float(centroid[0]), float(centroid[1]))

# === Plot graph_out ===
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_title(f"Graph_out — ID {id}", fontsize=16)
ax.axis("equal")
ax.axis("off")

# Draw polygons (colored by room type)
for n, d in G.nodes(data=True):
    if n not in polys:
        continue
    room_type = d.get("room_type", "Bedroom")
    uv = polys[n]
    color = np.array(CMAP_ROOMTYPE(room_type))[:3]
    ax.fill(uv[:, 0], uv[:, 1], color=color, alpha=0.45, edgecolor="black", linewidth=0.6)

# Draw edges and nodes
nx.draw_networkx_edges(G, pos, ax=ax, edge_color="cyan", width=2, alpha=0.7)
node_colors = [np.array(CMAP_ROOMTYPE(G.nodes[n].get("room_type", "Bedroom")))[:3] for n in G.nodes()]
nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=70, edgecolors="black", linewidths=0.6)

# === Add legend ===
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
    fontsize=9,
    title_fontsize=10
)

plt.tight_layout()
plt.show()

