import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from shapely.geometry import Polygon
from shapely.ops import unary_union
from utils_apt import load_pickle
from constants_apt import ROOM_NAMES, CMAP_ROOMTYPE

# === PATH SETUP ===
datapath = r"C:\WF\Thomas Sharon\Floorplan_Dataset\archive\modified-swiss-dwellings-v2\train"
p = {"graph_out": os.path.join(datapath, "graph_out")}

# === GET AVAILABLE FILE IDS ===
ids = sorted([
    int(fname.split(".")[0])
    for fname in os.listdir(p["graph_out"])
    if fname.endswith(".pickle")
])
print(f"Found {len(ids)} floorplans in folder\n")

# === ROOM TYPE FILTERS ===
NAME_TO_IDX = {name: i for i, name in enumerate(ROOM_NAMES)}
PRIVATE_NAMES = ["Bedroom", "Livingroom", "Kitchen", "Dining", "Bathroom"]
STAIRS_NAMES = ["Stairs"]
AUXILIARY_NAMES = ["Balcony", "Storeroom"]

PRIVATE_TYPES = {NAME_TO_IDX[n] for n in PRIVATE_NAMES if n in NAME_TO_IDX}
STAIRS_TYPES = {NAME_TO_IDX[n] for n in STAIRS_NAMES if n in NAME_TO_IDX}
AUXILIARY_TYPES = {NAME_TO_IDX[n] for n in AUXILIARY_NAMES if n in NAME_TO_IDX}

# === MAIN FUNCTION ===
def plot_full_analysis(id):
    try:
        graph_path = os.path.join(p["graph_out"], f"{id}.pickle")
        if not os.path.exists(graph_path):
            print(f"⚠️ Missing graph_out for ID {id}, skipping.")
            return False

        # --- Load graph ---
        G = load_pickle(graph_path)
        print(f"\n✅ Loaded ID {id} — {len(G.nodes)} rooms, {len(G.edges)} edges")

        # === REMOVE AUXILIARY ROOMS ===
        auxiliary_nodes = [n for n, d in G.nodes(data=True) if d.get("room_type") in AUXILIARY_TYPES]
        G.remove_nodes_from(auxiliary_nodes)
        print(f"🧹 Removed {len(auxiliary_nodes)} balconies and storerooms.")

        # --- Remove entrance edges (for apartment separation) ---
        H = G.copy()
        entrance_edges = [(u, v) for u, v, d in H.edges(data=True) if d.get("connectivity") == "entrance"]
        H.remove_edges_from(entrance_edges)

        # --- Find apartment components ---
        components = list(nx.connected_components(H))
        apartments = []
        for comp in components:
            types = [G.nodes[n].get("room_type") for n in comp if "room_type" in G.nodes[n]]
            if any((t in PRIVATE_TYPES) for t in types):
                apartments.append(comp)

        print(f"🏠 Found {len(apartments)} apartment units")

        # === Compute building footprint ===
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

        merged = unary_union(room_polys)
        outer_buffer, inner_buffer = 0.5, -0.4
        footprint = merged.buffer(outer_buffer).buffer(inner_buffer)
        if footprint.geom_type == "MultiPolygon":
            footprint = max(footprint.geoms, key=lambda g: g.area)
        footprint = footprint.simplify(0.05, preserve_topology=True)

        # === Create unified apartment outlines ===
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

        # === PLOT ALL 5 VIEWS ===
        fig, axs = plt.subplots(1, 5, figsize=(38, 8))

        # ---------------------------------------------------
        # 1️⃣ RAW GRAPH_OUT
        # ---------------------------------------------------
        ax = axs[0]
        ax.set_title(f"Graph_out — ID {id}", fontsize=14)
        ax.axis("equal"); ax.axis("off")

        for n, d in G.nodes(data=True):
            geom = d.get("geometry")
            if geom:
                poly = Polygon(geom)
                if poly.is_valid:
                    room_type = d.get("room_type", 0)
                    color = np.array(CMAP_ROOMTYPE(room_type))[:3]
                    x, y = poly.exterior.xy
                    ax.fill(x, y, color=color, alpha=0.55, edgecolor="black", linewidth=0.6)

        nx.draw_networkx_edges(G, {n: G.nodes[n]["centroid"] for n in G.nodes()},
                               edgelist=G.edges(), ax=ax, edge_color="cyan", width=1, alpha=0.6)

        # ---------------------------------------------------
        # 2️⃣ APARTMENT IDENTIFICATION
        # ---------------------------------------------------
        ax = axs[1]
        ax.set_title("Apartment Identification (Stairs in Black)", fontsize=14)
        ax.axis("equal"); ax.axis("off")
        cmap = plt.cm.get_cmap("tab10", len(apartments))
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
                                ax.fill(x, y, color=cmap(i)[:3], alpha=0.6, edgecolor="black", linewidth=0.6)
                    except Exception:
                        pass
        # stairs
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
        # 3️⃣ UNIFIED APARTMENT OUTLINES
        # ---------------------------------------------------
        ax = axs[2]
        ax.set_title("Unified Apartment Outlines", fontsize=14)
        ax.axis("equal"); ax.axis("off")
        for merged_buffered in apartment_polygons:
            if merged_buffered.geom_type == "Polygon":
                x, y = merged_buffered.exterior.xy
                ax.plot(x, y, "r-", lw=2.5)
            elif merged_buffered.geom_type == "MultiPolygon":
                for part in merged_buffered.geoms:
                    x, y = part.exterior.xy
                    ax.plot(x, y, "r-", lw=2.5)
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
        # 4️⃣ BUILDING FOOTPRINT
        # ---------------------------------------------------
        ax = axs[3]
        ax.set_title("Building Footprint", fontsize=14)
        ax.axis("equal"); ax.axis("off")
        x, y = footprint.exterior.xy
        ax.fill(x, y, fc="#ffb3b3", ec="black", lw=1.8, alpha=0.9)

        # ---------------------------------------------------
        # 5️⃣ COMBINED VIEW
        # ---------------------------------------------------
        ax = axs[4]
        ax.set_title("Footprint + Apartment Outlines", fontsize=14)
        ax.axis("equal"); ax.axis("off")
        # footprint
        x, y = footprint.exterior.xy
        ax.fill(x, y, fc="#ffe6e6", ec="black", lw=1.0, alpha=0.8)
        # outlines
        for merged_buffered in apartment_polygons:
            if merged_buffered.geom_type == "Polygon":
                x, y = merged_buffered.exterior.xy
                ax.plot(x, y, "r-", lw=2.5)
            elif merged_buffered.geom_type == "MultiPolygon":
                for part in merged_buffered.geoms:
                    x, y = part.exterior.xy
                    ax.plot(x, y, "r-", lw=2.5)
        # stairs
        for n, d in G.nodes(data=True):
            if d.get("room_type") in STAIRS_TYPES:
                geom = d.get("geometry")
                if geom:
                    poly = Polygon(geom)
                    if poly.is_valid:
                        x, y = poly.exterior.xy
                        ax.fill(x, y, color="black", alpha=0.9)
                        ax.plot(x, y, color="white", linewidth=1.2)

        plt.suptitle(f"Graph_out + Apartment + Footprint Analysis — Building ID {id}", fontsize=16)
        plt.tight_layout()
        plt.show()
        return True

    except Exception as e:
        print(f"⚠️ Error processing ID {id}: {e}")
        return False


# === LOOP THROUGH FIRST 20 FILES ===
count = 0
for id in ids[2501:2600]:
    success = plot_full_analysis(id)
    if success:
        count += 1
    if count >= 100:
        print("\n🛑 Reached 4000 files — stopping loop.")
        break
