# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.geometry import JOIN_STYLE

from MZA_Thesis.archive.zoning_comparison_msd.utils_apt import load_pickle
from MZA_Thesis.archive.zoning_comparison_msd.constants_apt import ROOM_NAMES


# ============================================================
# Load / Types
# ============================================================

def load_graph(datapath: str, building_id: int):
    """Load MSD graph_out pickle for a given building ID."""
    path = os.path.join(datapath, "graph_out", f"{building_id}.pickle")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return load_pickle(path)


def get_type_sets():
    """
    Return room type index mapping and type sets.

    NOTE:
    - 'core' is NOT defined by room type.
      Core is defined topologically as "before the apartment entrance".
    - We still need private_types to decide which connected components are apartments.
    """
    name_to_idx = {name: i for i, name in enumerate(ROOM_NAMES)}

    private_names = ["Bedroom", "Livingroom", "Kitchen", "Dining", "Bathroom"]
    auxiliary_names = ["Balcony"]  # add "Storeroom" here if you want to remove it too

    private_types = {name_to_idx[n] for n in private_names if n in name_to_idx}
    auxiliary_types = {name_to_idx[n] for n in auxiliary_names if n in name_to_idx}

    return name_to_idx, private_types, auxiliary_types


# ============================================================
# Graph logic
# ============================================================

def remove_auxiliary_rooms(G: nx.Graph, auxiliary_types: set) -> int:
    """Remove auxiliary nodes (e.g., Balcony) from graph in-place. Returns count removed."""
    auxiliary_nodes = [n for n, d in G.nodes(data=True) if d.get("room_type") in auxiliary_types]
    G.remove_nodes_from(auxiliary_nodes)
    return len(auxiliary_nodes)


def split_by_entrances(G: nx.Graph):
    """
    Create a copy of G without 'entrance' edges.
    This enforces your definition:
      - Core / circulation = nodes BEFORE the apartment entrance
      - Apartments = nodes AFTER the apartment entrance
    """
    H = G.copy()
    entrance_edges = [(u, v) for u, v, d in H.edges(data=True) if d.get("connectivity") == "entrance"]
    H.remove_edges_from(entrance_edges)
    return H


def detect_apartments_and_core_nodes(G: nx.Graph, private_types: set):
    """
    Apartments = connected components (after removing entrance edges) that contain ≥1 private room.
    Core nodes = all remaining nodes (after removing entrance edges) that are NOT part of any apartment.
    """
    H = split_by_entrances(G)

    apartments = []
    apartment_nodes = set()

    for comp in nx.connected_components(H):
        # component is an apartment if it contains at least one private room type
        has_private = any(G.nodes[n].get("room_type") in private_types for n in comp)
        if has_private:
            apartments.append(set(comp))
            apartment_nodes |= set(comp)

    # core = everything else left in H that is not within any apartment component
    core_nodes = set(H.nodes()) - apartment_nodes

    return apartments, core_nodes


# ============================================================
# Geometry extraction
# ============================================================

def extract_apartment_polygons(
    G: nx.Graph,
    apartments: list,
    auxiliary_types: set,
    buffer_amt: float = 0.2
):
    """
    Merge room polygons inside each apartment component into a unified apartment outline.
    Excludes auxiliary rooms if still present.
    Returns: list of shapely geometries (Polygon or MultiPolygon)
    """
    apartment_polygons = []

    for comp in apartments:
        polys = []
        for n in comp:
            if G.nodes[n].get("room_type") in auxiliary_types:
                continue

            geom = G.nodes[n].get("geometry")
            if not geom:
                continue

            try:
                poly = Polygon(geom)
                if poly.is_valid and not poly.is_empty:
                    polys.append(poly)
            except Exception:
                pass

        if not polys:
            continue

        merged = unary_union(polys)
        merged = merged.buffer(buffer_amt).buffer(-buffer_amt)

        if merged.geom_type == "MultiPolygon":
            merged = unary_union(merged)

        apartment_polygons.append(merged)

    return apartment_polygons


def extract_core_union_from_nodes(G: nx.Graph, core_nodes: set, buffer_amt: float = 0.15):
    """
    Core = nodes before apartment entrances (topology-based).
    Build a single union polygon from those node geometries.
    """
    polys = []
    for n in core_nodes:
        geom = G.nodes[n].get("geometry")
        if not geom:
            continue
        try:
            poly = Polygon(geom)
            if poly.is_valid and not poly.is_empty:
                polys.append(poly)
        except Exception:
            pass

    if not polys:
        return None

    merged = unary_union(polys)
    merged = merged.buffer(buffer_amt).buffer(-buffer_amt)

    # If multipolygon, keep as-is (plot handles it). If you prefer largest:
    # if merged.geom_type == "MultiPolygon":
    #     merged = max(merged.geoms, key=lambda g: g.area)

    return merged


def extract_building_footprint_from_apts_and_core(
    apartment_polygons: list,
    core_union,
    outer_buffer: float = 0.5,
    inner_buffer: float = -0.4,
    simplify_tol: float = 0.05,
):
    """
    Footprint = (union of apartment polygons) UNION (core union)
    then smooth + simplify.
    """
    apartments_union = unary_union(apartment_polygons) if apartment_polygons else None

    if apartments_union is None and core_union is None:
        raise ValueError("No apartment polygons and no core polygons found for footprint.")

    if apartments_union is None:
        merged = core_union
    elif core_union is None:
        merged = apartments_union
    else:
        merged = unary_union([apartments_union, core_union])

    footprint = merged.buffer(outer_buffer, join_style=JOIN_STYLE.mitre) \
                 .buffer(inner_buffer, join_style=JOIN_STYLE.mitre)


    if footprint.geom_type == "MultiPolygon":
        footprint = max(footprint.geoms, key=lambda g: g.area)

    footprint = footprint.simplify(simplify_tol, preserve_topology=True)
    return footprint


# ============================================================
# Plotting helpers
# ============================================================

def draw_union(ax, geom, fill=True, alpha=0.9):
    """Draw a Polygon or MultiPolygon union."""
    if geom is None or geom.is_empty:
        return
    geoms = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
    for g in geoms:
        if g.is_empty:
            continue
        x, y = g.exterior.xy
        if fill:
            ax.fill(x, y, color="black", alpha=alpha)
        ax.plot(x, y, color="white", linewidth=1.2)


def plot_all_views(
    G,
    apartments,
    apartment_polygons,
    core_nodes,
    core_union,
    footprint,
    building_id
):
    fig, axs = plt.subplots(1, 4, figsize=(30, 8))

    # -------------------------
    # 1) Apartment identification (room-level)
    # -------------------------
    ax = axs[0]
    ax.set_title("Apartment Identification (Core = before entrance, in Black)", fontsize=14)
    ax.axis("equal"); ax.axis("off")

    colors = plt.cm.get_cmap("tab10", max(1, len(apartments)))

    for j, comp in enumerate(apartments):
        for n in comp:
            if n in core_nodes:
                continue  # core nodes shouldn't be colored as apartment rooms

            geom = G.nodes[n].get("geometry")
            if not geom:
                continue

            try:
                poly = Polygon(geom)
                if poly.is_valid and not poly.is_empty:
                    x, y = poly.exterior.xy
                    ax.fill(x, y, color=colors(j)[:3], alpha=0.55,
                            edgecolor="black", linewidth=0.6)
            except Exception:
                pass

    # draw the core union on top
    draw_union(ax, core_union, fill=True, alpha=0.9)

    # -------------------------
    # 2) Unified apartment outlines
    # -------------------------
    ax = axs[1]
    ax.set_title("Unified Apartment Outlines + Core", fontsize=14)
    ax.axis("equal"); ax.axis("off")

    for apt in apartment_polygons:
        geoms = [apt] if apt.geom_type == "Polygon" else list(apt.geoms)
        for g in geoms:
            x, y = g.exterior.xy
            ax.plot(x, y, "r-", lw=2.5)

    draw_union(ax, core_union, fill=True, alpha=0.9)

    # -------------------------
    # 3) Footprint only
    # -------------------------
    ax = axs[2]
    ax.set_title("Building Footprint (Apts ∪ Core)", fontsize=14)
    ax.axis("equal"); ax.axis("off")

    x, y = footprint.exterior.xy
    ax.fill(x, y, fc="#ffb3b3", ec="black", lw=1.8, alpha=0.9)

    # -------------------------
    # 4) Combined
    # -------------------------
    ax = axs[3]
    ax.set_title("Footprint + Apartment Outlines + Core", fontsize=14)
    ax.axis("equal"); ax.axis("off")

    x, y = footprint.exterior.xy
    ax.fill(x, y, fc="#ffe6e6", ec="black", lw=1.0, alpha=0.8)

    for apt in apartment_polygons:
        geoms = [apt] if apt.geom_type == "Polygon" else list(apt.geoms)
        for g in geoms:
            x, y = g.exterior.xy
            ax.plot(x, y, "r-", lw=2.5)

    draw_union(ax, core_union, fill=True, alpha=0.9)

    plt.suptitle(f"Apartment + Core + Footprint — Building ID {building_id}", fontsize=16)
    plt.tight_layout()
    plt.show()


# ============================================================
# Example usage (fill your paths / IDs)
# ============================================================
# DATAPATH = r".../msd_dataset_creation"   # folder that contains graph_out/
# BUILDING_ID = 22844
#
# G = load_graph(DATAPATH, BUILDING_ID)
# name_to_idx, private_types, auxiliary_types = get_type_sets()
#
# # optional: remove auxiliary rooms
# removed = remove_auxiliary_rooms(G, auxiliary_types)
# print("Removed auxiliary rooms:", removed)
#
# apartments, core_nodes = detect_apartments_and_core_nodes(G, private_types)
# apartment_polygons = extract_apartment_polygons(G, apartments, auxiliary_types)
# core_union = extract_core_union_from_nodes(G, core_nodes)
# footprint = extract_building_footprint_from_apts_and_core(apartment_polygons, core_union)
#
# plot_all_views(G, apartments, apartment_polygons, core_nodes, core_union, footprint, BUILDING_ID)
