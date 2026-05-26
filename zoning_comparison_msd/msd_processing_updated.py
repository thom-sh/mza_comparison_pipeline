import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from shapely.geometry import Polygon, JOIN_STYLE
from shapely.ops import unary_union
from shapely.validation import make_valid

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from utils_apt import load_pickle
from constants_apt import ROOM_NAMES


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

    Core is NOT defined by room type.
    Core is defined topologically as the region before apartment entrance edges.
    """
    name_to_idx = {name: i for i, name in enumerate(ROOM_NAMES)}

    private_names = ["Bedroom", "Livingroom", "Kitchen", "Dining", "Bathroom"]
    auxiliary_names = ["Balcony"]

    private_types = {name_to_idx[n] for n in private_names if n in name_to_idx}
    auxiliary_types = {name_to_idx[n] for n in auxiliary_names if n in name_to_idx}

    return name_to_idx, private_types, auxiliary_types


# ============================================================
# Geometry helpers
# ============================================================

def safe_polygon(geom):
    if not geom:
        return None
    try:
        poly = Polygon(geom)
        if poly.is_empty:
            return None
        if not poly.is_valid:
            poly = make_valid(poly)
        if poly.is_empty:
            return None
        return poly
    except Exception:
        return None


def clean_geom(geom, simplify_tol=0.0):
    if geom is None or geom.is_empty:
        return geom

    geom = make_valid(geom)

    if geom.is_empty:
        return geom

    if simplify_tol > 0:
        geom = geom.simplify(simplify_tol, preserve_topology=True)

    return geom


def largest_polygon(geom):
    if geom is None or geom.is_empty:
        return None

    if geom.geom_type == "Polygon":
        return geom

    if geom.geom_type == "MultiPolygon":
        return max(geom.geoms, key=lambda g: g.area)

    if hasattr(geom, "geoms"):
        polys = [g for g in geom.geoms if g.geom_type == "Polygon" and not g.is_empty]
        if polys:
            return max(polys, key=lambda g: g.area)

    return None


def flatten_to_polygons(geom):
    if geom is None or geom.is_empty:
        return []

    if geom.geom_type == "Polygon":
        return [geom]

    if geom.geom_type == "MultiPolygon":
        return [g for g in geom.geoms if not g.is_empty]

    if hasattr(geom, "geoms"):
        parts = []
        for g in geom.geoms:
            if g.is_empty:
                continue
            if g.geom_type == "Polygon":
                parts.append(g)
            elif g.geom_type == "MultiPolygon":
                parts.extend([p for p in g.geoms if not p.is_empty])
        return parts

    return []


def polygon_parts(geom):
    """Return only polygonal parts from any shapely geometry."""
    return flatten_to_polygons(geom)


# ============================================================
# Graph logic
# ============================================================

def remove_auxiliary_rooms(G: nx.Graph, auxiliary_types: set) -> int:
    """Remove auxiliary nodes (e.g., Balcony) from graph in-place."""
    auxiliary_nodes = [n for n, d in G.nodes(data=True) if d.get("room_type") in auxiliary_types]
    G.remove_nodes_from(auxiliary_nodes)
    return len(auxiliary_nodes)


def split_by_entrances(G: nx.Graph):
    """
    Create a copy of G without 'entrance' edges.
    """
    H = G.copy()
    entrance_edges = [(u, v) for u, v, d in H.edges(data=True) if d.get("connectivity") == "entrance"]
    H.remove_edges_from(entrance_edges)
    return H


def detect_apartments_and_core_nodes(G: nx.Graph, private_types: set):
    """
    Apartments = connected components (after removing entrance edges)
    that contain at least one private room.
    Core nodes = all remaining nodes not part of an apartment component.
    """
    H = split_by_entrances(G)

    apartments = []
    apartment_nodes = set()

    for comp in nx.connected_components(H):
        has_private = any(G.nodes[n].get("room_type") in private_types for n in comp)
        if has_private:
            apartments.append(set(comp))
            apartment_nodes |= set(comp)

    core_nodes = set(H.nodes()) - apartment_nodes
    return apartments, core_nodes


# ============================================================
# Geometry extraction
# ============================================================

def extract_apartment_polygons(
    G: nx.Graph,
    apartments: list,
    auxiliary_types: set,
    buffer_amt: float = 0.08,
):
    """
    Merge room polygons inside each apartment component.
    Keep this conservative.
    """
    apartment_polygons = []

    for comp in apartments:
        polys = []
        for n in comp:
            if G.nodes[n].get("room_type") in auxiliary_types:
                continue

            poly = safe_polygon(G.nodes[n].get("geometry"))
            if poly is not None:
                polys.append(poly)

        if not polys:
            continue

        merged = unary_union(polys)

        if buffer_amt > 0:
            merged = merged.buffer(buffer_amt).buffer(-buffer_amt)

        merged = clean_geom(merged)

        poly_parts = polygon_parts(merged)
        if not poly_parts:
            continue

        if len(poly_parts) == 1:
            apartment_polygons.append(poly_parts[0])
        else:
            apartment_polygons.append(unary_union(poly_parts))

    return apartment_polygons


def extract_core_union_from_nodes(G: nx.Graph, core_nodes: set, buffer_amt: float = 0.15):
    """
    Core = nodes before apartment entrances.
    Core buffer is intentionally kept at 0.15.
    """
    polys = []
    for n in core_nodes:
        poly = safe_polygon(G.nodes[n].get("geometry"))
        if poly is not None:
            polys.append(poly)

    if not polys:
        return None

    merged = unary_union(polys)

    if buffer_amt > 0:
        merged = merged.buffer(buffer_amt).buffer(-buffer_amt)

    merged = clean_geom(merged)
    poly_parts = polygon_parts(merged)

    if not poly_parts:
        return None
    if len(poly_parts) == 1:
        return poly_parts[0]
    return unary_union(poly_parts)


def extract_building_footprint_from_apts_and_core(
    apartment_polygons: list,
    core_union,
    outer_buffer: float = 0.45,
    inner_buffer: float = -0.35,
    simplify_tol: float = 0.03,
):
    """
    Footprint = union(apartments, core), then outer closing.
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

    footprint = merged.buffer(
        outer_buffer,
        join_style=JOIN_STYLE.mitre
    ).buffer(
        inner_buffer,
        join_style=JOIN_STYLE.mitre
    )

    footprint = clean_geom(footprint, simplify_tol=simplify_tol)
    footprint = largest_polygon(footprint)
    return footprint


# ============================================================
# Gap filling by simultaneous apartment growth
# ============================================================

def simultaneous_apartment_growth(
    apartment_polygons,
    core_union,
    footprint,
    step=0.03,
    max_iter=400,
    min_residual_area=1e-4,
    simplify_tol=0.0,
):
    """
    Fill gaps by simultaneous outward growth of all apartments inside:
        free_domain = footprint - core

    Core is unchanged.
    Apartments expand equally until they meet each other and the footprint.
    """
    apartment_polygons = [clean_geom(a) for a in apartment_polygons if a is not None and not a.is_empty]
    core_union = clean_geom(core_union)
    footprint = clean_geom(footprint)

    if not apartment_polygons:
        return [], footprint

    free_domain = footprint if core_union is None else clean_geom(footprint.difference(core_union))
    free_domain_parts = polygon_parts(free_domain)
    free_domain = unary_union(free_domain_parts) if free_domain_parts else None

    if free_domain is None or free_domain.is_empty:
        return apartment_polygons, None

    current = apartment_polygons[:]

    for _ in range(max_iter):
        occupied = unary_union([g for g in current if g is not None and not g.is_empty])
        residual = clean_geom(free_domain.difference(occupied))

        if residual is None or residual.is_empty:
            break

        residual_area = sum(g.area for g in flatten_to_polygons(residual))
        if residual_area <= min_residual_area:
            break

        new_current = []

        for i, apt in enumerate(current):
            grown = clean_geom(apt.buffer(step))

            others = [
                current[j]
                for j in range(len(current))
                if j != i and current[j] is not None and not current[j].is_empty
            ]
            others_union = unary_union(others) if others else None

            candidate = grown.intersection(free_domain)

            if others_union is not None and not others_union.is_empty:
                candidate = candidate.difference(others_union)

            candidate = clean_geom(candidate)
            candidate_parts = polygon_parts(candidate)

            if candidate_parts:
                candidate = unary_union(candidate_parts)
                candidate = clean_geom(unary_union([apt, candidate]))
            else:
                candidate = apt

            if simplify_tol > 0 and candidate is not None and not candidate.is_empty:
                candidate = candidate.simplify(simplify_tol, preserve_topology=True)

            new_current.append(candidate)

        current = new_current

    final_apts = []
    for i, apt in enumerate(current):
        others = [
            current[j]
            for j in range(len(current))
            if j != i and current[j] is not None and not current[j].is_empty
        ]
        others_union = unary_union(others) if others else None

        clipped = apt.intersection(free_domain)
        if others_union is not None and not others_union.is_empty:
            clipped = clipped.difference(others_union)

        clipped = clean_geom(clipped)
        parts = polygon_parts(clipped)

        if not parts:
            final_apts.append(None)
        elif len(parts) == 1:
            final_apts.append(parts[0])
        else:
            final_apts.append(unary_union(parts))

    final_union = unary_union([g for g in final_apts if g is not None and not g.is_empty])
    residual = clean_geom(free_domain.difference(final_union))
    residual_parts = polygon_parts(residual)

    if not residual_parts:
        residual = None
    else:
        residual = unary_union(residual_parts)

    return final_apts, residual


# ============================================================
# Plotting helpers
# ============================================================

def draw_union(ax, geom, fill=True, alpha=0.9):
    for g in polygon_parts(geom):
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
    building_id,
    residual_gap=None,
):
    fig, axs = plt.subplots(1, 4, figsize=(30, 8))

    # 1) room-level
    ax = axs[0]
    ax.set_title("Apartment Identification (Core = before entrance, in Black)", fontsize=14)
    ax.axis("equal")
    ax.axis("off")

    detailed_colors = ["#C7D9E8", "#D7DCE0", "#9FC0DA", "#B7BEC5", "#7FA9C9", "#9AA4AD"]
    outline_color = "#222222"

    for j, comp in enumerate(apartments):
        apt_color = detailed_colors[j % len(detailed_colors)]
        for n in comp:
            if n in core_nodes:
                continue

            geom = G.nodes[n].get("geometry")
            if not geom:
                continue

            try:
                poly = Polygon(geom)
                if poly.is_valid and not poly.is_empty:
                    x, y = poly.exterior.xy
                    ax.fill(
                        x, y,
                        facecolor=apt_color,
                        alpha=0.85,
                        edgecolor=outline_color,
                        linewidth=0.8
                    )
            except Exception:
                pass

    draw_union(ax, core_union, fill=True, alpha=0.9)

    # 2) zoning
    ax = axs[1]
    ax.set_title("Gap-free Apartment Outlines + Core", fontsize=14)
    ax.axis("equal")
    ax.axis("off")

    for apt in apartment_polygons:
        for g in polygon_parts(apt):
            x, y = g.exterior.xy
            ax.plot(x, y, "r-", lw=2.5)

    draw_union(ax, core_union, fill=True, alpha=0.9)

    if residual_gap is not None and not residual_gap.is_empty:
        for g in polygon_parts(residual_gap):
            x, y = g.exterior.xy
            ax.fill(x, y, fc="yellow", ec="orange", alpha=0.7)

    # 3) footprint
    ax = axs[2]
    ax.set_title("Building Footprint (Apts ∪ Core)", fontsize=14)
    ax.axis("equal")
    ax.axis("off")

    if footprint is not None and not footprint.is_empty:
        x, y = footprint.exterior.xy
        ax.fill(x, y, fc="#ffb3b3", ec="black", lw=1.8, alpha=0.9)

    # 4) combined
    ax = axs[3]
    ax.set_title("Footprint + Gap-free Apartment Outlines + Core", fontsize=14)
    ax.axis("equal")
    ax.axis("off")

    if footprint is not None and not footprint.is_empty:
        x, y = footprint.exterior.xy
        ax.fill(x, y, fc="#ffe6e6", ec="black", lw=1.0, alpha=0.8)

    for apt in apartment_polygons:
        for g in polygon_parts(apt):
            x, y = g.exterior.xy
            ax.plot(x, y, "r-", lw=2.5)

    draw_union(ax, core_union, fill=True, alpha=0.9)

    plt.suptitle(f"Apartment + Core + Footprint — Building ID {building_id}", fontsize=16)
    plt.tight_layout()
    plt.show()
