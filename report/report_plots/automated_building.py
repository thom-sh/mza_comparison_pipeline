"""
Create a representative BSP-to-MZA zoning schematic from a building pickle.

Output:
    One PDF figure only. The figure is also displayed after saving.

Edit only the CONFIGURATION section below:
    INPUT_PICKLE  : path to your building_data pickle
    OUTPUT_DIR    : destination folder
    OUTPUT_NAME   : output PDF file name
    BUILDING_INDEX: index if the pickle contains a list of buildings
    STOREY_INDEX  : storey/floor index to plot, starting from 0

Expected pickle structure:
    The script is designed for MZA-style building_data pickles, e.g.
    building["polygons"]["gf_polygon"]
    building["polygons"]["storeys"][i]["zones"]

Core-zone classification:
    By default, the script assumes that the first n_cores zones in a storey
    are shared core/staircase zones, as used in the MZA building assembly.
    You can override this with CORE_ZONE_INDICES.
"""

# =========================
# CONFIGURATION
# =========================
INPUT_PICKLE = r"C:\WF\Thomas Sharon\Floorplan_Dataset\msd_pickle\75.pickle"
OUTPUT_DIR = r"C:\WF\Thomas Sharon\Floorplan_Dataset\figures"
OUTPUT_NAME = "bsp_mza_schematic.pdf"

BUILDING_INDEX = 0       # used when pickle contains a list of buildings
STOREY_INDEX = 0         # 0 = first storey

# If None, core zones are inferred as the first n_cores zones.
# Example: CORE_ZONE_INDICES = [0] or [0, 1]
CORE_ZONE_INDICES = None

# Representative BSP discretisation rebuilt from the footprint.
# Increase BSP_MIN_CELL_SIZE for fewer/larger leaves; decrease for finer leaves.
BSP_MIN_CELL_SIZE = 1.25
BSP_MAX_DEPTH = 12
BSP_MAX_LEAVES = 650
MIN_LEAF_AREA = 0.02

# Figure settings
FIGSIZE = (7.2, 4.2)
DPI = 300
SHOW_AFTER_SAVE = True

# =========================
# THESIS COLOR PALETTE
# =========================
LEGEND_EDGE_COLOR = "#bdc1c5"
POLYGON_EDGE_COLOR = "#777d84"
CORE_EDGE_COLOR = "#777d84"
LEAF_EDGE_COLOR = "#bdc1c5"
TREE_EDGE_COLOR = "#777d84"
TREE_FILL_COLOR = "white"

DWELLING_COLORS = [
    "#DCEAF7",
    "#AFCBE3",
    "#AFCBE3",
    "#DBDBDB",
    "#BABABA",
    "#999999",
]
CORE_COLOR = "#5F666D"

# =========================
# IMPORTS
# =========================
from pathlib import Path
import pickle
import math

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch
from matplotlib.lines import Line2D
import numpy as np

from shapely.geometry import Polygon, MultiPolygon, GeometryCollection, box
from shapely.ops import unary_union


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIX Two Text", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "axes.titleweight": "normal",
    "axes.labelweight": "normal",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


# =========================
# GEOMETRY HELPERS
# =========================
def iter_polygons(geom):
    """Yield simple Polygon objects from Polygon/MultiPolygon/GeometryCollection/list."""
    if geom is None:
        return
    if isinstance(geom, tuple) and geom and hasattr(geom[0], "geom_type"):
        geom = geom[0]
    if isinstance(geom, Polygon):
        if not geom.is_empty and geom.area > 0:
            yield geom
    elif isinstance(geom, MultiPolygon):
        for g in geom.geoms:
            yield from iter_polygons(g)
    elif isinstance(geom, GeometryCollection):
        for g in geom.geoms:
            yield from iter_polygons(g)
    elif isinstance(geom, (list, tuple)):
        for g in geom:
            yield from iter_polygons(g)


def to_2d_polygon(poly):
    """Convert a 2D/3D Shapely polygon to a valid 2D polygon."""
    if poly is None:
        return None
    if isinstance(poly, tuple) and poly and hasattr(poly[0], "geom_type"):
        poly = poly[0]
    if not isinstance(poly, Polygon):
        return None
    try:
        ext = [(float(c[0]), float(c[1])) for c in poly.exterior.coords]
        holes = []
        for ring in poly.interiors:
            holes.append([(float(c[0]), float(c[1])) for c in ring.coords])
        p = Polygon(ext, holes).buffer(0)
        if p.is_empty:
            return None
        return p
    except Exception:
        return None


def to_2d_geom(geom):
    """Convert polygonal geometry/list to cleaned 2D polygonal geometry."""
    polys = []
    for p in iter_polygons(geom):
        p2 = to_2d_polygon(p)
        if p2 is not None and not p2.is_empty and p2.area > 0:
            polys.extend(list(iter_polygons(p2)))
    if not polys:
        return None
    return unary_union(polys).buffer(0)


def get_first_polygonal(mapping, keys):
    """Return the first available polygonal object from a dictionary."""
    for key in keys:
        if isinstance(mapping, dict) and key in mapping:
            geom = to_2d_geom(mapping[key])
            if geom is not None and not geom.is_empty:
                return geom
    return None


def extract_floor_polygon_from_zone(zone):
    """Return the union of floor polygons from one zone dictionary."""
    if not isinstance(zone, dict):
        return None
    floors = zone.get("floors") or zone.get("floor") or []
    return to_2d_geom(floors)


def load_building_data(pickle_path, building_index=0):
    with open(pickle_path, "rb") as fh:
        data = pickle.load(fh)

    if isinstance(data, list):
        if not data:
            raise ValueError("The pickle contains an empty list.")
        return data[building_index]

    if isinstance(data, dict):
        return data

    raise TypeError(f"Unsupported pickle content type: {type(data)}")


def extract_footprint(building, zones_geom=None):
    """Extract footprint from MZA building dict; fallback to union of zones."""
    if isinstance(building, dict):
        polys = building.get("polygons", {})
        footprint = get_first_polygonal(
            polys,
            ["gf_polygon", "ground_floor", "footprint", "floor_polygon"]
        )
        if footprint is not None:
            return footprint

        footprint = get_first_polygonal(
            building,
            ["gf_polygon", "ground_floor", "footprint", "floor_polygon"]
        )
        if footprint is not None:
            return footprint

    if zones_geom is not None:
        return zones_geom

    raise ValueError("Could not find a building footprint in the pickle.")


def extract_storey(building, storey_index=0):
    if not isinstance(building, dict):
        raise TypeError("The selected building is not a dictionary.")

    polygons = building.get("polygons", {})
    storeys = polygons.get("storeys") or building.get("storeys")
    if not storeys:
        raise ValueError("No storeys found in the selected building.")

    if storey_index < 0 or storey_index >= len(storeys):
        raise IndexError(f"STOREY_INDEX={storey_index} is outside available range 0..{len(storeys)-1}.")

    return storeys[storey_index]


def extract_zone_geometries(building, storey, core_zone_indices=None):
    """Return dwelling polygons and core polygons for one storey."""
    zones = storey.get("zones", []) if isinstance(storey, dict) else []
    zone_polys = []
    for zone in zones:
        zp = extract_floor_polygon_from_zone(zone)
        if zp is not None and not zp.is_empty and zp.area > 0:
            zone_polys.append(zp)

    if not zone_polys:
        raise ValueError("No usable zone floor polygons found in the selected storey.")

    if core_zone_indices is None:
        n_cores = 1
        try:
            n_cores = int(building.get("building_data", {}).get("bldg:n_cores", 1))
        except Exception:
            n_cores = 1
        core_zone_indices = list(range(max(1, n_cores)))

    core_zone_indices = set(int(i) for i in core_zone_indices)
    core_polys = [p for i, p in enumerate(zone_polys) if i in core_zone_indices]
    dwelling_polys = [p for i, p in enumerate(zone_polys) if i not in core_zone_indices]

    # Fallback if inferred core index is not usable.
    if not core_polys:
        stair = to_2d_geom(building.get("staircase_polygon"))
        if stair is not None:
            core_polys = [stair]

    return dwelling_polys, core_polys, zone_polys


# =========================
# REPRESENTATIVE BSP
# =========================
def split_polygon_recursive(poly, min_cell_size, max_depth, min_area, max_leaves, leaves=None, depth=0):
    """Simple representative BSP: recursively split by the larger bounding-box dimension."""
    if leaves is None:
        leaves = []

    if poly is None or poly.is_empty:
        return leaves

    # Protect against runaway leaf counts.
    if len(leaves) >= max_leaves:
        leaves.append(poly)
        return leaves

    minx, miny, maxx, maxy = poly.bounds
    width = maxx - minx
    height = maxy - miny

    stop = (
        depth >= max_depth
        or min(width, height) <= min_cell_size
        or poly.area <= max(min_area, min_cell_size * min_cell_size)
    )
    if stop:
        for p in iter_polygons(poly):
            if p.area > min_area:
                leaves.append(p)
        return leaves

    if width >= height:
        split_x = (minx + maxx) / 2.0
        cutters = [box(minx, miny, split_x, maxy), box(split_x, miny, maxx, maxy)]
    else:
        split_y = (miny + maxy) / 2.0
        cutters = [box(minx, miny, maxx, split_y), box(minx, split_y, maxx, maxy)]

    children = []
    for cutter in cutters:
        child = poly.intersection(cutter).buffer(0)
        if child is not None and not child.is_empty and child.area > min_area:
            children.append(child)

    if len(children) < 2:
        for p in iter_polygons(poly):
            if p.area > min_area:
                leaves.append(p)
        return leaves

    for child in children:
        split_polygon_recursive(
            child,
            min_cell_size=min_cell_size,
            max_depth=max_depth,
            min_area=min_area,
            max_leaves=max_leaves,
            leaves=leaves,
            depth=depth + 1,
        )

    return leaves


# =========================
# PLOTTING HELPERS
# =========================
def plot_polygon(ax, geom, facecolor="none", edgecolor=POLYGON_EDGE_COLOR,
                 linewidth=0.8, alpha=1.0, zorder=1):
    """Plot Polygon/MultiPolygon with fill and outline."""
    if geom is None:
        return
    for poly in iter_polygons(geom):
        x, y = poly.exterior.xy
        ax.fill(x, y, facecolor=facecolor, edgecolor=edgecolor,
                linewidth=linewidth, alpha=alpha, zorder=zorder)
        # Holes, if any
        for hole in poly.interiors:
            hx, hy = hole.xy
            ax.fill(hx, hy, facecolor="white", edgecolor=edgecolor,
                    linewidth=linewidth * 0.8, alpha=1.0, zorder=zorder + 0.1)


def plot_outline(ax, geom, edgecolor=POLYGON_EDGE_COLOR, linewidth=1.0, zorder=5):
    if geom is None:
        return
    for poly in iter_polygons(geom):
        x, y = poly.exterior.xy
        ax.plot(x, y, color=edgecolor, linewidth=linewidth, zorder=zorder)
        for hole in poly.interiors:
            hx, hy = hole.xy
            ax.plot(hx, hy, color=edgecolor, linewidth=linewidth * 0.8, zorder=zorder)


def setup_plan_axis(ax, footprint):
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    minx, miny, maxx, maxy = footprint.bounds
    dx = maxx - minx
    dy = maxy - miny
    pad = 0.08 * max(dx, dy)
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)


def draw_bsp_tree_panel(ax):
    """Draw a compact conceptual BSP tree."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    nodes = {
        "A": (0.50, 0.88),
        "B": (0.30, 0.68),
        "C": (0.70, 0.68),
        "D": (0.18, 0.45),
        "E": (0.42, 0.45),
        "F": (0.34, 0.22),
        "G": (0.52, 0.22),
    }
    edges = [("A", "B"), ("A", "C"), ("B", "D"), ("B", "E"), ("E", "F"), ("E", "G")]
    leaf_nodes = {"C", "D", "F", "G"}

    for a, b in edges:
        xa, ya = nodes[a]
        xb, yb = nodes[b]
        ax.plot([xa, xb], [ya, yb], color=TREE_EDGE_COLOR, linewidth=0.8, zorder=1)

    for label, (x, y) in nodes.items():
        circle = Circle((x, y), 0.045, facecolor=TREE_FILL_COLOR,
                        edgecolor=TREE_EDGE_COLOR, linewidth=0.8, zorder=2)
        ax.add_patch(circle)
        ax.text(x, y, label, ha="center", va="center", fontsize=8, zorder=3)

    # Leaf indication using the same restrained palette.
    for label in leaf_nodes:
        x, y = nodes[label]
        ax.add_patch(Circle((x, y), 0.050, facecolor="none",
                            edgecolor=CORE_EDGE_COLOR, linewidth=1.0, zorder=2.5))

    ax.text(0.5, 0.04, "terminal nodes = leaves", ha="center", va="bottom", fontsize=7)


def add_panel_label(ax, label):
    ax.text(0.5, -0.08, label, transform=ax.transAxes,
            ha="center", va="top", fontsize=8)


# =========================
# MAIN FIGURE
# =========================
def main():
    input_path = Path(INPUT_PICKLE)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / OUTPUT_NAME

    building = load_building_data(input_path, BUILDING_INDEX)
    storey = extract_storey(building, STOREY_INDEX)
    dwelling_polys, core_polys, all_zone_polys = extract_zone_geometries(
        building, storey, CORE_ZONE_INDICES
    )

    all_zones_union = unary_union(all_zone_polys).buffer(0)
    footprint = extract_footprint(building, zones_geom=all_zones_union).buffer(0)
    core_geom = unary_union(core_polys).buffer(0) if core_polys else None
    dwellings_union = unary_union(dwelling_polys).buffer(0) if dwelling_polys else None

    # Representative BSP leaves are rebuilt from the footprint.
    bsp_leaves = split_polygon_recursive(
        footprint,
        min_cell_size=BSP_MIN_CELL_SIZE,
        max_depth=BSP_MAX_DEPTH,
        min_area=MIN_LEAF_AREA,
        max_leaves=BSP_MAX_LEAVES,
    )

    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE, dpi=DPI)
    ax_tree, ax_bsp, ax_core, ax_final = axes.ravel()

    # (a) BSP tree logic
    draw_bsp_tree_panel(ax_tree)
    ax_tree.set_title("BSP tree logic", pad=4)
    add_panel_label(ax_tree, "(a)")

    # (b) Footprint divided into BSP leaves
    setup_plan_axis(ax_bsp, footprint)
    for leaf in bsp_leaves:
        plot_polygon(ax_bsp, leaf, facecolor="white", edgecolor=LEAF_EDGE_COLOR,
                     linewidth=0.35, alpha=1.0, zorder=1)
    plot_outline(ax_bsp, footprint, edgecolor=POLYGON_EDGE_COLOR, linewidth=1.0, zorder=4)
    ax_bsp.set_title("BSP leaves on floor footprint", pad=4)
    add_panel_label(ax_bsp, "(b)")

    # (c) Core/staircase reference on leaves
    setup_plan_axis(ax_core, footprint)
    for leaf in bsp_leaves:
        plot_polygon(ax_core, leaf, facecolor="white", edgecolor=LEAF_EDGE_COLOR,
                     linewidth=0.30, alpha=0.9, zorder=1)
    if core_geom is not None:
        plot_polygon(ax_core, core_geom, facecolor=CORE_COLOR, edgecolor=CORE_EDGE_COLOR,
                     linewidth=0.9, alpha=1.0, zorder=3)
    plot_outline(ax_core, footprint, edgecolor=POLYGON_EDGE_COLOR, linewidth=1.0, zorder=4)
    ax_core.set_title("Shared core as zoning reference", pad=4)
    add_panel_label(ax_core, "(c)")

    # (d) Final dwelling/core zones
    setup_plan_axis(ax_final, footprint)
    # Optional faint leaves in background
    for leaf in bsp_leaves:
        plot_outline(ax_final, leaf, edgecolor=LEAF_EDGE_COLOR, linewidth=0.18, zorder=0)
    for i, dwelling in enumerate(dwelling_polys):
        color = DWELLING_COLORS[i % len(DWELLING_COLORS)]
        plot_polygon(ax_final, dwelling, facecolor=color, edgecolor=POLYGON_EDGE_COLOR,
                     linewidth=0.9, alpha=1.0, zorder=2)
    if core_geom is not None:
        plot_polygon(ax_final, core_geom, facecolor=CORE_COLOR, edgecolor=CORE_EDGE_COLOR,
                     linewidth=0.9, alpha=1.0, zorder=3)
    plot_outline(ax_final, footprint, edgecolor=POLYGON_EDGE_COLOR, linewidth=1.0, zorder=5)
    ax_final.set_title("Grouped dwelling and core zones", pad=4)
    add_panel_label(ax_final, "(d)")

    legend_handles = [
        Line2D([0], [0], color=LEAF_EDGE_COLOR, lw=0.6, label="BSP leaf boundary"),
        Patch(facecolor=DWELLING_COLORS[0], edgecolor=POLYGON_EDGE_COLOR, label="Dwelling zone"),
        Patch(facecolor=CORE_COLOR, edgecolor=CORE_EDGE_COLOR, label="Shared core"),
    ]
    leg = ax_final.legend(handles=legend_handles, loc="lower center",
                          bbox_to_anchor=(0.5, -0.24), ncol=1,
                          frameon=True, framealpha=1.0, borderpad=0.4)
    leg.get_frame().set_edgecolor(LEGEND_EDGE_COLOR)
    leg.get_frame().set_linewidth(0.8)

    plt.tight_layout(w_pad=1.2, h_pad=1.6)
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    print(f"Saved PDF: {output_path}")

    if SHOW_AFTER_SAVE:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
