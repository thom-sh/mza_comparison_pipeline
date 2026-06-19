#!/usr/bin/env python3
"""
Scientific methodology figure for one sample MSD building.

Layout:
- Top row: (a), (b), room legend
- Bottom row: (c), (d), dwelling legend

Workflow:
(a) MSD room graph
(b) Room graph after entrance-edge removal
(c) Partition after entrance-edge removal
(d) Gap-filled dwelling and fixed core zones

This script focuses on:
- figure layout
- colors
- legends
- plotting

All processing logic is imported from msd_processing.py.
"""

from pathlib import Path
import sys

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from shapely.geometry import Polygon


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from msd_processing import (
    load_graph,
    remove_auxiliary_rooms,
    detect_apartments_and_core_nodes,
    split_by_entrances,
    extract_apartment_polygons,
    extract_core_union_from_nodes,
    extract_building_footprint_from_apts_and_core,
    simultaneous_apartment_growth,
    polygon_parts,
    safe_polygon,
    get_type_sets,
)



# -----------------------------------------------------------------------------
# Global plot style
# -----------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "axes.titleweight": "normal",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


# -----------------------------------------------------------------------------
# Constants for figure appearance
# -----------------------------------------------------------------------------
ROOM_NAMES = [
    "Bedroom",
    "Livingroom",
    "Kitchen",
    "Dining",
    "Corridor",
    "Stairs",
    "Storeroom",
    "Bathroom",
    "Balcony",
    "Structure",
    "Door",
    "Entrance Door",
    "Window",
]

ROOM_COLORS = {
    "Bedroom": "#5B9BD5",
    "Livingroom": "#ED7D31",
    "Kitchen": "#2CA6A4",
    "Dining": "#F4B183",
    "Corridor": "#A56716",
    "Stairs": "#8E3B9C",
    "Storeroom": "#FFD966",
    "Bathroom": "#7F7FCE",
    "Balcony": "#6ABD5B",
    "Structure": "#000000",
    "Door": "#FFC000",
    "Entrance Door": "#500204",
    "Window": "#D62728",
}

DWELLING_COLORS = [
    "#DCEAF7",
    "#AFCBE3",
    "#7FA6C9",
    "#DBDBDB",
    "#BABABA",
    "#999999",
]

    # "#BFC3C7",
    # "#7E848A",
    # "#5F666D",

CORE_COLOR = "#5F666D"


@dataclass
class WorkflowResult:
    graph_original: nx.Graph
    graph_processed: nx.Graph
    split_graph: nx.Graph
    apartments: List[Set[int]]
    core_nodes: Set[int]
    apartment_polygons_raw: List
    apartment_polygons_filled: List
    core_polygons: List
    footprint: object
    residual_gap: object | None
    entrance_edges_original: List[Tuple[int, int]]


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------
def entrance_edges(graph: nx.Graph) -> List[Tuple[int, int]]:
    return [
        (u, v)
        for u, v, d in graph.edges(data=True)
        if d.get("connectivity") == "entrance"
    ]


def node_position(graph: nx.Graph) -> Dict[int, Tuple[float, float]]:
    pos: Dict[int, Tuple[float, float]] = {}
    for n, d in graph.nodes(data=True):
        c = d.get("centroid")
        if c is None or len(c) < 2:
            continue
        pos[n] = (float(c[0]), float(c[1]))
    return pos


def room_color(room_type: int) -> str:
    name = (
        ROOM_NAMES[room_type]
        if isinstance(room_type, int) and 0 <= room_type < len(ROOM_NAMES)
        else "Structure"
    )
    return ROOM_COLORS.get(name, "#BBBBBB")


def draw_room_polygons(
    ax,
    graph: nx.Graph,
    alpha: float = 0.78,
    edgecolor: str = "#777d84",
    linewidth: float = None,
):
    for _, d in graph.nodes(data=True):
        poly = safe_polygon(d.get("geometry"))
        if poly is None:
            continue
        x, y = poly.exterior.xy
        ax.fill(
            x,
            y,
            facecolor=room_color(d.get("room_type", 9)),
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=alpha,
        )


def draw_union(
    ax,
    geom,
    facecolor: str,
    edgecolor: str = "white",
    linewidth: float = None,
    alpha: float = 0.95,
):
    for g in polygon_parts(geom):
        x, y = g.exterior.xy
        ax.fill(
            x,
            y,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=alpha,
        )


def component_color_map(
    apartments: List[Set[int]],
    core_nodes: Set[int],
) -> Dict[int, str]:
    colors: Dict[int, str] = {}

    for i, comp in enumerate(apartments):
        fill = DWELLING_COLORS[i % len(DWELLING_COLORS)]
        for node in comp:
            colors[node] = fill

    for node in core_nodes:
        colors[node] = CORE_COLOR

    return colors


def set_consistent_extent(axs, geoms):
    valid = [g for g in geoms if g is not None and not g.is_empty]
    if not valid:
        return

    minx = min(g.bounds[0] for g in valid)
    miny = min(g.bounds[1] for g in valid)
    maxx = max(g.bounds[2] for g in valid)
    maxy = max(g.bounds[3] for g in valid)

    padx = 0.01 * (maxx - minx)
    pady = 0.01 * (maxy - miny)

    for ax in axs:
        ax.set_xlim(minx - padx, maxx + padx)
        ax.set_ylim(miny - pady, maxy + pady)
        ax.set_aspect("equal")
        ax.axis("off")


def add_panel_label(ax, label: str):
    ax.text(
        0.5,
        -0.001,
        label,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontweight="normal",
    )


# -----------------------------------------------------------------------------
# Workflow wrapper
# -----------------------------------------------------------------------------
def run_workflow(datapath: str, building_id: int) -> WorkflowResult:
    original = load_graph(datapath, building_id)

    processed = original.copy()

    _, private_types, auxiliary_types = get_type_sets()
    remove_auxiliary_rooms(processed, auxiliary_types)

    apartments, core_nodes = detect_apartments_and_core_nodes(
        processed,
        private_types,
    )

    split = split_by_entrances(processed)

    apartment_polys_raw = extract_apartment_polygons(
        processed,
        apartments,
        auxiliary_types,
        buffer_amt=0.08,
    )

    core_polygons = extract_core_union_from_nodes(
        processed,
        core_nodes,
        buffer_amt=0.15,
    )

    footprint = extract_building_footprint_from_apts_and_core(
        apartment_polygons=apartment_polys_raw,
        core_polygons=core_polygons,
        outer_buffer=0.45,
        inner_buffer=-0.35,
        simplify_tol=0.03,
    )

    apartment_polys_filled, residual_gap = simultaneous_apartment_growth(
        apartment_polygons=apartment_polys_raw,
        core_polygons=core_polygons,
        footprint=footprint,
        step=0.03,
        max_iter=400,
        min_residual_area=1e-4,
        simplify_tol=0.02,
    )

    return WorkflowResult(
        graph_original=original,
        graph_processed=processed,
        split_graph=split,
        apartments=apartments,
        core_nodes=core_nodes,
        apartment_polygons_raw=apartment_polys_raw,
        apartment_polygons_filled=apartment_polys_filled,
        core_polygons=core_polygons,
        footprint=footprint,
        residual_gap=residual_gap,
        entrance_edges_original=entrance_edges(processed),
    )


# -----------------------------------------------------------------------------
# Main figure
# -----------------------------------------------------------------------------
def plot_methodology_figure(
    result: WorkflowResult,
    building_id: int,
    output_path: str,
    dpi: int = 300,
    show: bool = True,
):
    fig, axs = plt.subplots(
        2, 3,
        figsize=(5.5, 3.5),
        gridspec_kw={"width_ratios": [1, 1, 0.5]}
    )
    axs = axs.flatten()

    ax_a = axs[0]
    ax_b = axs[1]
    room_legend_ax = axs[2]
    ax_c = axs[3]
    ax_d = axs[4]
    dwelling_legend_ax = axs[5]

    # ------------------------------------------------------------------
    # (a) MSD room graph
    # ------------------------------------------------------------------
    draw_room_polygons(ax_a, result.graph_original, alpha=0.76)
    pos = node_position(result.graph_original)

    entrance_set = set(tuple(sorted(e)) for e in result.entrance_edges_original)
    non_entrance = [
        e for e in result.graph_original.edges()
        if tuple(sorted(e)) not in entrance_set
    ]

    nx.draw_networkx_edges(
        result.graph_original,
        pos,
        edgelist=non_entrance,
        ax=ax_a,
        edge_color="#8B8B8B",
        width=1.2,
        alpha=0.80,
    )
    nx.draw_networkx_edges(
        result.graph_original,
        pos,
        edgelist=result.entrance_edges_original,
        ax=ax_a,
        edge_color="#D62728",
        width=1.5,
        alpha=0.95,
    )
    nx.draw_networkx_nodes(
        result.graph_original,
        pos,
        ax=ax_a,
        node_color="white",
        edgecolors="#777d84",
        node_size=6,
        linewidths=None,
    )
    add_panel_label(ax_a, "(a)")

    # ------------------------------------------------------------------
    # (b) Room graph after entrance-edge removal
    # ------------------------------------------------------------------
    draw_room_polygons(ax_b, result.graph_processed, alpha=0.76)
    split_pos = node_position(result.graph_processed)

    nx.draw_networkx_edges(
        result.split_graph,
        split_pos,
        ax=ax_b,
        edge_color="#8B8B8B",
        width=1.2,
        alpha=0.80,
    )
    nx.draw_networkx_nodes(
        result.graph_processed,
        split_pos,
        ax=ax_b,
        node_color="white",
        edgecolors="#777d84",
        node_size=6,
        linewidths=None,
    )
    add_panel_label(ax_b, "(b)")

    # ------------------------------------------------------------------
    # Top-right room legend
    # ------------------------------------------------------------------
    room_legend_ax.axis("off")
    room_handles = [
        Line2D([0], [0], color="#D62728", lw=2.0, label="Entrance edge"),
        Line2D([0], [0], color="#9E9E9E", lw=1.4, label="Other connectivity"),
    ] + [
        Patch(facecolor=ROOM_COLORS[name], edgecolor=None, label=name)
        for name in [
            "Bedroom",
            "Livingroom",
            "Kitchen",
            "Dining",
            "Bathroom",
            "Corridor",
            "Stairs",
            "Balcony",
        ]
    ]

    room_legend_ax.legend(
        handles=room_handles,
        loc="upper left",
        bbox_to_anchor=(-0.15, 0.95),
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor="#bdc1c5",
        facecolor="white",
        fontsize=7.5,
        handlelength=1.4,
        handletextpad=0.5,
        borderpad=0.5,
        labelspacing=0.35,
        columnspacing=1.2,
        ncol=1,
    )

    # ------------------------------------------------------------------
    # (c) Partition after entrance-edge removal
    # ------------------------------------------------------------------
    split_colors = component_color_map(result.apartments, result.core_nodes)

    for n, d in result.graph_processed.nodes(data=True):
        poly = safe_polygon(d.get("geometry"))
        if poly is None:
            continue
        x, y = poly.exterior.xy
        ax_c.fill(
            x,
            y,
            facecolor=split_colors.get(n, "#CCCCCC"),
            edgecolor="#777d84",
            linewidth=None,
            alpha=0.92,
        )
    add_panel_label(ax_c, "(c)")

    # ------------------------------------------------------------------
    # (d) Gap-filled dwelling and fixed core zones
    # ------------------------------------------------------------------
    for i, apt in enumerate(result.apartment_polygons_filled):
        draw_union(
            ax_d,
            apt,
            facecolor=DWELLING_COLORS[i % len(DWELLING_COLORS)],
            edgecolor="#777d84",
            linewidth=None,
            alpha=0.96,
        )

    for core in result.core_polygons:
        draw_union(
            ax_d,
            core,
            facecolor=CORE_COLOR,
            edgecolor="#777d84",
            linewidth=None,
            alpha=0.96,
        )

    add_panel_label(ax_d, "(d)")

    # ------------------------------------------------------------------
    # Bottom-right dwelling legend
    # ------------------------------------------------------------------
    dwelling_legend_ax.axis("off")
    dwelling_handles = [
        Patch(
            facecolor=DWELLING_COLORS[i % len(DWELLING_COLORS)],
            edgecolor=None,
            label=f"Dwelling {i + 1}",
        )
        for i in range(len(result.apartment_polygons_filled))
    ]
    dwelling_handles.append(
        Patch(facecolor=CORE_COLOR, edgecolor=None, label="Stairwell")
    )

    dwelling_legend_ax.legend(
        handles=dwelling_handles,
        loc="upper left",
        bbox_to_anchor=(-0.15, 0.95),
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor="#bdc1c5",
        facecolor="white",
        fontsize=7.5,
        handlelength=1.4,
        handletextpad=0.5,
        borderpad=0.5,
        labelspacing=0.35,
        ncol=1,
    )

    # ------------------------------------------------------------------
    # Consistent extents for plot panels only
    # ------------------------------------------------------------------
    all_geoms = []
    for _, d in result.graph_original.nodes(data=True):
        poly = safe_polygon(d.get("geometry"))
        if poly is not None:
            all_geoms.append(poly)

    all_geoms.extend([g for g in result.apartment_polygons_filled if g is not None])
    
    all_geoms.extend([g for g in result.core_polygons if g is not None])

    set_consistent_extent([ax_a, ax_b, ax_c, ax_d], all_geoms)

    fig.subplots_adjust(
        left=0.05,
        right=0.93,
        top=0.97,
        bottom=0.06,
        wspace=0.15,
        hspace=0.15,
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)

    if show:
        plt.show()

    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:

    # Parent folder

    PROJECT_DIR = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(PROJECT_DIR))

    datapath = PROJECT_DIR / "data" / "raw_msd"
    BUILDING_IDS = [75]

    for building_id in BUILDING_IDS:
        output = rf"C:\WF\Thomas Sharon\Master_Thesis_Report\figures\msd_methodology\msd_methodology_building_{building_id}.pdf"
        dpi = 300
        show = True

        result = run_workflow(datapath, building_id)

        plot_methodology_figure(
            result,
            building_id,
            output,
            dpi=dpi,
            show=show,
        )

        print(f"Saved methodology figure to: {output}")
        print(f"Building ID: {building_id}")
        print(f"Detected dwellings: {len(result.apartments)}")
        print(f"Core nodes: {len(result.core_nodes)}")
        if result.residual_gap is None or result.residual_gap.is_empty:
            print("Gap filling successful: zoning is footprint-complete.")
        else:
            print(f"Residual gap area still present: {result.residual_gap.area:.6f}")


if __name__ == "__main__":
    main()