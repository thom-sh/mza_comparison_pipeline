#!/usr/bin/env python3
from pathlib import Path
import pickle
import re
import math

import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString, MultiPolygon
from shapely.ops import unary_union
from shapely import affinity

# ============================================================
# PATHS / OUTPUT
# ============================================================
PROJECT_DIR = Path(__file__).resolve().parent

# Repository root:
# .../mza_sensitivity_analysis
REPO_DIR = PROJECT_DIR.parent

PKL_PATH = REPO_DIR / "data" / "sa_building_data" / "building_data_merged.pkl"

OUT_DIR = PROJECT_DIR / "output" / "variants_plot"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# COMBINED_PNG_NAME = "variant_zoning_panel.png"
COMBINED_SVG_NAME = "variant_zoning_panel.pdf"

INDIVIDUAL_DIRNAME = "individual_variants"
SAVE_SVG_INDIVIDUAL = False

# ============================================================
# EXTRA DISPLAY ROTATIONS
# Base alignment:
#   core outer edge || x-axis
# Then:
#   all variants +180°, except V7 +90°
# ============================================================
EXTRA_ROTATION_BY_VARIANT = {
    "V1": 180.0,
    "V2": 180.0,
    "V3": 180.0,
    "V4": 270.0,
    "V5": 180.0,
    "V6": 270.0,
    "V7": 270.0,
    "V8": 180.0,
}

# ============================================================
# STYLE
# ============================================================
plt.rcParams.update({
    "font.family": "cmr10",
    "mathtext.fontset": "cm",

    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,

    "axes.titleweight": "normal",
    "axes.labelweight": "normal",

    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

PANEL_BG = "white"
OUTLINE = "#777d84"
COLORS = {
    "v1": "#DCEAF7",
    "res_1": "#DCEAF7",
    "res_2": "#AFCBE3",
    "res_3": "#7FA6C9",
    "res_4": "#DBDBDB",
    "core_unheated": "#9aa3ad",
    "core_heated": "#4B4B4B",
}

# DWELLING_COLORS = [
#     "#DCEAF7",  # Dwelling 1
#     "#AFCBE3",  # Dwelling 2
#     "#7FA6C9",  # Dwelling 3
#     "#BFC3C7",  # Dwelling 4
#     "#7E848A",  # Dwelling 5
# ]

# CORE_COLOR = "#4B4B4B"


VARIANT_ORDER = ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8"]


# ============================================================
# HELPERS
# ============================================================
def variant_from_building_id(building_id: str) -> str:
    suffix = str(building_id).strip().split("_")[-1]
    return f"V{suffix}"


def zone_index_from_name(name: str):
    m = re.search(r"Zone_(\d+)$", str(name).strip())
    return int(m.group(1)) if m else None


def variant_has_core_zone(vkey: str) -> bool:
    return vkey in {"V2", "V3", "V4", "V5", "V6", "V8"}


def is_core_zone(zone_name: str, vkey: str) -> bool:
    idx = zone_index_from_name(zone_name)
    return variant_has_core_zone(vkey) and idx == 1


def representative_storeys(building: dict):
    vkey = variant_from_building_id(building["building_id"])
    if vkey == "V1":
        return building["polygons"]["storeys"]
    return building["polygons"]["storeys"][:1]


def residential_fill(zone_name: str, vkey: str) -> str:
    idx = zone_index_from_name(zone_name)
    if idx is None:
        return COLORS["res_1"]

    if vkey in {"V3", "V4", "V7", "V8"}:
        return COLORS["res_1"] if idx % 2 == 0 else COLORS["res_2"]

    if vkey in {"V5", "V6"}:
        cycle = [COLORS["res_1"], COLORS["res_2"], COLORS["res_3"], COLORS["res_4"]]
        return cycle[(idx - 2) % 4] if idx >= 2 else cycle[0]

    if vkey == "V2":
        return COLORS["res_2"]

    return COLORS["v1"]


def xy_only(coords):
    out = []
    for c in coords:
        if len(c) >= 2:
            out.append((float(c[0]), float(c[1])))
    return out


def flatten_polygon(poly):
    ext = xy_only(list(poly.exterior.coords))
    holes = [xy_only(list(r.coords)) for r in poly.interiors]
    return Polygon(ext, holes)


def normalize_angle_to_x_axis(angle_deg: float) -> float:
    while angle_deg > 90:
        angle_deg -= 180
    while angle_deg <= -90:
        angle_deg += 180
    return angle_deg


def segment_angle(seg: LineString) -> float:
    (x1, y1), (x2, y2) = list(seg.coords)
    return math.degrees(math.atan2(y2 - y1, x2 - x1))


def polygon_segments(poly: Polygon):
    coords = list(poly.exterior.coords)
    for a, b in zip(coords[:-1], coords[1:]):
        yield LineString([a, b])


def longest_segment(poly):
    best = None
    best_len = -1.0
    for seg in polygon_segments(poly):
        L = seg.length
        if L > best_len:
            best = seg
            best_len = L
    return best


def outer_core_edge_segment(core_poly: Polygon, building_union):
    boundary = building_union.boundary
    best = None
    best_score = -1.0

    for seg in polygon_segments(core_poly):
        if seg.length == 0:
            continue
        overlap = seg.intersection(boundary).length
        ratio = overlap / seg.length if seg.length > 0 else 0.0
        score = overlap + 1e-6 * seg.length
        if ratio > 0.90 and score > best_score:
            best = seg
            best_score = score

    if best is not None:
        return best

    for seg in polygon_segments(core_poly):
        if seg.length == 0:
            continue
        overlap = seg.intersection(boundary).length
        score = overlap + 1e-6 * seg.length
        if score > best_score:
            best = seg
            best_score = score

    return best


def explode_polygons(geom):
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    return []


def collect_zone_geometries(building):
    vkey = variant_from_building_id(building["building_id"])
    items = []
    flat_geoms = []

    for storey in representative_storeys(building):
        for zone in storey.get("zones", []) or []:
            zone_name = str(zone.get("name", "")).strip()

            if zone_name == "Zone_Complete_Building" and vkey != "V1":
                continue

            floors = zone.get("floors", []) or []
            if not floors:
                continue

            geom_2d = flatten_polygon(floors[0][0])
            items.append((zone_name, geom_2d))
            flat_geoms.append(geom_2d)

    return items, flat_geoms


def oriented_geometries(building):
    vkey = variant_from_building_id(building["building_id"])
    items, flat_geoms = collect_zone_geometries(building)

    if not flat_geoms:
        return []

    building_union = unary_union(flat_geoms)
    origin = building_union.centroid

    core_poly = None
    for zone_name, geom in items:
        if is_core_zone(zone_name, vkey):
            core_poly = geom
            break

    if core_poly is not None:
        ref_seg = outer_core_edge_segment(core_poly, building_union)
    else:
        ref_parts = explode_polygons(building_union)
        ref_seg = longest_segment(ref_parts[0]) if ref_parts else None

    base_angle = 0.0 if ref_seg is None else normalize_angle_to_x_axis(segment_angle(ref_seg))
    extra_angle = float(EXTRA_ROTATION_BY_VARIANT.get(vkey, 180.0))
    total_angle = -base_angle + extra_angle

    return [
        (zone_name, affinity.rotate(geom, total_angle, origin=origin))
        for zone_name, geom in items
    ]


def plot_poly(ax, geom, fill):
    ext = xy_only(list(geom.exterior.coords))
    ax.fill(
        [c[0] for c in ext],
        [c[1] for c in ext],
        facecolor=fill,
        edgecolor=OUTLINE,
        linewidth=1.2,
        joinstyle="round",
        zorder=2,
    )

    for ring in geom.interiors:
        hole = xy_only(list(ring.coords))
        ax.fill(
            [c[0] for c in hole],
            [c[1] for c in hole],
            facecolor=PANEL_BG,
            edgecolor=OUTLINE,
            linewidth=0.8,
            zorder=3,
        )


def plot_building_panel(ax, building):
    vkey = variant_from_building_id(building["building_id"])
    rotated_items = oriented_geometries(building)
    geoms_for_bounds = []

    for zone_name, geom in rotated_items:
        geoms_for_bounds.append(geom)

        if is_core_zone(zone_name, vkey):
            fill = COLORS["core_heated"] if vkey == "V8" else COLORS["core_unheated"]
        else:
            fill = residential_fill(zone_name, vkey)

        plot_poly(ax, geom, fill)

    if geoms_for_bounds:
        merged = unary_union(geoms_for_bounds)
        minx, miny, maxx, maxy = merged.bounds
        span = max(maxx - minx, maxy - miny)

        pad_x = 0.12 * span
        
        pad_bottom = 0.09 * span
        pad_top = 0.30 * span

        ax.set_xlim(minx - pad_x, maxx + pad_x)
        ax.set_ylim(miny - pad_bottom, maxy + pad_top)

    ax.set_aspect("equal")
    ax.set_facecolor(PANEL_BG)
    ax.axis("off")


def draw_aligned_row_labels(fig, axes, variant_order):
    row_specs = {}

    for ax, vkey in zip(axes.flat, variant_order):
        bbox = ax.get_position()
        ss = ax.get_subplotspec()
        row = ss.rowspan.start
        row_specs.setdefault(row, []).append(
            (bbox.x0 + bbox.width / 2.0, bbox.y0, vkey)
        )

    for row, items in row_specs.items():
        y = min(item[1] for item in items) - 0.001
        for x, _, vkey in items:
            fig.text(
                x, y, vkey,
                ha="center",
                va="top",
                fontsize=22,
                fontweight="normal",
                family="serif",
            )


def save_individual_variant(building, out_dir: Path):
    vkey = variant_from_building_id(building["building_id"])

    fig, ax = plt.subplots(figsize=(6.14, 3.8))
    fig.patch.set_facecolor(PANEL_BG)
    plot_building_panel(ax, building)

    plt.tight_layout(rect=(0.03, 0.08, 0.97, 0.97))

    bbox = ax.get_position()
    fig.text(
        bbox.x0 + bbox.width / 2.0,
        bbox.y0 - 0.015,
        vkey,
        ha="center",
        va="top",
        fontsize=10,
        fontweight="normal",
        family="serif",
    )

    fig.savefig(
        out_dir / f"{vkey}_zone_layout.png",
        dpi=250,
        facecolor=fig.get_facecolor(),
        bbox_inches="tight",
    )

    if SAVE_SVG_INDIVIDUAL:
        fig.savefig(
            out_dir / f"{vkey}_zone_layout.pdf",
            facecolor=fig.get_facecolor(),
            bbox_inches="tight",
        )

    # plt.show()
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    individual_dir = OUT_DIR 
    individual_dir.mkdir(parents=True, exist_ok=True)

    with open(PKL_PATH, "rb") as f:
        buildings = pickle.load(f)

    buildings_by_variant = {
        variant_from_building_id(b["building_id"]): b
        for b in buildings
    }

    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    fig.patch.set_facecolor(PANEL_BG)

    for ax, vkey in zip(axes.flat, VARIANT_ORDER):
        if vkey in buildings_by_variant:
            plot_building_panel(ax, buildings_by_variant[vkey])
        else:
            ax.axis("off")

    plt.tight_layout(rect=(0.03, 0.08, 0.97, 0.97))
    draw_aligned_row_labels(fig, axes, VARIANT_ORDER)

    # fig.savefig(
    #     OUT_DIR / COMBINED_PNG_NAME,
    #     dpi=250,
    #     facecolor=fig.get_facecolor(),
    #     bbox_inches="tight",
    # )
    fig.savefig(
        OUT_DIR / COMBINED_SVG_NAME,
        facecolor=fig.get_facecolor(),
        bbox_inches="tight",
    )
    plt.show()
    plt.close(fig)

    for vkey in VARIANT_ORDER:
        if vkey in buildings_by_variant:
            save_individual_variant(buildings_by_variant[vkey], individual_dir)

    print(f"Saved combined outputs in: {OUT_DIR}")
    print(f"Saved individual outputs in: {individual_dir}")


if __name__ == "__main__":
    main()