import os
import json
import math
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from lxml import etree


# ============================================================
#                         CONFIG HELPERS
# ============================================================

NS = {
    "gml": "http://www.opengis.net/gml",
    "bldg": "http://www.opengis.net/citygml/building/2.0",
    "core": "http://www.opengis.net/citygml/2.0",
}


# ============================================================
#                      GEOJSON LOADING
# ============================================================

def load_geojson_polygon(geojson_path: str) -> np.ndarray:
    """
    Load the first polygon from a GeoJSON footprint file.
    Returns an (N, 2) array of x, y coordinates.
    """
    with open(geojson_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    features = data.get("features", [])
    if not features:
        raise ValueError("No features found in GeoJSON file.")

    geom = features[0]["geometry"]
    gtype = geom["type"]

    if gtype == "Polygon":
        coords = geom["coordinates"][0]
    elif gtype == "MultiPolygon":
        coords = geom["coordinates"][0][0]
    else:
        raise ValueError(f"Unsupported GeoJSON geometry type: {gtype}")

    arr = np.array(coords, dtype=float)

    if arr.shape[1] >= 2:
        return arr[:, :2]

    raise ValueError("GeoJSON coordinates do not contain x and y values.")


# ============================================================
#                    CITYGML PARSING
# ============================================================

def parse_poslist_to_xyz(poslist_text: str) -> np.ndarray:
    """
    Convert a gml:posList text block into an (N, 3) array.
    """
    values = [float(v) for v in poslist_text.strip().split()]
    if len(values) % 3 != 0:
        raise ValueError("gml:posList does not contain 3D coordinates.")
    return np.array(values, dtype=float).reshape((-1, 3))


def find_building_element(tree: etree._ElementTree, target_id: Optional[str] = None):
    """
    Return the selected building element.
    If target_id is None, returns the first bldg:Building found.
    """
    buildings = tree.xpath("//bldg:Building", namespaces=NS)
    if not buildings:
        raise ValueError("No bldg:Building elements found in the CityGML file.")

    if target_id is None:
        return buildings[0]

    for b in buildings:
        gid = b.get("{http://www.opengis.net/gml}id")
        if gid == target_id:
            return b

    raise ValueError(f"Could not find building with gml:id = {target_id}")


def extract_surface_polygons(building_el) -> List[Tuple[str, np.ndarray]]:
    """
    Extract surface polygons from a CityGML building.

    Returns a list of tuples:
        (surface_type, coords_xyz)

    where surface_type is one of:
        "GroundSurface", "RoofSurface", "WallSurface", or "Other"
    """
    surfaces = []

    # Find all boundary surfaces inside the selected building
    bounded_surfaces = building_el.xpath(".//bldg:boundedBy/*", namespaces=NS)

    for surf in bounded_surfaces:
        tag = etree.QName(surf.tag).localname

        polygons = surf.xpath(".//gml:Polygon", namespaces=NS)
        for poly in polygons:
            poslists = poly.xpath(".//gml:exterior//gml:posList", namespaces=NS)
            for pos in poslists:
                coords = parse_poslist_to_xyz(pos.text)
                surfaces.append((tag, coords))

    return surfaces


# ============================================================
#                         PLOTTING
# ============================================================

def set_equal_xy(ax, coords: np.ndarray, margin_ratio: float = 0.10):
    minx, miny = coords[:, 0].min(), coords[:, 1].min()
    maxx, maxy = coords[:, 0].max(), coords[:, 1].max()

    dx = maxx - minx
    dy = maxy - miny
    margin = max(dx, dy) * margin_ratio

    ax.set_xlim(minx - margin, maxx + margin)
    ax.set_ylim(miny - margin, maxy + margin)
    ax.set_aspect("equal", adjustable="box")


def set_equal_3d(ax, xyz_all: np.ndarray, margin_ratio: float = 0.10):
    minx, miny, minz = xyz_all.min(axis=0)
    maxx, maxy, maxz = xyz_all.max(axis=0)

    dx = maxx - minx
    dy = maxy - miny
    dz = maxz - minz
    max_range = max(dx, dy, dz)
    margin = max_range * margin_ratio

    cx = 0.5 * (minx + maxx)
    cy = 0.5 * (miny + maxy)
    cz = 0.5 * (minz + maxz)

    ax.set_xlim(cx - max_range / 2 - margin, cx + max_range / 2 + margin)
    ax.set_ylim(cy - max_range / 2 - margin, cy + max_range / 2 + margin)
    ax.set_zlim(cz - max_range / 2 - margin, cz + max_range / 2 + margin)


def plot_geojson_panel(ax, polygon_xy: np.ndarray):
    patches = [MplPolygon(polygon_xy[:, :2], closed=True)]
    collection = PatchCollection(
        patches,
        facecolor="#d9d9d9",
        edgecolor="black",
        linewidth=1.5
    )
    ax.add_collection(collection)

    cx = polygon_xy[:, 0].mean()
    cy = polygon_xy[:, 1].mean()
    ax.scatter([cx], [cy], marker="x", s=50, color="black", label="Footprint centroid")

    set_equal_xy(ax, polygon_xy)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("(a) Oriented footprint exported as GeoJSON")
    ax.legend(loc="best", fontsize=8)


def plot_citygml_panel(ax, surfaces: List[Tuple[str, np.ndarray]]):
    facecolors = {
        "GroundSurface": "#cfcfcf",
        "RoofSurface": "#b0b0b0",
        "WallSurface": "#e0e0e0",
        "Other": "#d9d9d9",
    }

    all_xyz = []

    for surf_type, coords in surfaces:
        all_xyz.append(coords)

        verts = [coords[:, :3]]
        poly = Poly3DCollection(
            verts,
            facecolor=facecolors.get(surf_type, facecolors["Other"]),
            edgecolor="black",
            linewidths=0.6,
            alpha=0.95,
        )
        ax.add_collection3d(poly)

    if not all_xyz:
        raise ValueError("No surface polygons found for CityGML plotting.")

    xyz_all = np.vstack(all_xyz)
    set_equal_3d(ax, xyz_all)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("(b) CityGML building after footprint replacement")

    # Adjust viewpoint for a clean thesis-style snapshot
    ax.view_init(elev=24, azim=-58)


def create_two_panel_figure(
    geojson_path: str,
    gml_path: str,
    output_path: str,
    target_gml_id: Optional[str] = None,
    figure_title: str = "",
):
    # Load GeoJSON footprint
    polygon_xy = load_geojson_polygon(geojson_path)

    # Load and parse CityGML
    tree = etree.parse(gml_path)
    building_el = find_building_element(tree, target_gml_id)
    surfaces = extract_surface_polygons(building_el)

    # Matplotlib styling
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 11

    fig = plt.figure(figsize=(13, 6.3))

    ax1 = fig.add_subplot(1, 2, 1)
    plot_geojson_panel(ax1, polygon_xy)

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    plot_citygml_panel(ax2, surfaces)

    if figure_title:
        fig.suptitle(figure_title, fontsize=13, y=0.98)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print(f"Saved figure to: {output_path}")


# ============================================================
#                            MAIN
# ============================================================

def main():
    # ========================================================
    # EDIT THESE PATHS ONLY
    # ========================================================
    GEOJSON_PATH = r"C:\WF\Thomas Sharon\Floorplan_Dataset\gml_msd\footprint\footprint_75.geojson"
    GML_PATH = r"C:\WF\Thomas Sharon\Floorplan_Dataset\gml_msd_trial\output\building_75_replaced.gml"
    OUTPUT_PATH = r"C:\WF\Thomas Sharon\Floorplan_Dataset\gml_rd\citygml_replacement_workflow_75.pdf"

    # Optional: set to None if there is only one building in the file
    TARGET_GML_ID = None
    FIGURE_TITLE = "CityGML footprint replacement workflow — Building 75"
    # ========================================================

    create_two_panel_figure(
        geojson_path=GEOJSON_PATH,
        gml_path=GML_PATH,
        output_path=OUTPUT_PATH,
        target_gml_id=TARGET_GML_ID,
        figure_title=FIGURE_TITLE,
    )


if __name__ == "__main__":
    main()