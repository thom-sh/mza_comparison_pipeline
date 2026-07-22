"""
Create a two-panel thesis figure for the CityGML footprint replacement step.

This version:
(a) reads and plots the oriented footprint from a GeoJSON file,
(b) reads and displays a PNG screenshot of the replaced CityGML/LoD2 building.

Edit only the CONFIGURATION block in main().
"""

import os
import json
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection


# ============================================================
#                      GEOJSON LOADING
# ============================================================

def load_geojson_polygon(geojson_path: str) -> np.ndarray:
    """
    Load the first Polygon or MultiPolygon footprint from a GeoJSON file.
    Returns an (N, 2) array of x, y coordinates.
    """
    with open(geojson_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Supports FeatureCollection, Feature, or direct geometry GeoJSON
    if data.get("type") == "FeatureCollection":
        features = data.get("features", [])
        if not features:
            raise ValueError("No features found in GeoJSON file.")
        geom = features[0]["geometry"]
    elif data.get("type") == "Feature":
        geom = data["geometry"]
    else:
        geom = data

    gtype = geom["type"]

    if gtype == "Polygon":
        coords = geom["coordinates"][0]
    elif gtype == "MultiPolygon":
        coords = geom["coordinates"][0][0]
    else:
        raise ValueError(f"Unsupported GeoJSON geometry type: {gtype}")

    arr = np.array(coords, dtype=float)

    if arr.shape[1] < 2:
        raise ValueError("GeoJSON coordinates do not contain x and y values.")

    return arr[:, :2]


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

def add_panel_label_below(ax, text, y_offset=-0.28):
    ax.text(
        0.5,
        y_offset,
        text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
    )

def plot_geojson_panel(ax, polygon_xy: np.ndarray):
    """
    Plot the oriented external footprint from GeoJSON.
    """
    patch = MplPolygon(polygon_xy[:, :2], closed=True)
    collection = PatchCollection(
        [patch],
        facecolor="#F7F8F9",
        edgecolor="#777d84",
        linewidth=1.5,
        alpha=0.35,
    )
    ax.add_collection(collection)

    ax.plot(
    polygon_xy[:, 0],
    polygon_xy[:, 1],
    color="#777d84",
    linewidth=1.5,
)

    cx = polygon_xy[:, 0].mean()
    cy = polygon_xy[:, 1].mean()
    # ax.scatter([cx], [cy], marker="x", s=50, label="Footprint centroid")

    set_equal_xy(ax, polygon_xy)
    ax.grid(True, linestyle="--", alpha=0.35)
    # ax.set_xlabel("x [m]")
    # ax.set_ylabel("y [m]")
    # Panel label below the plot
    add_panel_label_below(ax, "(a)")
    # ax.legend(loc="best", fontsize=8)


def plot_png_panel(ax, png_path: str):
    """
    Display a PNG screenshot without stretching.
    The outer axis box is controlled outside this function.
    """
    if not os.path.exists(png_path):
        raise FileNotFoundError(f"PNG screenshot not found: {png_path}")

    img = plt.imread(png_path)

    ax.imshow(img, aspect="equal")
    ax.axis("on")

    add_panel_label_below(ax, "(b)", y_offset=-0.28)


def create_two_panel_figure(
    geojson_path: str,
    png_path: str,
    output_path: str,
    figure_title: Optional[str] = None,
):
    """
    Create and save the two-panel figure.
    """
    polygon_xy = load_geojson_polygon(geojson_path)

    plt.rcParams["font.family"] = "cmr10"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.size"] = 11
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(6.14, 2.4),
        gridspec_kw={"width_ratios": [1, 1]},
    )

    plot_geojson_panel(axes[0], polygon_xy)

    # Match the physical subplot shape of panel (b) to panel (a)
    minx, miny = polygon_xy[:, 0].min(), polygon_xy[:, 1].min()
    maxx, maxy = polygon_xy[:, 0].max(), polygon_xy[:, 1].max()
    dx = maxx - minx
    dy = maxy - miny
    box_aspect = dy / dx if dx != 0 else 1.0

    axes[0].set_box_aspect(box_aspect)
    axes[1].set_box_aspect(box_aspect)

    plot_png_panel(axes[1], png_path)

    # if figure_title:
        # fig.suptitle(figure_title, fontsize=13, y=0.98)

    fig.tight_layout(rect=[0, 0.08, 1, 1])

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print(f"Saved figure to: {output_path}")


# ============================================================
#                            MAIN
# ============================================================

def main():
    # ========================================================
    # CONFIGURATION: edit these paths only
    # ========================================================
    # gml_footprint_replacement/figures/scripts/
    PROJECT_DIR = Path(__file__).resolve().parents[1]

    BUILDING_ID = 75
    DATASET = "msd"

    GEOJSON_PATH = PROJECT_DIR / "data" / DATASET / "footprint" / f"footprint_{BUILDING_ID}.geojson"

    PNG_PATH = PROJECT_DIR / "figures" / "data" / f"citygml_{BUILDING_ID}.png"

    OUTPUT_PATH = PROJECT_DIR / "figures" / "output" / f"citygml_replacement_workflow_{BUILDING_ID}.pdf"

    FIGURE_TITLE = f"CityGML footprint replacement workflow — Building {BUILDING_ID}"
    # ========================================================

    create_two_panel_figure(
        geojson_path=str(GEOJSON_PATH),
        png_path=str(PNG_PATH),
        output_path=str(OUTPUT_PATH),
        figure_title=FIGURE_TITLE,
    )


if __name__ == "__main__":
    main()
