from pathlib import Path
import pickle

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from shapely.ops import unary_union


# ============================================================
# Hard-coded inputs
# ============================================================
PICKLE_PATH = Path(r"C:\WF\Thomas Sharon\Floorplan_Dataset\msd_pickle\75.pickle")
OUTPUT_DIR = Path(r"C:\WF\Thomas Sharon\Floorplan_Dataset\figures")
OUTPUT_NAME = "msd75_bsp_tree_from_pickle"


# ============================================================
# Plot helpers
# ============================================================
def iter_polygons(geom):
    if geom is None or geom.is_empty:
        return
    if geom.geom_type == "Polygon":
        yield geom
    elif geom.geom_type == "MultiPolygon":
        yield from geom.geoms
    elif hasattr(geom, "geoms"):
        for g in geom.geoms:
            yield from iter_polygons(g)


def draw_tree(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("BSP tree", fontsize=11, pad=2)

    # Schematic BSP hierarchy, using only the labels from the reference figure.
    pos = {
        "X": (0.50, 0.88),
        "Y": (0.32, 0.54),
        "C": (0.68, 0.54),
        "D": (0.20, 0.20),
        "E": (0.43, 0.20),
    }
    edges = [("X", "Y"), ("X", "C"), ("Y", "D"), ("Y", "E")]

    r = 0.065

    for a, b in edges:
        x1, y1 = pos[a]
        x2, y2 = pos[b]
        ax.plot(
            [x1, x2],
            [y1, y2],
            color="black",
            lw=0.9,
            solid_capstyle="round",
            zorder=1,
        )

    for lab, (x, y) in pos.items():
        circle = Circle(
            (x, y),
            radius=r,
            facecolor="white",
            edgecolor="black",
            lw=0.9,
            zorder=2,
        )
        ax.add_patch(circle)
        ax.text(
            x,
            y,
            lab,
            ha="center",
            va="center",
            fontsize=9,
            family="serif",
            zorder=3,
        )


def make_letter_mapping(floor_plan):
    """
    Use only C, D, and E as final partition labels.

    C is assigned to the core/stair polygon if room_type == 1 exists.
    D and E are assigned to the two dwelling polygons from top to bottom.
    """
    polygons = [item["polygon"] for item in floor_plan]
    room_types = [item.get("room_type") for item in floor_plan]

    core_indices = [i for i, t in enumerate(room_types) if int(t) == 1]
    dwelling_indices = [i for i, t in enumerate(room_types) if int(t) == 0]
    dwelling_indices = sorted(
        dwelling_indices,
        key=lambda i: polygons[i].centroid.y,
        reverse=True,
    )

    label_by_index = {}

    if core_indices:
        label_by_index[core_indices[0]] = "C"

    if len(dwelling_indices) >= 1:
        label_by_index[dwelling_indices[0]] = "D"

    if len(dwelling_indices) >= 2:
        label_by_index[dwelling_indices[1]] = "E"

    # Fallback for unusual structures
    for i in range(len(polygons)):
        if i not in label_by_index:
            for lab in ["C", "D", "E"]:
                if lab not in label_by_index.values():
                    label_by_index[i] = lab
                    break

    return label_by_index


def draw_plan(ax, polygons, label_by_index):
    ax.set_aspect("equal")
    ax.axis("off")

    for i, poly in enumerate(polygons):
        for p in iter_polygons(poly):
            x, y = p.exterior.xy
            ax.fill(
                x,
                y,
                facecolor="white",
                edgecolor="black",
                linewidth=0.8,
                zorder=1,
            )

    for i, poly in enumerate(polygons):
        label = label_by_index.get(i, "")
        if not label:
            continue

        point = poly.representative_point()
        ax.text(
            point.x,
            point.y,
            label,
            ha="center",
            va="center",
            fontsize=9,
            family="serif",
            zorder=3,
        )

    union = unary_union(polygons)
    minx, miny, maxx, maxy = union.bounds
    dx = maxx - minx
    dy = maxy - miny
    pad = 0.08 * max(dx, dy)

    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)


# ============================================================
# Main
# ============================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(PICKLE_PATH, "rb") as f:
        data = pickle.load(f)

    floor_plan = data["floor_plan"]
    polygons = [item["polygon"] for item in floor_plan]
    label_by_index = make_letter_mapping(floor_plan)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig = plt.figure(figsize=(2.05, 4.75))
    grid = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.05, 1.35],
        hspace=0.03,
    )

    ax_tree = fig.add_subplot(grid[0, 0])
    ax_plan = fig.add_subplot(grid[1, 0])

    draw_tree(ax_tree)
    draw_plan(ax_plan, polygons, label_by_index)

    pdf_path = OUTPUT_DIR / f"{OUTPUT_NAME}.pdf"
    png_path = OUTPUT_DIR / f"{OUTPUT_NAME}.png"

    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
    # fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.show()
    plt.close(fig)

    print(f"Saved PDF: {pdf_path}")
    print(f"Saved PNG: {png_path}")


if __name__ == "__main__":
    main()
