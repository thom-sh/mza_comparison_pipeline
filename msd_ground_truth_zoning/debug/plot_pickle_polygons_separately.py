import math
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon


# ============================================================
# Helpers
# ============================================================

def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def iter_polygon_parts(geom):
    """Yield Polygon parts from a Polygon or MultiPolygon."""
    if geom is None or geom.is_empty:
        return

    if geom.geom_type == "Polygon":
        yield geom
    elif geom.geom_type == "MultiPolygon":
        for part in geom.geoms:
            if not part.is_empty:
                yield part
    else:
        raise TypeError(f"Unsupported geometry type: {geom.geom_type}")


def draw_geometry(ax, geom, facecolor="#DCEAF7", edgecolor="#777d84", linewidth=1.0):
    """Draw a shapely Polygon/MultiPolygon on an axis."""
    min_x, min_y, max_x, max_y = geom.bounds

    for poly in iter_polygon_parts(geom):
        x, y = poly.exterior.xy
        coords = list(zip(x, y))
        patch = MplPolygon(
            coords,
            closed=True,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
        )
        ax.add_patch(patch)

        # draw holes if present
        for interior in poly.interiors:
            ix, iy = interior.xy
            hole_patch = MplPolygon(
                list(zip(ix, iy)),
                closed=True,
                facecolor="white",
                edgecolor=edgecolor,
                linewidth=linewidth,
            )
            ax.add_patch(hole_patch)

    ax.set_aspect("equal")
    ax.set_xlim(min_x - 0.1, max_x + 0.1)
    ax.set_ylim(min_y - 0.1, max_y + 0.1)
    ax.axis("off")


# ============================================================
# Main
# ============================================================

def main() -> None:
    # === HARD-CODED INPUT ===
    pickle_path = Path(r"C:\WF\Thomas Sharon\Floorplan_Dataset\msd_pickle\68.pickle")

    # === OPTIONAL OUTPUT SETTINGS ===
    save_combined_figure = False
    combined_out = pickle_path.with_name(f"{pickle_path.stem}_separate_polygons.png")

    save_individual_figures = False
    individual_out_dir = pickle_path.with_name(f"{pickle_path.stem}_polygon_plots")

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 7,
        "axes.titlesize": 7,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })

    data = load_pickle(str(pickle_path))
    building_id = data.get("building_id", "unknown")
    floor_plan = data.get("floor_plan", [])

    if not floor_plan:
        raise ValueError(f"No 'floor_plan' entries found in: {pickle_path}")

    n = len(floor_plan)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.0 * nrows))
    if not isinstance(axes, (list, tuple)):
        try:
            axes = axes.ravel()
        except AttributeError:
            axes = [axes]

    for i, item in enumerate(floor_plan):
        ax = axes[i]
        geom = item.get("polygon")
        room_type = item.get("room_type", "NA")

        if geom is None:
            ax.set_title(f"Polygon {i + 1}\n(room_type={room_type})")
            ax.text(0.5, 0.5, "No geometry", ha="center", va="center")
            ax.axis("off")
            continue

        draw_geometry(ax, geom)
        ax.set_title(f"Polygon {i + 1}\nroom_type = {room_type}")

    # hide unused axes
    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Building ID {building_id} — polygons plotted separately", fontsize=8)
    plt.tight_layout()

    if save_combined_figure:
        fig.savefig(combined_out, dpi=300, bbox_inches="tight")
        print(f"Saved combined figure to: {combined_out}")

    if save_individual_figures:
        individual_out_dir.mkdir(parents=True, exist_ok=True)
        for i, item in enumerate(floor_plan):
            geom = item.get("polygon")
            room_type = item.get("room_type", "NA")
            if geom is None:
                continue

            fig_i, ax_i = plt.subplots(figsize=(3, 3))
            draw_geometry(ax_i, geom)
            ax_i.set_title(f"Polygon {i + 1} | room_type = {room_type}")
            out_path = individual_out_dir / f"polygon_{i + 1:02d}.png"
            fig_i.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close(fig_i)

        print(f"Saved individual polygon plots to: {individual_out_dir}")

    plt.show()


if __name__ == "__main__":
    main()
