import os
import math
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union


# ============================================================
# THESIS STYLE
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.titlesize": 8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

LEGEND_BOX_EDGE = "#bdc1c5"
POLYGON_EDGE_COLOR = "#777d84"
CORE_EDGE_COLOR = "#777d84"

DWELLING_COLORS = [
    "#DCEAF7",
    "#AFCBE3",
    "#7FA6C9",
    "#DBDBDB",
    "#BABABA",
    "#999999",
]

CORE_COLOR = "#5F666D"


# ============================================================
# HELPERS
# ============================================================

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def ensure_polygonal(geom):
    if geom is None:
        return None

    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom

    try:
        poly = Polygon(geom)
        if not poly.is_empty:
            return poly
    except Exception:
        pass

    return None


def polygon_parts(geom):
    if geom is None:
        return []

    if isinstance(geom, Polygon):
        return [geom]

    if isinstance(geom, MultiPolygon):
        return [g for g in geom.geoms if not g.is_empty]

    return []


def get_building_id(data, path):
    return os.path.splitext(os.path.basename(path))[0]


def parse_floorplan_pickle(path):
    data = load_pickle(path)

    if not isinstance(data, dict) or "floor_plan" not in data:
        raise ValueError(f"Unexpected pickle format in {path}")

    building_id = get_building_id(data, path)

    dwellings = []
    cores = []

    for item in data["floor_plan"]:
        geom = ensure_polygonal(item.get("polygon"))
        room_type = item.get("room_type", 0)

        if geom is None:
            continue

        parts = polygon_parts(geom)

        if room_type == 1:
            cores.extend(parts)
        else:
            dwellings.extend(parts)

    return building_id, dwellings, cores


def add_geom_patches(ax, geoms, facecolor, edgecolor, linewidth=0.25):
    patches = []
    for geom in geoms:
        for poly in polygon_parts(geom):
            x, y = poly.exterior.coords.xy
            xy = list(zip(x, y))
            patches.append(MplPolygon(xy, closed=True))

    if patches:
        pc = PatchCollection(
            patches,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth
        )
        ax.add_collection(pc)


def compute_global_window(parsed_data, pad_ratio=0.01):
    """
    Compute one common axis window for all plots so every box is the same size.
    """
    widths = []
    heights = []

    for _, dwellings, cores in parsed_data:
        all_geoms = dwellings + cores
        if not all_geoms:
            continue

        merged = unary_union(all_geoms)
        minx, miny, maxx, maxy = merged.bounds
        widths.append(maxx - minx)
        heights.append(maxy - miny)

    if not widths or not heights:
        return 1.0, 1.0

    max_w = max(widths)
    max_h = max(heights)
    max_dim = max(max_w, max_h)

    box_size = max_dim * (1 + 2 * pad_ratio)
    return box_size, box_size


def plot_single_plan(ax, dwellings, cores, title, pad_ratio):
    for i, geom in enumerate(dwellings):
        add_geom_patches(
            ax,
            [geom],
            facecolor=DWELLING_COLORS[i % len(DWELLING_COLORS)],
            edgecolor=POLYGON_EDGE_COLOR,
            linewidth=0.25
        )

    add_geom_patches(
        ax,
        cores,
        facecolor=CORE_COLOR,
        edgecolor=CORE_EDGE_COLOR,
        linewidth=0.25
    )

    all_geoms = dwellings + cores
    if all_geoms:
        merged = unary_union(all_geoms)
        minx, miny, maxx, maxy = merged.bounds
        cx = 0.5 * (minx + maxx)
        cy = 0.5 * (miny + maxy)

        w = maxx - minx
        h = maxy - miny
        max_dim = max(w, h)

        if max_dim <=0:
            max_dim = 1.0
        
        box_size = max_dim * (1 + 2 * pad_ratio)

        ax.set_xlim(cx - box_size / 2, cx + box_size / 2)
        ax.set_ylim(cy - box_size / 2, cy + box_size / 2)

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(str(title), pad=0.2, fontsize = 5.8)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_edgecolor(LEGEND_BOX_EDGE)


# ============================================================
# PATH / ID UTILITIES
# ============================================================

def build_pickle_paths_from_ids(folder, ids, max_ids=200):
    ids = ids[:max_ids]
    paths = []

    for bid in ids:
        path = os.path.join(folder, f"{bid}.pickle")
        if os.path.exists(path):
            paths.append(path)
        else:
            print(f"Missing pickle for ID {bid}: {path}")

    return paths


# ============================================================
# MAIN GRID DISPLAY
# ============================================================

def display_zoning_pickles(
    pickle_paths,
    ncols=9,
    figure_title="MSD ground truth (panel d)",
    save_path=None,
    dpi=300,
):
    parsed = []
    for path in pickle_paths:
        try:
            building_id, dwellings, cores = parse_floorplan_pickle(path)
            parsed.append((building_id, dwellings, cores))
        except Exception as e:
            print(f"Skipping {path}: {e}")

    if not parsed:
        raise ValueError("No valid pickle files found.")

    n = len(parsed)
    nrows = math.ceil(n / ncols)

#    global_w, global_h = compute_global_window(parsed, pad_ratio=0.01)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(8.27, 11.69),  # A4 size in inches
        dpi=dpi
    )

    fig.subplots_adjust(
    left=0.01,
    right=0.99,
    top=0.985,
    bottom=0.01,
    wspace=0.001,
    hspace=0.19
)

    if nrows == 1 and ncols == 1:
        axes = [axes]
    elif nrows == 1 or ncols == 1:
        axes = list(axes)
    else:
        axes = axes.flatten()

    for ax, (building_id, dwellings, cores) in zip(axes, parsed):
        plot_single_plan(ax, dwellings, cores, building_id, pad_ratio=0.04)

    for ax in axes[len(parsed):]:
        ax.axis("off")

    # fig.suptitle(figure_title, fontsize=16, y=0.995)
    # plt.tight_layout(rect=[0, 0, 1, 0.988])

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.01)
        print(f"Saved figure to: {save_path}")

    plt.show()


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    pickle_folder = r"C:\WF\Thomas Sharon\Floorplan_Dataset\msd_pickle"

    ids_to_plot = [
    68, 75, 179, 329, 467, 696, 807, 1291, 1321, 1361, 1575, 1588, 1595,
    1601, 1663, 1686, 1712, 1728, 1817, 1925, 1934, 1953, 1996, 2018, 2030,
    2049, 2136, 2244, 2389, 2401, 2410, 2540, 2568, 2896, 3002, 3057, 3283,
    3594, 3669, 4026, 4234, 4239, 4258, 4321, 4832, 5069, 5086, 5102, 5103,
    5319, 5322, 5863, 6362, 6370, 6599, 6676, 7299, 7343, 7737, 7760, 7792,
    7824, 7869, 7899, 7914, 7916, 8039, 8202, 8241, 8260, 8264, 8308, 8309,
    8314, 8412, 8413, 8443, 8447, 8460, 8549, 8851, 8860, 8863, 8866, 8877,
    8881, 9205, 9678, 9729, 10277, 10388, 10405, 10655, 10959, 11226, 11434,
    11818, 11906, 11967, 13488, 13544, 13858, 14016, 14063, 14123, 14128,
    14131, 14747, 14818, 14819, 14881, 14897, 15364, 22206, 22211, 22844,
    22886, 23213, 23246, 23562, 23865, 23871, 24153, 24173, 24288, 24472,
    24476, 24501, 24542, 24966, 25184, 25307, 25320, 25947, 26170, 26175,
    26471, 26593, 26653, 26838, 26858, 26939, 28611, 28949, 29270, 29399,
    29686, 29729, 30405, 30453, 39307, 42392, 43687, 44248, 44871, 45570,
    45576, 45631, 45644, 45658, 45724, 46073, 46492, 47229, 47492, 48408,
    48966, 49004, 49035, 49051, 49320, 49951, 50528, 50530, 50537, 51001,
    51680, 51693, 176, 322, 405, 553, 712, 721, 803, 993, 1827, 1943, 1976,
    2801, 3039, 3616, 5325, 7801, 8364, 8424, 9222, 9682, 10288, 10376, 11160,
    11240, 23229, 24140, 26465, 49018, 49602, 50543
]

    pickle_paths = build_pickle_paths_from_ids(
        folder=pickle_folder,
        ids=ids_to_plot,
        max_ids=200
    )

    display_zoning_pickles(
        pickle_paths=pickle_paths,
        ncols=12,
        figure_title="MSD ground truth (panel d)",
        save_path=r"C:\WF\Thomas Sharon\Master_Thesis_Report\figures\msd_gt_panel.pdf",
        dpi=300,
    )