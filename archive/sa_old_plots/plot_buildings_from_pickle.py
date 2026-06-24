#!/usr/bin/env python3
from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Any, Iterable, Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon


# ============================================================
# CONFIGURATION
# ============================================================
# Set to a concrete file if you want. If None, the script tries a few common repo-relative paths.
PKL_PATH: Optional[Path] = None

# Plot only one building id, e.g. "DESHPDHK0000lce9_3".
# Leave as None to plot the first MAX_BUILDINGS buildings from the pickle.
BUILDING_ID: Optional[str] = "DESHPDHK0000lce9_3"

# Number of buildings to show when BUILDING_ID is None
MAX_BUILDINGS = 1

# How geometry should be chosen for the top view:
#   "floors_first" -> use floors, else ceilings, else walls
#   "floors"       -> use only floors
#   "all"          -> draw floors + ceilings + walls
SURFACE_MODE = "floors_first"

# Output / display
OUTPUT_DIR = None
OUTPUT_BASENAME = "buildings_from_pickle"
SAVE_INDIVIDUAL_PNGS = True
SAVE_OVERVIEW_PNG = True
SHOW_PLOTS = True
DPI = 180
N_COLS = 3
ANNOTATE_ZONES = True
# ============================================================


DEFAULT_PKL_CANDIDATES = [
    Path("data/building_data/building_data_merged.pkl"),
    Path("../data/building_data/building_data_merged.pkl"),
    Path(__file__).resolve().parent / "data" / "building_data" / "building_data_merged.pkl",
    Path(__file__).resolve().parent.parent / "data" / "building_data" / "building_data_merged.pkl",
]


def resolve_pkl_path(pkl_path: Optional[Path]) -> Path:
    if pkl_path is not None:
        p = Path(pkl_path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Pickle file not found: {p}")
        return p

    for candidate in DEFAULT_PKL_CANDIDATES:
        p = candidate.expanduser().resolve()
        if p.is_file():
            return p

    tried = "\n".join(str(c.expanduser().resolve()) for c in DEFAULT_PKL_CANDIDATES)
    raise FileNotFoundError(
        "Could not find the building pickle automatically.\n"
        "Set PKL_PATH in the configuration block.\n"
        f"Tried:\n{tried}"
    )


def load_buildings(pkl_path: Path) -> list[dict[str, Any]]:
    with open(pkl_path, "rb") as fh:
        data = pickle.load(fh)

    if not isinstance(data, list):
        raise ValueError("PKL has unexpected format: expected a list of building dictionaries.")

    buildings: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        bid = item.get("building_id") or item.get("Building ID") or item.get("id")
        if bid is None:
            continue
        buildings.append(item)

    if not buildings:
        raise ValueError("No valid building dictionaries with a building id were found in the pickle.")

    return buildings


def iter_storeys(payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
    polygons = payload.get("polygons", {})
    if not isinstance(polygons, dict):
        return []
    return polygons.get("storeys", []) or []


def iter_zones(payload: dict[str, Any]) -> Iterable[tuple[int, dict[str, Any]]]:
    for storey_idx, storey in enumerate(iter_storeys(payload), start=1):
        for zone in (storey.get("zones", []) or []):
            if isinstance(zone, dict):
                yield storey_idx, zone


def _coerce_point(point: Any) -> Optional[tuple[float, float]]:
    if isinstance(point, (list, tuple)) and len(point) >= 2:
        try:
            return float(point[0]), float(point[1])
        except Exception:
            return None
    return None


def polygon_to_xy(polygon_obj: Any) -> Optional[list[tuple[float, float]]]:
    """
    Tries to extract a 2D boundary from shapely-like polygons or coordinate lists.
    Z is ignored for top-view plotting.
    """
    coords = None

    try:
        if hasattr(polygon_obj, "exterior") and hasattr(polygon_obj.exterior, "coords"):
            coords = list(polygon_obj.exterior.coords)
        elif hasattr(polygon_obj, "coords"):
            coords = list(polygon_obj.coords)
        elif isinstance(polygon_obj, (list, tuple)):
            coords = list(polygon_obj)
    except Exception:
        coords = None

    if not coords:
        return None

    xy: list[tuple[float, float]] = []
    for point in coords:
        p = _coerce_point(point)
        if p is not None:
            xy.append(p)

    if len(xy) < 3:
        return None

    if xy[0] != xy[-1]:
        xy.append(xy[0])

    return xy


def polygon_centroid(xy: list[tuple[float, float]]) -> tuple[float, float]:
    # Fallback centroid based on average coordinates; robust enough for labels.
    xs = [p[0] for p in xy[:-1]] if len(xy) > 1 and xy[0] == xy[-1] else [p[0] for p in xy]
    ys = [p[1] for p in xy[:-1]] if len(xy) > 1 and xy[0] == xy[-1] else [p[1] for p in xy]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def extract_surface_polygons(surface_list: Any) -> list[list[tuple[float, float]]]:
    out: list[list[tuple[float, float]]] = []
    for item in surface_list or []:
        polygon_obj = None
        if isinstance(item, (list, tuple)) and len(item) >= 1:
            polygon_obj = item[0]
        else:
            polygon_obj = item
        xy = polygon_to_xy(polygon_obj)
        if xy is not None:
            out.append(xy)
    return out


def zone_polygons(zone: dict[str, Any], surface_mode: str = "floors_first") -> list[list[tuple[float, float]]]:
    floors = extract_surface_polygons(zone.get("floors") or [])
    ceilings = extract_surface_polygons(zone.get("ceilings") or [])
    walls = extract_surface_polygons(zone.get("walls") or [])

    if surface_mode == "floors":
        return floors
    if surface_mode == "all":
        return floors + ceilings + walls

    # default: best available horizontal/top-view geometry first
    return floors or ceilings or walls


def color_for_index(i: int):
    cmap = plt.get_cmap("tab20")
    return cmap(i % cmap.N)


def sanitize_filename(text: str) -> str:
    keep = []
    for ch in str(text):
        keep.append(ch if ch.isalnum() or ch in ("-", "_", ".") else "_")
    return "".join(keep).strip("_")


def plot_single_building(ax: plt.Axes, payload: dict[str, Any], title: Optional[str] = None) -> dict[str, Any]:
    building_id = str(payload.get("building_id") or payload.get("Building ID") or payload.get("id") or "UNKNOWN")
    zone_counter = 0
    plotted_polygons = 0
    labels_done: set[str] = set()

    x_min = math.inf
    x_max = -math.inf
    y_min = math.inf
    y_max = -math.inf

    for storey_idx, zone in iter_zones(payload):
        zone_name = str(zone.get("name", f"Zone_{zone_counter + 1}")).strip()
        poly_list = zone_polygons(zone, surface_mode=SURFACE_MODE)
        color = color_for_index(zone_counter)

        for xy in poly_list:
            patch = MplPolygon(
                xy,
                closed=True,
                facecolor=color,
                edgecolor="black",
                linewidth=0.8,
                alpha=0.35,
            )
            ax.add_patch(patch)
            plotted_polygons += 1

            for x, y in xy:
                x_min = min(x_min, x)
                x_max = max(x_max, x)
                y_min = min(y_min, y)
                y_max = max(y_max, y)

        if ANNOTATE_ZONES and poly_list and zone_name not in labels_done:
            cx, cy = polygon_centroid(poly_list[0])
            ax.text(cx, cy, f"S{storey_idx}\n{zone_name}", ha="center", va="center", fontsize=7)
            labels_done.add(zone_name)

        zone_counter += 1

    if plotted_polygons == 0:
        ax.text(0.5, 0.5, "No plottable geometry found", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    else:
        dx = max(x_max - x_min, 1.0)
        dy = max(y_max - y_min, 1.0)
        pad_x = 0.08 * dx
        pad_y = 0.08 * dy
        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        ax.set_ylim(y_min - pad_y, y_max + pad_y)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title or building_id)

    storey_count = sum(1 for _ in iter_storeys(payload))
    return {
        "building_id": building_id,
        "n_storeys": storey_count,
        "n_zones": zone_counter,
        "n_plotted_polygons": plotted_polygons,
    }


def select_buildings(buildings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if BUILDING_ID is not None:
        wanted = str(BUILDING_ID).strip()
        selected = [b for b in buildings if str(b.get("building_id") or b.get("Building ID") or b.get("id")).strip() == wanted]
        if not selected:
            raise KeyError(f"Building id '{wanted}' not found in the pickle.")
        return selected
    return buildings[: max(1, int(MAX_BUILDINGS))]


def save_individual_plots(buildings: list[dict[str, Any]], output_dir: Path) -> None:
    for building in buildings:
        bid = str(building.get("building_id") or building.get("Building ID") or building.get("id") or "UNKNOWN")
        fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
        summary = plot_single_building(ax, building, title=bid)
        out_path = output_dir / f"{sanitize_filename(bid)}.png"
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
        print(
            f"[SAVED] {out_path} | storeys={summary['n_storeys']} "
            f"zones={summary['n_zones']} plotted_polygons={summary['n_plotted_polygons']}"
        )
        if not SHOW_PLOTS:
            plt.close(fig)


def save_overview_plot(buildings: list[dict[str, Any]], output_dir: Path) -> None:
    n = len(buildings)
    n_cols = max(1, int(N_COLS))
    n_rows = math.ceil(n / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 5.0 * n_rows), constrained_layout=True)

    if not isinstance(axes, (list, tuple)):
        try:
            axes = axes.ravel()
        except Exception:
            axes = [axes]
    else:
        axes = list(axes)

    for ax, building in zip(axes, buildings):
        bid = str(building.get("building_id") or building.get("Building ID") or building.get("id") or "UNKNOWN")
        plot_single_building(ax, building, title=bid)

    for ax in axes[len(buildings):]:
        ax.axis("off")

    out_path = output_dir / f"{OUTPUT_BASENAME}_overview.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    print(f"[SAVED] {out_path}")

    if not SHOW_PLOTS:
        plt.close(fig)


def print_summary(buildings: list[dict[str, Any]], pkl_path: Path) -> None:
    print("=" * 72)
    print(f"Pickle file: {pkl_path}")
    print(f"Buildings loaded: {len(buildings)}")
    print("Selected building ids:")
    for b in buildings:
        bid = b.get("building_id") or b.get("Building ID") or b.get("id")
        print(f"  - {bid}")
    print("=" * 72)


def main() -> None:
    pkl_path = resolve_pkl_path(PKL_PATH)
    all_buildings = load_buildings(pkl_path)
    buildings = select_buildings(all_buildings)

    output_dir = OUTPUT_DIR.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print_summary(buildings, pkl_path)

    if SAVE_INDIVIDUAL_PNGS:
        save_individual_plots(buildings, output_dir)

    if SAVE_OVERVIEW_PNG and len(buildings) > 1:
        save_overview_plot(buildings, output_dir)

    if SHOW_PLOTS:
        plt.show()


if __name__ == "__main__":
    main()
