import os
import csv
import networkx as nx
from shapely.geometry import Polygon
from shapely.ops import unary_union

from utils_msd import load_pickle

# === import your existing processing functions (topology-based core) ===
from msd_processing import (
    get_type_sets,                          # returns (name_to_idx, private_types, auxiliary_types)
    remove_auxiliary_rooms,
    detect_apartments_and_core_nodes,       # returns (apartments, core_nodes)
    extract_apartment_polygons,
    extract_core_union_from_nodes,
    extract_building_footprint_from_apts_and_core,
)

# === PATH SETUP ===
datapath = r"C:\WF\Thomas Sharon\Floorplan_Dataset\archive\modified-swiss-dwellings-v2\train"
graph_out_dir = os.path.join(datapath, "graph_out")
output = r"C:\WF\Thomas Sharon\Floorplan_Dataset\msd_dataset_creation"

# === OUTPUT CSV ===
out_csv = os.path.join(output, "msd_apt_list_full_filtered.csv")

# === GET ALL IDS FROM FOLDER ===
# IDs = []
# === GET ALL BUILDING IDS FROM graph_out DIRECTORY ===
IDs = [
    os.path.splitext(f)[0]     # remove .pickle extension
    for f in os.listdir(graph_out_dir)
    if f.endswith(".pickle")
]

# Optional: sort for reproducibility
IDs = sorted(IDs, key=int)

print("Total buildings found:", len(IDs))
print("First 10 IDs:", IDs[:10])


# ============================================================
# Helpers
# ============================================================

def safe_polygon(geom):
    """Convert stored geometry to a valid Shapely polygon (or None)."""
    if not geom:
        return None
    try:
        poly = Polygon(geom)
        if poly.is_valid and not poly.is_empty:
            return poly
    except Exception:
        return None
    return None


def union_area_from_nodes(G, nodes, buffer_amt=0.0):
    """Union geometries of given nodes and return area (0.0 if none)."""
    polys = []
    for n in nodes:
        poly = safe_polygon(G.nodes[n].get("geometry"))
        if poly is not None:
            polys.append(poly)
    if not polys:
        return 0.0
    geom = unary_union(polys)
    if buffer_amt:
        geom = geom.buffer(buffer_amt).buffer(-buffer_amt)
    return float(geom.area)


def core_union_to_polygons(core_union):
    """Normalize Polygon/MultiPolygon core_union -> list[Polygon]."""
    if core_union is None or core_union.is_empty:
        return []
    if core_union.geom_type == "Polygon":
        return [core_union]
    if core_union.geom_type == "MultiPolygon":
        return list(core_union.geoms)
    return []


# ============================================================
# CSV HEADER (fixed-width: max 8 apartments, max 2 cores)
# ============================================================
header = (
    ["building_id", "num_apartments"]
    + [f"apt_{k}_area" for k in range(1, 9)]
    + ["num_cores", "core_1_area", "core_2_area", "footprint_area"]
)

written = 0
skipped = 0
failed = 0

os.makedirs(os.path.dirname(out_csv), exist_ok=True)

with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=header)
    writer.writeheader()

    # types from your existing functions
    name_to_idx, private_types, auxiliary_types = get_type_sets()

    for ID in IDs:
        try:
            # === LOAD GRAPH ===
            G = load_pickle(os.path.join(graph_out_dir, f"{ID}.pickle"))

            # === REMOVE AUXILIARY ROOMS ===
            remove_auxiliary_rooms(G, auxiliary_types)

            # === APARTMENTS + CORE (core = before entrance) ===
            apartments, core_nodes = detect_apartments_and_core_nodes(G, private_types)
            num_apartments = len(apartments)

            # # ---- FILTER: max 8 apartments ----
            if num_apartments > 6:   # allow up to 6 apts, but skip if more (too complex for now)
                skipped += 1
                continue

            # ---- CORE COMPONENTS (for counting + 1–2 filter) ----
            core_sub = G.subgraph(core_nodes).copy()
            core_components = list(nx.connected_components(core_sub))
            num_cores = len(core_components)

            if not (1 <= num_cores <= 2):
                skipped += 1
                continue

            # === CORE AREAS (per core component) ===
            core_areas = []
            for comp in core_components:
                core_areas.append(union_area_from_nodes(G, comp, buffer_amt=0.0))

            core_areas = sorted(core_areas, reverse=True)[:2]

            # === APARTMENT AREAS (use your existing polygon extractor) ===
            apartment_polygons = extract_apartment_polygons(G, apartments, auxiliary_types, buffer_amt=0.2)
            apartment_areas = [float(p.area) for p in apartment_polygons]
            apartment_areas = sorted(apartment_areas, reverse=True)[:8]

            # === BUILDING FOOTPRINT (apartments ∪ core) ===
            core_union = extract_core_union_from_nodes(G, core_nodes, buffer_amt=0.15)
            footprint = extract_building_footprint_from_apts_and_core(
                apartment_polygons=apartment_polygons,
                core_union=core_union,
                outer_buffer=0.5,
                inner_buffer=-0.4,
                simplify_tol=0.275,
            )
            footprint_area = float(footprint.area) if footprint is not None else 0.0

            # === BUILD ROW (empty cells for missing apts/cores) ===
            row = {k: "" for k in header}
            row["building_id"] = ID
            row["num_apartments"] = num_apartments
            row["num_cores"] = num_cores
            row["footprint_area"] = round(footprint_area, 2)

            for idx, a in enumerate(apartment_areas, start=1):
                row[f"apt_{idx}_area"] = round(a, 2)

            if len(core_areas) >= 1:
                row["core_1_area"] = round(core_areas[0], 2)
            if len(core_areas) >= 2:
                row["core_2_area"] = round(core_areas[1], 2)

            writer.writerow(row)
            written += 1

        except Exception as e:
            failed += 1
            print(f"ID {ID} failed: {e}")

print(f"✅ Wrote {written} rows to: {out_csv}")
print(f"⏭️ Skipped (filters): {skipped}")
print(f"❌ Failed (errors): {failed}")
