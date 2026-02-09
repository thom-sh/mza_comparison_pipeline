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
out_csv = os.path.join(output, "msd_apt_list_1.csv")

# === GET ALL IDS FROM FOLDER ===
IDs = [68,
 75,
 108,
 176,
 179,
 322,
 329,
 341,
 343,
 367,
 405,
 467,
 470,
 471,
 474,
 476,
 524,
 546,
 553,
 559,
 594,
 613,
 621,
 624,
 696,
 712,
 721,
 803,
 807,
 973,
 982,
 990,
 993,
 994,
 1201,
 1261,
 1291,
 1321,
 1322,
 1326,
 1361,
 1366,
 1544,
 1575,
 1588,
 1595,
 1601,
 1663,
 1686,
 1712,
 1728,
 1802,
 1817,
 1827,
 1856,
 1925,
 1934,
 1943,
 1948,
 1953,
 1956,
 1976,
 1980,
 1996,
 2000,
 2006,
 2018,
 2030,
 2038,
 2041,
 2049,
 2075,
 2097,
 2136,
 2139,
 2244,
 2258,
 2389,
 2401,
 2410,
 2419,
 2422,
 2425,
 2428,
 2437,
 2538,
 2540,
 2544,
 2568,
 2801,
 2877,
 2896,
 2898,
 2900,
 3002,
 3039,
 3043,
 3053,
 3057,
 3098,
 3283,
 3511,
 3512,
 3594,
 3616,
 3656,
 3659,
 3663,
 3669,
 3727,
 4026,
 4067,
 4069,
 4234,
 4239,
 4243,
 4252,
 4258,
 4321,
 4828,
 4832,
 4872,
 5069,
 5070,
 5086,
 5102,
 5103,
 5105,
 5319,
 5320,
 5321,
 5322,
 5324,
 5325,
 5863,
 5864,
 5880,
 5919,
 5964,
 6151,
 6332,
 6335,
 6354,
 6362,
 6367,
 6368,
 6370,
 6599,
 6644,
 6676,
 6677,
 6775,
 7293,
 7299,
 7343,
 7737,
 7740,
 7760,
 7787,
 7792,
 7801,
 7820,
 7824,
 7848,
 7869,
 7872,
 7887,
 7899,
 7914,
 7916,
 7972,
 8039,
 8192,
 8202,
 8241,
 8243,
 8260,
 8264,
 8308,
 8309,
 8314,
 8346,
 8364,
 8380,
 8400,
 8412,
 8413,
 8414,
 8424,
 8432,
 8443,
 8447,
 8460,
 8514,
 8520,
 8523,
 8534,
 8549,
 8562,
 8697,
 8707,
 8851,
 8860,
 8863,
 8866,
 8877,
 8881,
 9056,
 9102,
 9130,
 9132,
 9205,
 9222,
 9226,
 9256,
 9481,
 9678,
 9682,
 9729,
 10277,
 10288,
 10376,
 10382,
 10388,
 10394,
 10405,
 10612,
 10633,
 10655,
 10959,
 11108,
 11160,
 11226,
 11240,
 11244,
 11434,
 11498,
 11501,
 11574,
 11670,
 11688,
 11693,
 11818,
 11904,
 11906,
 11967,
 11995,
 12005,
 12945,
 12948,
 13485,
 13488,
 13541,
 13544,
 13858,
 13875,
 13881,
 13987,
 14016,
 14063,
 14123,
 14128,
 14131,
 14134,
 14193,
 14717,
 14727,
 14747,
 14764,
 14818,
 14819,
 14881,
 14897,
 14959,
 15118,
 15361,
 15364,
 15411,
 18749,
 22206,
 22211,
 22844,
 22886,
 23213,
 23229,
 23246,
 23562,
 23865,
 23871,
 23901,
 24097,
 24131,
 24140,
 24153,
 24173,
 24227,
 24240,
 24263,
 24288,
 24313,
 24395,
 24472,
 24476,
 24501,
 24511,
 24542,
 24549,
 24694,
 24770,
 24966,
 25184,
 25194,
 25208,
 25307,
 25320,
 25947,
 26170,
 26175,
 26465,
 26471,
 26593,
 26600,
 26636,
 26653,
 26693,
 26838,
 26844,
 26858,
 26870,
 26939,
 27594,
 27740,
 28393,
 28422,
 28461,
 28611,
 28949,
 29010,
 29270,
 29338,
 29341,
 29399,
 29445,
 29648,
 29661,
 29686,
 29719,
 29729,
 30373,
 30405,
 30453,
 39307,
 41652,
 42392,
 43687,
 44248,
 44871,
 44946,
 45397,
 45570,
 45576,
 45583,
 45631,
 45644,
 45658,
 45661,
 45670,
 45724,
 46073,
 46492,
 47133,
 47229,
 47236,
 47492,
 47945,
 48408,
 48596,
 48966,
 49004,
 49018,
 49035,
 49051,
 49307,
 49312,
 49320,
 49322,
 49602,
 49895,
 49898,
 49913,
 49951,
 50023,
 50528,
 50530,
 50537,
 50543,
 50911,
 50948,
 50953,
 51001,
 51076,
 51657,
 51662,
 51680,
 51693,
 51722]


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
            # if num_apartments > 8:
            #     skipped += 1
            #     continue

            # ---- CORE COMPONENTS (for counting + 1–2 filter) ----
            core_sub = G.subgraph(core_nodes).copy()
            core_components = list(nx.connected_components(core_sub))
            num_cores = len(core_components)

            # if not (1 <= num_cores <= 2):
            #     skipped += 1
            #     continue

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
