from pathlib import Path
from msd_processing import (
    get_type_sets,
    load_graph,
    remove_auxiliary_rooms,
    detect_apartments_and_core_nodes,
    extract_apartment_polygons,
    extract_core_union_from_nodes,
    extract_building_footprint_from_apts_and_core,
    simultaneous_apartment_growth,
    plot_all_views,
    save_zoning_pickle
)


def load_building_ids(ids_file: Path) -> list[int]:
    ids = [75]

    with ids_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            ids.append(int(line))

    return ids

def main():
    PROJECT_DIR = Path(__file__).resolve().parent

    IDS_FILE = PROJECT_DIR / "data" / "msd_thesis_building_ids.txt"
    datapath = PROJECT_DIR / "data" / "raw_msd"
    OUTPUT_PICKLE_FOLDER = PROJECT_DIR / "data" / "ground_truth"

    OUTPUT_PICKLE_FOLDER.mkdir(parents=True, exist_ok=True)

    IDs = load_building_ids(IDS_FILE)

    name_to_idx, private_types, auxiliary_types = get_type_sets()

    for building_id in IDs:
        G = load_graph(datapath, building_id)
        print(f"Loaded graph_out for ID {building_id}: {len(G.nodes)} rooms, {len(G.edges)} edges")

        removed = remove_auxiliary_rooms(G, auxiliary_types)
        print(f"Removed {removed} auxiliary rooms.")

        apartments, core_nodes = detect_apartments_and_core_nodes(G, private_types)
        print(f"Detected {len(apartments)} apartment unit(s).")
        print(f"Core nodes: {len(core_nodes)}")

        apartment_polygons = extract_apartment_polygons(
            G,
            apartments,
            auxiliary_types,
            buffer_amt=0.08,
        )

        core_polygons = extract_core_union_from_nodes(
            G,
            core_nodes,
            buffer_amt=0.15,   # kept as requested
        )

        footprint = extract_building_footprint_from_apts_and_core(
            apartment_polygons=apartment_polygons,
            core_polygons=core_polygons,
            outer_buffer=0.45,
            inner_buffer=-0.35,
            simplify_tol=0.03,
        )

        apartment_polygons_filled, residual_gap = simultaneous_apartment_growth(
            apartment_polygons=apartment_polygons,
            core_polygons=core_polygons,
            footprint=footprint,
            step=0.03,
            max_iter=400,
            min_residual_area=1e-4,
            simplify_tol=0.01,
        )

        save_zoning_pickle(
            dwelling_polygons=apartment_polygons_filled,
            core_polygons=core_polygons,
            out_path=OUTPUT_PICKLE_FOLDER / f"{building_id}.pickle",
            building_id=building_id,
        )

        if residual_gap is None or residual_gap.is_empty:
            print("Gap filling successful: zoning is footprint-complete.")
        else:
            print(f"Residual gap area still present: {residual_gap.area:.6f}")

        plot_all_views(
            G=G,
            apartments=apartments,
            apartment_polygons=apartment_polygons_filled,
            core_nodes=core_nodes,
            core_polygons=core_polygons,
            footprint=footprint,
            building_id=building_id,
            residual_gap=residual_gap,
        )


if __name__ == "__main__":
    main()