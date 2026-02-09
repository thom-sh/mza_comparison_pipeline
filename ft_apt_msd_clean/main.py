from msd_processing import (
    get_type_sets,
    load_graph,
    remove_auxiliary_rooms,
    detect_apartments_and_core_nodes,
    extract_apartment_polygons,
    extract_core_union_from_nodes,
    extract_building_footprint_from_apts_and_core,
    plot_all_views,
)

# ============================================================
# Main
# ============================================================

def main():
    datapath = r"C:\WF\Thomas Sharon\Floorplan_Dataset\archive\modified-swiss-dwellings-v2\train"
    IDs = [1366]

    # core is topology-based (before apartment entrance), so we only need private + auxiliary types here
    name_to_idx, private_types, auxiliary_types = get_type_sets()

    for building_id in IDs:
        G = load_graph(datapath, building_id)
        print(f"Loaded graph_out for ID {building_id}: {len(G.nodes)} rooms, {len(G.edges)} edges")

        removed = remove_auxiliary_rooms(G, auxiliary_types)
        print(f"🧹 Removed {removed} balconies (aux rooms).")

        # Apartments + Core nodes (core = before entrance)
        apartments, core_nodes = detect_apartments_and_core_nodes(G, private_types)
        print(f"\n🏠 Detected {len(apartments)} apartment unit(s).")
        print(f"🧭 Core nodes (before entrance): {len(core_nodes)}\n")

        # Geometry extraction
        apartment_polygons = extract_apartment_polygons(G, apartments, auxiliary_types)
        core_union = extract_core_union_from_nodes(G, core_nodes)

        footprint = extract_building_footprint_from_apts_and_core(
            apartment_polygons=apartment_polygons,
            core_union=core_union,
        )

        plot_all_views(
            G=G,
            apartments=apartments,
            apartment_polygons=apartment_polygons,
            core_nodes=core_nodes,
            core_union=core_union,
            footprint=footprint,
            building_id=building_id,
        )


if __name__ == "__main__":
    main()
