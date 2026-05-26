import os 
from msd_processing_patched import (
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


def main():
    datapath = r"C:\WF\Thomas Sharon\Floorplan_Dataset\archive\modified-swiss-dwellings-v2\train"
    IDs = [7824]
    OUTPUT_PICKLE_FOLDER = r"C:\WF\Thomas Sharon\Floorplan_Dataset\msd_pickle"

    # IDs = [
    #     68, 75, 179, 329, 467, 696, 807, 1291, 1321, 1361, 1575, 1588, 1595,
    #     1601, 1663, 1686, 1712, 1728, 1817, 1925, 1934, 1953, 1996, 2018, 2030,
    #     2049, 2136, 2244, 2389, 2401, 2410, 2540, 2568, 2896, 3002, 3057, 3283,
    #     3594, 3669, 4026, 4234, 4239, 4258, 4321, 4832, 5069, 5086, 5102, 5103,
    #     5319, 5322, 5863, 6362, 6370, 6599, 6676, 7299, 7343, 7737, 7760, 7792,
    #     6644, 7869, 7899, 7914, 7916, 8039, 8202, 8241, 8260, 8264, 8308, 8309,
    #     8314, 8412, 8413, 8443, 8447, 8460, 8549, 8851, 8860, 8863, 8866, 8877,
    #     8881, 9205, 9678, 9729, ]
    # IDs = [10388, 10405, 10655, 10959, 11226, 11434,
    #     11818, 11906, 11967, 13488, 13544, 13858, 14016, 14063, 14123, 14128,
    #     14131, 14747, 14818, 14819, 14881, 14897, 15364, 22206, 22211, 22844,
    #     22886, 23213, 23246, 23562, 23865, 23871, 24153, 24173, 24288, 24472,
    #     24476, 24501, 24542, 24966] 
    # IDs =[25184, 25307, 25320, 25947, 26170, 26175,
    #     26471, 26593, 26653, 26838, 26858, 26939, 28611, 28949, 29010, 29270,
    #     29399, 29686, 29729, 30405, 30453, 39307, 42392, 43687, 44248, 44871,
    #     45570, 45576, 45631, 45644, 45658, 45724, 46073, 46492, 47229, 47492,
    #     48408, 48966, 49004, 49035, 49051, 49320, 49951, 50528, 50530, 50537,
    #     51001, 51680, 51693, 176, 322, 405, 553, 712, 721, 803, 993, 1827, 1943,
    #     1976, 2801, 3039, 3616, 5325, 7801, 8364, 8424, 9222, 9682, 10288, 10376,
    #     11160, 11240, 23229, 24140, 26465, 49018, 49602, 50543
    #     ]

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

        core_union = extract_core_union_from_nodes(
            G,
            core_nodes,
            buffer_amt=0.15,   # kept as requested
        )

        footprint = extract_building_footprint_from_apts_and_core(
            apartment_polygons=apartment_polygons,
            core_union=core_union,
            outer_buffer=0.45,
            inner_buffer=-0.35,
            simplify_tol=0.03,
        )

        apartment_polygons_filled, residual_gap = simultaneous_apartment_growth(
            apartment_polygons=apartment_polygons,
            core_union=core_union,
            footprint=footprint,
            step=0.03,
            max_iter=400,
            min_residual_area=1e-4,
            simplify_tol=0.01,
        )

        save_zoning_pickle(
            dwelling_polygons=apartment_polygons_filled,
            core_polygons=core_union,
            out_path= os.path.join(OUTPUT_PICKLE_FOLDER, f"{building_id}.pickle"),
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
            core_union=core_union,
            footprint=footprint,
            building_id=building_id,
            residual_gap=residual_gap,
        )


if __name__ == "__main__":
    main()