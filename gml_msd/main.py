"""
Batch pipeline for:
1. Creating GeoJSON footprints from Swiss dataset
2. Replacing building footprints inside a CityGML file
"""

import os
from footprint_ext_debug import create_footprint_geojson
from gml_footprint_replacer3 import replace_building_footprint

# === CONFIGURATION ===

# Folder containing Swiss *.pickle files
DATAPATH = r"C:\WF\Thomas Sharon\Floorplan_Dataset\archive\modified-swiss-dwellings-v2\train"
# DATAPATH = r"N:\9_SF-Public\Austausch\Thomas Sharon\Master_Thesis_Sharon\Floorplan_Dataset\archive\modified-swiss-dwellings-v2\train"

# Input CityGML file
INPUT_GML = r"C:\Sharon\rom_auto_multizoning_msd\data\LoD2_Berlin_Moabit_neu.gml"
# INPUT_GML = r"C:\Sharon\rom_auto_multizoning_msd\data\LoD2_Berlin_Moabit_neu.gml"


# ===== MULTIPLE SWISS BUILDING IDS HERE =====
# BUILDING_IDS = [68, 75, 553, 108, 124, 154, 176, 329, 341, 343, 367, 405, 443, 447, 461, 463, 467, 474, 477, 496, 524, 546, 559, 613, 696, 803, 807, 974, 993, 1261]     # ← EDIT THIS LIST

# BUILDING_IDS = [1588, 1602, 1663, 1686, 1939, 1943, 1956, 1972, 1996, 2075, 2097, 2244, 2258, 2389, 2538, 2542, 2751, 2894, 3451, 3594, 5443]

# BUILDING_IDS = [5641, 6151, 7291, 7293, 7310, 7899, 8413, 8443, 8460, 8514, 8520, 8529, 8562, 8697, 8707, 8866, 8867, 9056, 9132, 9729, 9813, 10277, 10288, 10382, 10394, 11434, 11160]

BUILDING_IDS = [28393]

# BUILDING_IDS = [68, 75, 108, 176, 179, 322, 329, 341, 343, 367, 405, 467, 470, 471, 474, 524, 553, 559, 613, 621, 624, 696, 712, 721, 803, 807, 973, 982, 990, 993, 1201, 1261, 1291, 1321, 1361, 1366, 1544, 1575, 1588, 1595, 1601, 1663, 1686, 1712, 1728, 1802, 1817, 1827, 1856, 1925, 1934, 1943, 1948, 1953, 1956, 1976, 1980, 1996, 2018, 2030, 2041, 2049, 2075, 2097, 2136, 2139, 2244, 2258, 2389, 2401, 2410, 2419, 2540, 2544, 2568, 2801, 2877, 2896, 2900, 3002, 3039, 3043, 3053, 3057, 3098, 3283, 3511, 3512, 3594, 3616, 3656, 3659, 3663, 3669, 3727, 4026, 4067, 4069, 4234, 4239, 4243, 4252, 4258, 4321, 4828, 4832, 5069, 5086, 5102, 5103, 5319, 5321, 5322, 5324, 5325, 5863, 6151, 6332, 6335, 6354, 6362, 6370, 6599, 6644, 6676, 6677, 7293, 7299, 7343, 7737, 7740, 7760, 7787, 7792, 7801, 7820, 7824, 7869, 7887, 7899, 7914, 7916, 7972, 8039, 8192, 8202, 8241, 8243, 8260, 8264, 8308, 8309, 8314, 8346, 8364, 8380, 8400, 8412, 8413, 8424, 8432, 8443, 8447, 8460, 8514, 8520, 8523, 8549, 8697, 8707, 8851, 8860, 8863, 8866, 8877, 8881, 9056, 9102, 9130, 9205, 9222, 9256, 9481, 9678, 9682, 9729, 10277, 10288, 10376, 10382, 10388, 10394, 10405, 10612, 10655, 10959, 11108, 11160, 11226, 11240, 11434, 11574, 11670, 11688, 11693, 11818, 11904, 11906, 11967, 11995, 12005, 12945, 12948, 13485, 13488, 13541, 13544, 13858, 13881, 14016, 14063, 14123, 14128, 14131, 14134, 14193, 14727, 14747, 14818, 14819, 14881, 14897, 14959, 15118, 15361, 15364, 15411, 22206, 22211, 22844, 22886, 23213, 23229, 23246, 23562, 23865, 23871, 24140, 24153, 24173, 24227, 24240, 24263, 24288, 24313, 24472, 24476, 24501, 24511, 24542, 24966, 25184, 25194, 25307, 25320, 25947, 26170, 26175, 26465, 26471, 26593, 26653, 26693, 26838, 26858, 26939, 27594, 28393, 28422, 28461, 28611, 28949, 29010, 29270, 29338, 29399, 29648, 29686, 29729, 30373, 30405, 30453, 39307, 41652, 42392, 43687, 44248, 44871, 45397, 45570, 45576, 45631, 45644, 45658, 45724, 46073, 46492, 47133, 47229, 47492, 47945, 48408, 48596, 48966, 49004, 49018, 49035, 49051, 49307, 49320, 49322, 49602, 49895, 49898, 49913, 49951, 50023, 50528, 50530, 50537, 50543, 50911, 50948, 50953, 51001, 51076, 51657, 51662, 51680, 51693, 51722]

# Target building ID inside the CityGML
TARGET_GML_ID = "DEBE01YYK0002Uqm"

# Output folder
OUTPUT_DIR = r"C:\WF\Thomas Sharon\Floorplan_Dataset\gml_msd"
# OUTPUT_DIR = r"N:\9_SF-Public\Austausch\Thomas Sharon\Master_Thesis_Sharon\Floorplan_Dataset\gml_msd"


# === MAIN PIPELINE ===
def main():

    print("\n======================================")
    print("🏗️  BATCH START: MULTIPLE BUILDINGS")
    print("======================================")

    for bid in BUILDING_IDS:

        print(f"\n\n===============================")
        print(f"➡️ Processing Swiss building ID: {bid}")
        print("===============================")

        # Output paths (unique per building)
        output_geojson = os.path.join(OUTPUT_DIR, "footprint",f"footprint_{bid}.geojson")
        output_gml     = os.path.join(OUTPUT_DIR, "LoD2", f"LoD2_Berlin_Moabit_replaced_{bid}.gml")

        # --- STEP 1: Extract footprint ---
        print("\n🏗️ STEP 1: Generating Swiss footprint…")
        geojson_fp = create_footprint_geojson(
            building_id=bid,
            datapath=DATAPATH,
            output_geojson=output_geojson
        )

        # --- STEP 2: Replace footprint in GML ---
        print("\n🏙️ STEP 2: Replacing building footprint in CityGML…")
        result_gml = replace_building_footprint(
            input_gml=INPUT_GML,
            output_gml=output_gml,
            target_id=TARGET_GML_ID,
            geojson_fp=geojson_fp
        )

        print("\n✅ DONE")
        print(f"📁 GeoJSON footprint saved to: {geojson_fp}")
        print(f"📁 Updated GML saved to:       {result_gml}")

    print("\n======================================")
    print("🎉 BATCH COMPLETED FOR ALL BUILDINGS")
    print("======================================\n")


if __name__ == "__main__":
    main()
 