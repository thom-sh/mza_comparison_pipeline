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
DATAPATH = r"N:\9_SF-Public\Austausch\Thomas Sharon\Master_Thesis_Sharon\Floorplan_Dataset\archive\modified-swiss-dwellings-v2\train"

# Input CityGML file
INPUT_GML = r"C:\Sharon\rom_auto_multizoning\data\LoD2_Berlin_Moabit_neu.gml"

# ===== MULTIPLE SWISS BUILDING IDS HERE =====
# BUILDING_IDS = [68, 75, 553, 108, 124, 154, 176, 329, 341, 343, 367, 405, 443, 447, 461, 463, 467, 474, 477, 496, 524, 546, 559, 613, 696, 803, 807, 974, 993, 1261]     # ← EDIT THIS LIST

# BUILDING_IDS = [1588, 1602, 1663, 1686, 1939, 1943, 1956, 1972, 1996, 2075, 2097, 2244, 2258, 2389, 2538, 2542, 2751, 2894, 3451, 3594, 5443]

BUILDING_IDS = [5641, 6151, 7291, 7293, 7310, 7899, 8413, 8443, 8460, 8514, 8520, 8529, 8562, 8697, 8707, 8866, 8867, 9056, 9132, 9729, 9813, 10277, 10288, 10382, 10394, 11434, 11160]

# BUILDING_IDS = [1663, 1330, 1451, 1464, 1575, 1576]

# Target building ID inside the CityGML
TARGET_GML_ID = "DEBE01YYK0002Uqm"

# Output folder
OUTPUT_DIR = r"C:\WF\Thomas Sharon\Floorplan_Dataset\gml_msd"


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
        output_geojson = os.path.join(OUTPUT_DIR, f"footprint_{bid}.geojson")
        output_gml     = os.path.join(OUTPUT_DIR, f"LoD2_Berlin_Moabit_replaced_{bid}.gml")

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
 