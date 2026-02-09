"""
Batch pipeline for:
1. Creating GeoJSON footprints from Swiss dataset
2. Replacing building footprints inside a CityGML file
"""

import os
from footprint_extractor import create_footprint_geojson
from gml_footprint_replacer import replace_building_footprint

# === CONFIGURATION ===

# Folder containing Swiss *.pickle files
DATAPATH = r"C:\WF\Thomas Sharon\Floorplan_Dataset\rd_pickle"

# Input CityGML file
INPUT_GML = r"C:\Sharon\rom_auto_multizoning_rd\data\LoD2_Berlin_Moabit_neu.gml"

# ===== MULTIPLE SWISS BUILDING IDS HERE =====
BUILDING_IDS = [16]     # ← EDIT THIS LIST

# Target building ID inside the CityGML
TARGET_GML_ID = "DEBE01YYK0002Uqm"

# Output folder
OUTPUT_DIR = r"C:\WF\Thomas Sharon\Floorplan_Dataset\gml_rd"


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
        output_geojson = os.path.join(OUTPUT_DIR, "footprint", f"footprint_{bid}.geojson")
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
