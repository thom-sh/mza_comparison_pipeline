"""
Batch pipeline for generating CityGML input files for MZA.

The same GML footprint replacement workflow is used for both datasets:

1. RD pipeline
   - Reads ground-truth pickle files from:
     rd_ground_truth_extraction/data/ground_truth/
   - Reads full building ID list from:
     rd_ground_truth_extraction/data/rd_thesis_building_ids.txt
   - Writes generated GeoJSON and GML files to:
     gml_footprint_replacement/data/rd/

2. MSD pipeline
   - Reads ground-truth pickle files from:
     msd_ground_truth_extraction/data/ground_truth/
   - Reads full building ID list from:
     msd_ground_truth_extraction/data/msd_thesis_building_ids.txt
   - Writes generated GeoJSON and GML files to:
     gml_footprint_replacement/data/msd/

The footprint extraction and GML replacement scripts are shared.
Only the dataset, ID mode, input paths, and output folders change.
"""

from pathlib import Path

from footprint_extractor import create_footprint_geojson
from gml_footprint_replacer import replace_building_footprint


# ============================================================
# USER CONFIGURATION
# ============================================================

# Choose which dataset pipeline to run.
# Options: "rd" or "msd"
DATASET = "rd"

# Choose how building IDs are selected.
# Options:
#   "batch"    -> use the full building ID list from the dataset txt file
#   "selected" -> use the manually defined IDs below
ID_MODE = "selected"

# Manually selected building IDs.
# Used only when ID_MODE = "selected".
SELECTED_BUILDING_IDS_BY_DATASET = {
    "rd": [33, 34, 35,],
    "msd": [68, 75],
}

# Target building ID inside the CityGML template.
# This is the placeholder building whose footprint will be replaced.
TARGET_GML_ID = "DEBE01YYK0002Uqm"


# ============================================================
# PATH CONFIGURATION
# ============================================================

# Folder where this main.py file is located:
# .../gml_footprint_replacement
PROJECT_DIR = Path(__file__).resolve().parent

# Repository root:
# .../MZA_Thesis
REPO_DIR = PROJECT_DIR.parent

# Local data folder inside gml_footprint_replacement
DATA_DIR = PROJECT_DIR / "data"

# Shared input CityGML template used for both RD and MSD
INPUT_GML = DATA_DIR / "gml_template" / "LoD2_Berlin_Moabit_neu.gml"

# Dataset-specific input and output folders
DATASET_CONFIG = {
    "rd": {
        "input_pickle_dir": REPO_DIR / "rd_ground_truth_extraction" / "data" / "ground_truth",
        "building_id_file": REPO_DIR / "rd_ground_truth_extraction" / "data" / "rd_thesis_building_ids.txt",
        "output_dir": DATA_DIR / "rd",
    },
    "msd": {
        "input_pickle_dir": REPO_DIR / "msd_ground_truth_extraction" / "data" / "ground_truth",
        "building_id_file": REPO_DIR / "msd_ground_truth_extraction" / "data" / "msd_thesis_building_ids.txt",
        "output_dir": DATA_DIR / "msd",
    },
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_building_ids(ids_file: Path) -> list[int]:
    """
    Load building IDs from a text file.

    The file can contain IDs in one of these forms:
    - one ID per line
    - comma-separated IDs
    - space-separated IDs

    Empty lines and lines starting with '#' are ignored.
    """

    if not ids_file.exists():
        raise FileNotFoundError(f"Building ID file not found:\n{ids_file}")

    ids = []

    with ids_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            # Allow comma-separated or space-separated IDs
            line = line.replace(",", " ")
            parts = line.split()

            for part in parts:
                ids.append(int(part))

    if not ids:
        raise ValueError(f"No building IDs found in:\n{ids_file}")

    return ids


def get_building_ids(dataset: str, id_mode: str) -> list[int]:
    """
    Return the building IDs for the selected dataset and mode.

    id_mode = "batch":
        Reads all IDs from the dataset-specific text file.

    id_mode = "selected":
        Uses the manually defined list in SELECTED_BUILDING_IDS_BY_DATASET.
    """

    if id_mode == "batch":
        ids_file = DATASET_CONFIG[dataset]["building_id_file"]
        return load_building_ids(ids_file)

    if id_mode == "selected":
        ids = SELECTED_BUILDING_IDS_BY_DATASET.get(dataset, [])

        if not ids:
            raise ValueError(
                f"No selected building IDs defined for dataset '{dataset}'."
            )

        return ids

    raise ValueError("Invalid ID_MODE. Use 'batch' or 'selected'.")


def validate_config(dataset: str, id_mode: str) -> None:
    """Check whether the selected dataset, ID mode, and required paths are valid."""

    if dataset not in DATASET_CONFIG:
        raise ValueError(
            f"Invalid DATASET='{dataset}'. Use one of: {list(DATASET_CONFIG.keys())}"
        )

    if id_mode not in ["batch", "selected"]:
        raise ValueError("Invalid ID_MODE. Use 'batch' or 'selected'.")

    input_pickle_dir = DATASET_CONFIG[dataset]["input_pickle_dir"]
    building_id_file = DATASET_CONFIG[dataset]["building_id_file"]

    if not INPUT_GML.exists():
        raise FileNotFoundError(f"Input GML template not found:\n{INPUT_GML}")

    if not input_pickle_dir.exists():
        raise FileNotFoundError(
            f"Input pickle folder not found for dataset '{dataset}':\n{input_pickle_dir}"
        )

    if id_mode == "batch" and not building_id_file.exists():
        raise FileNotFoundError(
            f"Building ID list not found for dataset '{dataset}':\n{building_id_file}"
        )


# ============================================================
# MAIN PIPELINE
# ============================================================

def main() -> None:
    validate_config(DATASET, ID_MODE)

    input_pickle_dir = DATASET_CONFIG[DATASET]["input_pickle_dir"]
    output_dir = DATASET_CONFIG[DATASET]["output_dir"]
    building_ids = get_building_ids(DATASET, ID_MODE)

    # Output subfolders
    output_footprint_dir = output_dir / "footprint"
    output_gml_dir = output_dir / "gml_replaced"

    output_footprint_dir.mkdir(parents=True, exist_ok=True)
    output_gml_dir.mkdir(parents=True, exist_ok=True)

    print("\n======================================")
    print(f"BATCH START: {DATASET.upper()} GML FOOTPRINT REPLACEMENT")
    print("======================================")
    print(f"Dataset:             {DATASET}")
    print(f"ID mode:             {ID_MODE}")
    print(f"Number of buildings: {len(building_ids)}")
    print(f"Input pickle folder: {input_pickle_dir}")
    print(f"Input GML template:  {INPUT_GML}")
    print(f"Output folder:       {output_dir}")

    for bid in building_ids:
        print("\n\n===============================")
        print(f"Processing {DATASET.upper()} building ID: {bid}")
        print("===============================")

        # Output paths for this building
        output_geojson = output_footprint_dir / f"footprint_{bid}.geojson"
        output_gml = output_gml_dir / f"LoD2_Berlin_Moabit_replaced_{bid}.gml"

        # ----------------------------------------------------
        # STEP 1: Extract footprint from ground-truth pickle
        # ----------------------------------------------------
        print("\nSTEP 1: Generating footprint GeoJSON...")

        geojson_fp = create_footprint_geojson(
            building_id=bid,
            datapath=str(input_pickle_dir),
            output_geojson=str(output_geojson),
        )

        # ----------------------------------------------------
        # STEP 2: Replace template CityGML footprint
        # ----------------------------------------------------
        print("\nSTEP 2: Replacing building footprint in CityGML...")

        result_gml = replace_building_footprint(
            input_gml=str(INPUT_GML),
            output_gml=str(output_gml),
            target_id=TARGET_GML_ID,
            geojson_fp=str(geojson_fp),
        )

        print("\nDONE")
        print(f"GeoJSON footprint saved to: {geojson_fp}")
        print(f"Updated GML saved to:       {result_gml}")

    print("\n======================================")
    print(f"BATCH COMPLETED FOR {DATASET.upper()}")
    print("======================================\n")


if __name__ == "__main__":
    main()