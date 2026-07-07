"""
Batch pipeline for generating CityGML input files for the external MZA workflow.

The same footprint replacement workflow is used for both datasets:

- RD reads ground-truth pickles from rd_ground_truth_extraction/data/ground_truth/
- MSD reads ground-truth pickles from msd_ground_truth_extraction/data/ground_truth/
- Generated GeoJSON footprints and CityGML files are written to
  gml_footprint_replacement/data/<dataset>/

Typical use from this folder:

    python main.py --dataset both --id-mode batch
    python main.py --dataset rd --ids 33 34 35
    python main.py --dataset msd --ids 68,75
"""

from __future__ import annotations

import argparse
from pathlib import Path

from footprint_extractor import create_footprint_geojson
from gml_footprint_replacer import replace_building_footprint


# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

# Target building ID inside the CityGML template. This is the placeholder
# building whose footprint will be replaced.
TARGET_GML_ID = "DEBE01YYK0002Uqm"

# Folder where this main.py file is located:
# .../gml_footprint_replacement
PROJECT_DIR = Path(__file__).resolve().parent

# Repository root.
REPO_DIR = PROJECT_DIR.parent

# Local data folder inside gml_footprint_replacement.
DATA_DIR = PROJECT_DIR / "data"

# Shared input CityGML template used for both RD and MSD.
INPUT_GML = DATA_DIR / "gml_template" / "LoD2_Berlin_Moabit_neu.gml"

# Dataset-specific input and output folders.
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

def parse_building_ids(values: list[str] | None) -> list[int] | None:
    """Parse IDs passed as repeated args, comma-separated args, or both."""

    if not values:
        return None

    ids: list[int] = []
    for value in values:
        for part in value.replace(",", " ").split():
            ids.append(int(part))

    if not ids:
        return None

    return ids


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

    ids: list[int] = []

    with ids_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            for part in line.replace(",", " ").split():
                ids.append(int(part))

    if not ids:
        raise ValueError(f"No building IDs found in:\n{ids_file}")

    return ids


def get_building_ids(
    dataset: str,
    id_mode: str,
    selected_ids: list[int] | None,
) -> list[int]:
    """
    Return selected IDs for one dataset.

    id_mode = "batch":
        Reads all IDs from the dataset-specific text file.

    id_mode = "selected":
        Uses IDs passed through --ids.
    """

    if id_mode == "batch":
        ids_file = DATASET_CONFIG[dataset]["building_id_file"]
        return load_building_ids(ids_file)

    if id_mode == "selected":
        if not selected_ids:
            raise ValueError("ID mode 'selected' requires at least one --ids value.")
        return selected_ids

    raise ValueError("Invalid id_mode. Use 'batch' or 'selected'.")


def validate_dataset_config(dataset: str, id_mode: str) -> None:
    """Check whether the selected dataset and required paths are valid."""

    if dataset not in DATASET_CONFIG:
        raise ValueError(
            f"Invalid dataset '{dataset}'. Use one of: {list(DATASET_CONFIG.keys())}"
        )

    if id_mode not in ["batch", "selected"]:
        raise ValueError("Invalid id_mode. Use 'batch' or 'selected'.")

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


def validate_ground_truth_files(dataset: str, building_ids: list[int]) -> None:
    """Fail early when an ID is missing its ground-truth pickle."""

    input_pickle_dir = DATASET_CONFIG[dataset]["input_pickle_dir"]
    missing = [
        building_id
        for building_id in building_ids
        if not (input_pickle_dir / f"{building_id}.pickle").exists()
    ]

    if missing:
        preview = ", ".join(str(x) for x in missing[:20])
        suffix = "" if len(missing) <= 20 else f", ... ({len(missing)} missing total)"
        raise FileNotFoundError(
            f"Missing ground-truth pickle files for {dataset}: {preview}{suffix}"
        )


def run_dataset(
    dataset: str,
    id_mode: str,
    selected_ids: list[int] | None,
    target_gml_id: str,
    show_plot: bool,
    skip_existing: bool,
) -> None:
    """Generate GeoJSON footprints and replaced CityGML files for one dataset."""

    validate_dataset_config(dataset, id_mode)

    input_pickle_dir = DATASET_CONFIG[dataset]["input_pickle_dir"]
    output_dir = DATASET_CONFIG[dataset]["output_dir"]
    building_ids = get_building_ids(dataset, id_mode, selected_ids)
    validate_ground_truth_files(dataset, building_ids)

    output_footprint_dir = output_dir / "footprint"
    output_gml_dir = output_dir / "gml_replaced"

    output_footprint_dir.mkdir(parents=True, exist_ok=True)
    output_gml_dir.mkdir(parents=True, exist_ok=True)

    print("\n======================================")
    print(f"BATCH START: {dataset.upper()} GML FOOTPRINT REPLACEMENT")
    print("======================================")
    print(f"Dataset:             {dataset}")
    print(f"ID mode:             {id_mode}")
    print(f"Number of buildings: {len(building_ids)}")
    print(f"Input pickle folder: {input_pickle_dir}")
    print(f"Input GML template:  {INPUT_GML}")
    print(f"Output folder:       {output_dir}")

    for index, building_id in enumerate(building_ids, start=1):
        output_geojson = output_footprint_dir / f"footprint_{building_id}.geojson"
        output_gml = output_gml_dir / f"LoD2_Berlin_Moabit_replaced_{building_id}.gml"

        if skip_existing and output_geojson.exists() and output_gml.exists():
            print(
                f"\n[{index}/{len(building_ids)}] Skipping {dataset.upper()} "
                f"{building_id}: outputs already exist."
            )
            continue

        print("\n\n===============================")
        print(f"[{index}/{len(building_ids)}] Processing {dataset.upper()} building ID: {building_id}")
        print("===============================")

        print("\nSTEP 1: Generating footprint GeoJSON...")
        geojson_fp = create_footprint_geojson(
            building_id=building_id,
            datapath=str(input_pickle_dir),
            output_geojson=str(output_geojson),
            show_plot=show_plot,
        )

        print("\nSTEP 2: Replacing building footprint in CityGML...")
        result_gml = replace_building_footprint(
            input_gml=str(INPUT_GML),
            output_gml=str(output_gml),
            target_id=target_gml_id,
            geojson_fp=str(geojson_fp),
        )

        print("\nDONE")
        print(f"GeoJSON footprint saved to: {geojson_fp}")
        print(f"Updated GML saved to:       {result_gml}")

    print("\n======================================")
    print(f"BATCH COMPLETED FOR {dataset.upper()}")
    print("======================================\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate footprint-based CityGML files for the external MZA workflow."
    )
    parser.add_argument(
        "--dataset",
        choices=["rd", "msd", "both"],
        default="both",
        help="Dataset to process. Default: both.",
    )
    parser.add_argument(
        "--id-mode",
        choices=["batch", "selected"],
        default="batch",
        help="Use all thesis IDs from file or only IDs passed with --ids. Default: batch.",
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        help="Building IDs for --id-mode selected. Accepts spaces and commas.",
    )
    parser.add_argument(
        "--target-gml-id",
        default=TARGET_GML_ID,
        help=f"Placeholder building ID inside the template GML. Default: {TARGET_GML_ID}.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show a Matplotlib preview for each generated footprint.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip IDs where both the GeoJSON and GML output already exist.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    selected_ids = parse_building_ids(args.ids)

    if args.dataset == "both" and args.id_mode == "selected":
        raise ValueError("Use --dataset rd or --dataset msd with --id-mode selected.")

    datasets = ["rd", "msd"] if args.dataset == "both" else [args.dataset]

    for dataset in datasets:
        run_dataset(
            dataset=dataset,
            id_mode=args.id_mode,
            selected_ids=selected_ids,
            target_gml_id=args.target_gml_id,
            show_plot=args.plot,
            skip_existing=args.skip_existing,
        )


if __name__ == "__main__":
    main()
