"""
Filter comparison metric CSV files to the final selected 200 MSD building IDs.

Inputs:
  - all_global_stats.csv
  - all_region_metrics.csv

Outputs:
  - all_global_stats_selected_200.csv
  - all_region_metrics_selected_200.csv
  - selected_200_filter_report.txt

The script uses only the building_id column for filtering.
"""

from __future__ import annotations

import csv
from pathlib import Path


# ---------------------------------------------------------------------
# 1) Final selected 200 MSD building IDs
# ---------------------------------------------------------------------
SELECTED_BUILDING_IDS = [
    68, 75, 179, 329, 467, 696, 807, 1291, 1321, 1361, 1575, 1588, 1595,
    1601, 1663, 1686, 1712, 1728, 1817, 1925, 1934, 1953, 1996, 2018, 2030,
    2049, 2136, 2244, 2389, 2401, 2410, 2540, 2568, 2896, 3002, 3057, 3283,
    3594, 3669, 4026, 4234, 4239, 4258, 4321, 4832, 5069, 5086, 5102, 5103,
    5319, 5322, 5863, 6362, 6370, 6599, 6676, 7299, 7343, 7737, 7760, 7792,
    7824, 7869, 7899, 7914, 7916, 8039, 8202, 8241, 8260, 8264, 8308, 8309,
    8314, 8412, 8413, 8443, 8447, 8460, 8549, 8851, 8860, 8863, 8866, 8877,
    8881, 9205, 9678, 9729, 10277, 10388, 10405, 10655, 10959, 11226, 11434,
    11818, 11906, 11967, 13488, 13544, 13858, 14016, 14063, 14123, 14128,
    14131, 14747, 14818, 14819, 14881, 14897, 15364, 22206, 22211, 22844,
    22886, 23213, 23246, 23562, 23865, 23871, 24153, 24173, 24288, 24472,
    24476, 24501, 24542, 24966, 25184, 25307, 25320, 25947, 26170, 26175,
    26471, 26593, 26653, 26838, 26858, 26939, 28611, 28949, 29270, 29399,
    29686, 29729, 30405, 30453, 39307, 42392, 43687, 44248, 44871, 45570,
    45576, 45631, 45644, 45658, 45724, 46073, 46492, 47229, 47492, 48408,
    48966, 49004, 49035, 49051, 49320, 49951, 50528, 50530, 50537, 51001,
    51680, 51693, 176, 322, 405, 553, 712, 721, 803, 993, 1827, 1943, 1976,
    2801, 3039, 3616, 5325, 7801, 8364, 8424, 9222, 9682, 10288, 10376]


# ---------------------------------------------------------------------
# 2) Paths
# ---------------------------------------------------------------------
BASE_DIR = Path(r"C:\WF\Thomas Sharon\Floorplan_Dataset\comparison_logic")

GLOBAL_IN = BASE_DIR / "all_global_stats.csv"
REGION_IN = BASE_DIR / "all_region_metrics.csv"

GLOBAL_OUT = BASE_DIR / "all_global_stats_selected_200.csv"
REGION_OUT = BASE_DIR / "all_region_metrics_selected_200.csv"
REPORT_OUT = BASE_DIR / "selected_200_filter_report.txt"


# ---------------------------------------------------------------------
# 3) Helper functions
# ---------------------------------------------------------------------
def parse_building_id(value: str) -> int:
    """Convert building_id values safely, including values read as '75.0'."""
    value = str(value).strip()
    if value == "":
        raise ValueError("Empty building_id")
    return int(float(value))


def filter_csv_by_building_ids(input_csv: Path, output_csv: Path, selected_ids: set[int]) -> dict:
    """
    Filter input_csv to rows whose building_id is in selected_ids.
    Returns a small summary dictionary.
    """
    if not input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    total_rows = 0
    kept_rows = 0
    seen_ids: set[int] = set()
    skipped_bad_id = 0

    with input_csv.open("r", newline="", encoding="utf-8-sig") as f_in, \
         output_csv.open("w", newline="", encoding="utf-8") as f_out:

        reader = csv.DictReader(f_in)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {input_csv}")
        if "building_id" not in reader.fieldnames:
            raise ValueError(f"CSV has no 'building_id' column: {input_csv}")

        writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            total_rows += 1
            try:
                bid = parse_building_id(row["building_id"])
            except ValueError:
                skipped_bad_id += 1
                continue

            if bid in selected_ids:
                # Standardise building_id formatting as integer text in output.
                row["building_id"] = str(bid)
                writer.writerow(row)
                kept_rows += 1
                seen_ids.add(bid)

    missing_ids = sorted(selected_ids - seen_ids)

    return {
        "input": str(input_csv),
        "output": str(output_csv),
        "total_rows": total_rows,
        "kept_rows": kept_rows,
        "unique_buildings_kept": len(seen_ids),
        "missing_buildings": missing_ids,
        "skipped_bad_id_rows": skipped_bad_id,
    }


def main() -> None:
    selected_ids = set(SELECTED_BUILDING_IDS)

    if len(SELECTED_BUILDING_IDS) != 200 or len(selected_ids) != 200:
        raise ValueError(
            f"Expected exactly 200 unique IDs, got {len(SELECTED_BUILDING_IDS)} entries "
            f"and {len(selected_ids)} unique IDs."
        )

    global_summary = filter_csv_by_building_ids(GLOBAL_IN, GLOBAL_OUT, selected_ids)
    region_summary = filter_csv_by_building_ids(REGION_IN, REGION_OUT, selected_ids)

    report_lines = [
        "Filter report for selected 200 MSD building IDs",
        "=" * 55,
        f"Number of selected IDs: {len(selected_ids)}",
        "",
        "Global stats CSV:",
        f"  input rows:             {global_summary['total_rows']}",
        f"  kept rows:              {global_summary['kept_rows']}",
        f"  unique buildings kept:  {global_summary['unique_buildings_kept']}",
        f"  missing buildings:      {len(global_summary['missing_buildings'])}",
        f"  skipped bad ID rows:    {global_summary['skipped_bad_id_rows']}",
        "",
        "Region metrics CSV:",
        f"  input rows:             {region_summary['total_rows']}",
        f"  kept rows:              {region_summary['kept_rows']}",
        f"  unique buildings kept:  {region_summary['unique_buildings_kept']}",
        f"  missing buildings:      {len(region_summary['missing_buildings'])}",
        f"  skipped bad ID rows:    {region_summary['skipped_bad_id_rows']}",
        "",
    ]

    all_missing = sorted(set(global_summary["missing_buildings"]) | set(region_summary["missing_buildings"]))
    if all_missing:
        report_lines.append("Missing selected building IDs in one or both files:")
        report_lines.append(", ".join(map(str, all_missing)))
    else:
        report_lines.append("All selected building IDs were found in both files.")

    REPORT_OUT.write_text("\n".join(report_lines), encoding="utf-8")

    print("[DONE] Filtered CSVs created:")
    print(f"  - {GLOBAL_OUT}")
    print(f"  - {REGION_OUT}")
    print(f"  - {REPORT_OUT}")


if __name__ == "__main__":
    main()
