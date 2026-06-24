import pandas as pd

# ===============================================================
# CONFIG
# ===============================================================
DIAGNOSTIC_CSV = r"C:\WF\Thomas Sharon\Floorplan_Dataset\gml_rd\comparison_output\diagnostic_table_with_topology_rd_final.csv"

OUT_CATEGORY_IDS_CSV = r"C:\WF\Thomas Sharon\Floorplan_Dataset\gml_rd\comparison_output\diagnostic_category_building_ids_rd_final.csv"
OUT_CATEGORY_IDS_TXT = r"C:\WF\Thomas Sharon\Floorplan_Dataset\gml_rd\comparison_output\diagnostic_category_building_ids_rd_final.txt"


# ===============================================================
# LOAD
# ===============================================================
df = pd.read_csv(DIAGNOSTIC_CSV)

if "building_id" not in df.columns:
    raise ValueError("CSV must contain a 'building_id' column.")

if "diagnosis" not in df.columns:
    raise ValueError("CSV must contain a 'diagnosis' column.")


# ===============================================================
# PRINT BUILDING IDS BY CATEGORY
# ===============================================================
category_rows = []
txt_lines = []

categories = df["diagnosis"].value_counts().index.tolist()

print("\n===================================================")
print("BUILDING IDS BY DIAGNOSTIC CATEGORY")
print("===================================================\n")

for category in categories:
    subset = df[df["diagnosis"] == category].copy()
    building_ids = sorted(subset["building_id"].astype(int).tolist())

    print(category)
    print("-" * len(category))
    print(f"Count: {len(building_ids)}")
    print(f"Building IDs: {building_ids}")
    print()

    txt_lines.append(category)
    txt_lines.append("-" * len(category))
    txt_lines.append(f"Count: {len(building_ids)}")
    txt_lines.append("Building IDs:")
    txt_lines.append(", ".join(map(str, building_ids)))
    txt_lines.append("")

    for bid in building_ids:
        category_rows.append({
            "diagnostic_category": category,
            "building_id": bid
        })


# ===============================================================
# SAVE OUTPUTS
# ===============================================================
df_out = pd.DataFrame(category_rows)
df_out.to_csv(OUT_CATEGORY_IDS_CSV, index=False)

with open(OUT_CATEGORY_IDS_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(txt_lines))

print("===================================================")
print("Saved:")
print(OUT_CATEGORY_IDS_CSV)
print(OUT_CATEGORY_IDS_TXT)
print("===================================================")