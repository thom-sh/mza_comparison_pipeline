import pandas as pd

# === PATH TO CSV (WILL BE OVERWRITTEN) ===
csv_path = r"C:\WF\Thomas Sharon\Floorplan_Dataset\msd_dataset_creation\msd_apt_list_2.csv"

# === BUILDING IDs TO REMOVE ===
ids_to_remove = [594,1322,4872]

# === LOAD CSV ===
df = pd.read_csv(csv_path)

# === REMOVE ROWS ===
mask_remove = df["building_id"].isin(ids_to_remove)
num_removed = mask_remove.sum()

df_cleaned = df[~mask_remove]

# === OVERWRITE CSV ===
df_cleaned.to_csv(csv_path, index=False)

# === REPORT ===
print(f"✅ Removed {num_removed} rows.")
print(f"📁 File overwritten:\n{csv_path}")
print(f"📊 Remaining rows: {len(df_cleaned)}")
