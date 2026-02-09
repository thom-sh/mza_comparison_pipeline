import pandas as pd

# === PATH TO CSV ===
csv_path = r"C:\WF\Thomas Sharon\Floorplan_Dataset\msd_dataset_creation\msd_apt_list_3.csv"

# === BUILDING ID TO INSPECT ===
building_id = 5919   # change this ID as needed

# === LOAD CSV ===
df = pd.read_csv(csv_path)

# === SELECT ROW ===
row = df[df["building_id"] == building_id]

# === DISPLAY ===
if row.empty:
    print(f"❌ Building ID {building_id} not found.")
else:
    print(f"✅ Data for building ID {building_id}:\n")
    print(row.to_string(index=False))
