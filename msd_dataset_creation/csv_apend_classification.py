import pandas as pd

# === PATHS ===
in_csv = r"C:\WF\Thomas Sharon\Floorplan_Dataset\msd_dataset_creation\msd_apt_list_1.csv"
out_csv = r"C:\WF\Thomas Sharon\Floorplan_Dataset\msd_dataset_creation\msd_apt_list_2.csv"

# === LOAD CSV ===
df = pd.read_csv(in_csv)

# === APARTMENT AREA COLUMNS ===
apt_cols = [f"apt_{i}_area" for i in range(1, 9)]

# === MEAN APARTMENT AREA PER BUILDING ===
df["mean_apartment_area"] = df[apt_cols].mean(axis=1, skipna=True)

# Safety
df = df[df["mean_apartment_area"].notna()]

# === GERMAN MFH UNIT SIZE CLASSES ===
german_unit_sizes = {
    "pre_1919": 65.0,
    "1919_1945": 65.0,
    "1946_1970": 68.0,
    "1971_1990": 70.0,
    "1991_1994": 72.0,
    "post_1995": 75.0,
    "fallback": 94.0
}

# === ASSIGN CLOSEST GERMAN UNIT CLASS ===
def assign_german_unit_class(mean_area):
    return min(
        german_unit_sizes,
        key=lambda k: abs(mean_area - german_unit_sizes[k])
    )

df["german_unit_class"] = df["mean_apartment_area"].apply(assign_german_unit_class)

# (optional but useful)
df["german_unit_target_m2"] = df["german_unit_class"].map(german_unit_sizes)

# === WRITE NEW CSV ===
df.to_csv(out_csv, index=False)

print(f"✅ New CSV written with German unit class appended:")
print(out_csv)
print(f"Rows written: {len(df)}")
