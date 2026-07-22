import pandas as pd

# === PATH ===
csv_path = r"C:\WF\Thomas Sharon\Floorplan_Dataset\msd_dataset_creation\msd_apt_list_2.csv"

# === LOAD CSV ===
df = pd.read_csv(csv_path)

# === GROUP BUILDING IDS BY GERMAN UNIT CLASS ===
ids_by_class = (
    df.groupby("german_unit_class")["building_id"]
      .apply(lambda x: sorted(x.dropna().astype(int).tolist()))
      .to_dict()
)

# === RESULT ===
ids_by_class
