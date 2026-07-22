import pandas as pd
from IPython.display import display

csv_path = r"C:\WF\Thomas Sharon\Floorplan_Dataset\msd_dataset_creation\msd_apt_list_2.csv"

df = pd.read_csv(csv_path)

display(list(df["building_id"]))

# print("\n".join(df["building_id"].dropna().astype(int).astype(str)))
