import pandas as pd
import numpy as np
from pprint import pprint

# ============================================================
# PATHS
# ============================================================
in_csv  = r"C:\WF\Thomas Sharon\Floorplan_Dataset\msd_dataset_creation\msd_apt_list_1.csv"
out_csv = r"C:\WF\Thomas Sharon\Floorplan_Dataset\msd_dataset_creation\msd_apt_list_swiss_class.csv"

# ============================================================
# LOAD + MEAN APARTMENT AREA
# ============================================================
df = pd.read_csv(in_csv)

apt_cols = [f"apt_{i}_area" for i in range(1, 9)]
df["mean_apartment_area"] = df[apt_cols].mean(axis=1, skipna=True)

# keep only valid rows
df = df.replace([np.inf, -np.inf], np.nan)
df = df[df["mean_apartment_area"].notna()].copy()

# ============================================================
# SWISS UNIT SIZE "TARGETS" FROM YOUR DATA (QUANTILES)
#   - This creates Swiss reference sizes based on the dataset
# ============================================================
q25 = float(df["mean_apartment_area"].quantile(0.25))
q50 = float(df["mean_apartment_area"].quantile(0.50))
q75 = float(df["mean_apartment_area"].quantile(0.75))
q95 = float(df["mean_apartment_area"].quantile(0.95))  # robust "very large" target

swiss_unit_sizes = {
    "small_swiss": round(q25, 1),
    "medium_swiss": round(q50, 1),
    "large_swiss": round(q75, 1),
    "very_large_swiss": round(q95, 1),
}

print("\n✅ Swiss unit size targets derived from your data:")
pprint(swiss_unit_sizes)

# ============================================================
# ASSIGN A SWISS CLASS (NEAREST TARGET SIZE)
#   - analogous to your German nearest-target mapping
# ============================================================
def assign_swiss_unit_class(mean_area: float) -> str:
    return min(swiss_unit_sizes, key=lambda k: abs(mean_area - swiss_unit_sizes[k]))

df["swiss_unit_class"] = df["mean_apartment_area"].apply(assign_swiss_unit_class)
df["swiss_unit_target_m2"] = df["swiss_unit_class"].map(swiss_unit_sizes)

# ============================================================
# COLLECT IDS PER SWISS CLASS (WHAT YOU WANTED)
# ============================================================
ids_by_swiss_class = (
    df.groupby("swiss_unit_class")["building_id"]
      .apply(lambda x: sorted(x.dropna().astype(int).tolist()))
      .to_dict()
)

print("\n✅ Building IDs per Swiss class:")
pprint(ids_by_swiss_class)

# ============================================================
# WRITE OUTPUT CSV (OPTIONAL)
# ============================================================
df.to_csv(out_csv, index=False)
print(f"\n✅ CSV written with Swiss unit class columns: {out_csv}")
print(f"Rows written: {len(df)}")
