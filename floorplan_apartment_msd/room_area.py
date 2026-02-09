import os
import pandas as pd
from shapely.geometry import Polygon

from utils_apt import load_pickle
from constants_apt import ROOM_NAMES

# === CONFIG ===
datapath = r"C:\WF\Thomas Sharon\Floorplan_Dataset\archive\modified-swiss-dwellings-v2\train"
ID = 5919  # change as needed
graph_path = os.path.join(datapath, "graph_out", f"{ID}.pickle")

# === LOAD GRAPH ===
G = load_pickle(graph_path)

# Map type index -> readable name
IDX_TO_NAME = {i: name for i, name in enumerate(ROOM_NAMES)}

def safe_polygon(geom):
    if not geom:
        return None
    try:
        poly = Polygon(geom)
        if poly.is_valid and not poly.is_empty:
            return poly
    except Exception:
        return None
    return None

rows = []
for node_id, d in G.nodes(data=True):
    rtype_idx = d.get("room_type", None)
    rtype_name = IDX_TO_NAME.get(rtype_idx, f"UNKNOWN_{rtype_idx}")

    poly = safe_polygon(d.get("geometry"))
    area = float(poly.area) if poly is not None else None

    rows.append({
        "building_id": ID,
        "room_node_id": node_id,
        "room_type_idx": rtype_idx,
        "room_type_name": rtype_name,
        "room_area_m2": area
    })

df = pd.DataFrame(rows)

# Sort by room type then area (optional)
df = df.sort_values(["room_type_name", "room_area_m2"], ascending=[True, False])

# === DISPLAY ===
print(f"Building {ID} — rooms: {len(df)}")
print(df.to_string(index=False))

# Optional: quick totals per room type
print("\nTotal area per room type:")
print(df.groupby("room_type_name")["room_area_m2"].sum().sort_values(ascending=False).round(2))
