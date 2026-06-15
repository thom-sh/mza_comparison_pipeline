import os
import pickle
import matplotlib.pyplot as plt

# ---------- your display function ----------
def display_polygons(polygons_m):
    if not polygons_m:
        print("No polygons to display.")
        return

    plt.figure(figsize=(10, 10))
    ax = plt.gca()

    all_x, all_y = [], []

    for idx, poly in enumerate(polygons_m, 1):
        xs = [p[0] for p in poly] + [poly[0][0]]
        ys = [p[1] for p in poly] + [poly[0][1]]

        all_x.extend(xs)
        all_y.extend(ys)

        ax.plot(xs, ys, linewidth=2, label=f"Polygon {idx}")
        ax.scatter(xs, ys, s=10)

        cx = sum(xs[:-1]) / (len(xs) - 1)
        cy = sum(ys[:-1]) / (len(ys) - 1)
        ax.text(cx, cy, f"P{idx}", fontsize=12, ha="center", va="center")

    ax.set_aspect("equal", adjustable="box")

    margin = 1
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_title("Scaled Floorplan Polygons (Real-World Meters)")
    ax.legend()
    plt.show()


# ---------- polygon extraction helpers ----------
def looks_like_point(p):
    return isinstance(p, (list, tuple)) and len(p) >= 2 and isinstance(p[0], (int, float)) and isinstance(p[1], (int, float))

def looks_like_polygon(poly):
    return isinstance(poly, (list, tuple)) and len(poly) >= 3 and all(looks_like_point(p) for p in poly)

def normalize_polygon(poly):
    # ensure list of (float, float)
    return [(float(p[0]), float(p[1])) for p in poly]

def find_polygons_anywhere(obj, found, max_depth=6, depth=0):
    """Recursively scan nested dict/list/tuple for polygons."""
    if depth > max_depth:
        return

    if looks_like_polygon(obj):
        found.append(normalize_polygon(obj))
        return

    if isinstance(obj, dict):
        for v in obj.values():
            find_polygons_anywhere(v, found, max_depth, depth + 1)

    elif isinstance(obj, (list, tuple)):
        for item in obj:
            find_polygons_anywhere(item, found, max_depth, depth + 1)


# ---------- load + extract + plot ----------
DATAPATH = r"C:\WF\Thomas Sharon\Floorplan_Dataset\rd_pickle"
file_path = os.path.join(DATAPATH, "11.pickle")

with open(file_path, "rb") as f:
    data = pickle.load(f)

print("Top-level type:", type(data))
if isinstance(data, dict):
    print("Top-level keys (first 30):", list(data.keys())[:30])
elif isinstance(data, (list, tuple)):
    print("Top-level length:", len(data))

polygons_m = []
find_polygons_anywhere(data, polygons_m)

print(f"Found {len(polygons_m)} polygon(s).")
if polygons_m:
    print("Example polygon (first 3 points):", polygons_m[0][:3])

display_polygons(polygons_m)
