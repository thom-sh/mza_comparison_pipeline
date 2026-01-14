# display_polygons.py
# ------------------------------------------------------
# Display final scaled polygons (in meters) in Matplotlib
# ------------------------------------------------------

import matplotlib.pyplot as plt


def display_polygons(polygons_m):
    """
    polygons_m = [
        [(x1, y1), (x2, y2), ...],   # polygon 1 in meters
        [(x1, y1), (x2, y2), ...],   # polygon 2 in meters
        ...
    ]
    """

    if not polygons_m:
        print("No polygons to display.")
        return

    plt.figure(figsize=(10, 10))
    ax = plt.gca()

    # For setting axis limits
    all_x = []
    all_y = []

    for idx, poly in enumerate(polygons_m, 1):
        xs = [p[0] for p in poly] + [poly[0][0]]
        ys = [p[1] for p in poly] + [poly[0][1]]

        all_x.extend(xs)
        all_y.extend(ys)

        ax.plot(xs, ys, linewidth=2, label=f"Polygon {idx}")
        ax.scatter(xs, ys, s=10)

        # Label centroid
        cx = sum(xs[:-1]) / (len(xs) - 1)
        cy = sum(ys[:-1]) / (len(ys) - 1)
        ax.text(cx, cy, f"P{idx}", fontsize=12, ha="center", va="center")

    # Set equal aspect for metric correctness
    ax.set_aspect("equal", adjustable="box")

    # Auto zoom
    margin = 1  # meters
    xmin, xmax = min(all_x), max(all_x)
    ymin, ymax = min(all_y), max(all_y)

    ax.set_xlim(xmin - margin, xmax + margin)
    ax.set_ylim(ymin - margin, ymax + margin)

    # Add grid in meters
    ax.grid(True, linestyle="--", alpha=0.5)

    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_title("Scaled Floorplan Polygons (Real-World Meters)")
    ax.legend()

    plt.show()
