import matplotlib.pyplot as plt
import numpy as np
import math
from pdf2image import convert_from_path

# ----- Global interactive state -----
current_polygon = []
all_polygons = []
colors = []
snap_distance = 10
vertex_snap_distance = 10

fig = None
ax = None

img_crop = None
CW = None
CH = None

# Store original margin coordinates
L = 0
T = 0
R = 0
B = 0


def dist(a, b):
    return math.dist(a, b)


def random_color():
    import random
    return (random.random(), random.random(), random.random())


def get_all_vertices():
    verts = []
    for p in all_polygons:
        verts.extend(p)
    return verts


def find_nearest(pt):
    best = None
    dmin = float("inf")
    for v in get_all_vertices():
        d = dist(pt, v)
        if d < dmin and d <= vertex_snap_distance:
            dmin = d
            best = v
    return best


def detect_margins(img):
    gray = np.mean(img, axis=2)
    mask = gray < 250
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    top = np.argmax(rows)
    bottom = len(rows) - np.argmax(rows[::-1])

    left = np.argmax(cols)
    right = len(cols) - np.argmax(cols[::-1])

    cropped = img[top:bottom, left:right]
    return cropped, (left, top, right, bottom)


def redraw():
    ax.clear()
    ax.imshow(img_crop, extent=[0, CW, CH, 0])

    for poly, col in zip(all_polygons, colors):
        xs = [p[0] for p in poly] + [poly[0][0]]
        ys = [p[1] for p in poly] + [poly[0][1]]
        ax.plot(xs, ys, color=col, linewidth=2)

    if current_polygon:
        xs = [p[0] for p in current_polygon]
        ys = [p[1] for p in current_polygon]
        ax.plot(xs, ys, "r-o")

    fig.canvas.draw_idle()


def onclick(event):
    if event.xdata is None:
        return

    p = (event.xdata, event.ydata)
    snap = find_nearest(p) or p

    if not current_polygon:
        current_polygon.append(snap)
        redraw()
        return

    if len(current_polygon) >= 3 and dist(snap, current_polygon[0]) < snap_distance:
        all_polygons.append(current_polygon.copy())
        colors.append(random_color())
        current_polygon.clear()
        redraw()
        return

    current_polygon.append(snap)
    redraw()


def onkey(event):
    if event.key == "q":
        plt.close()
    if event.key == "ctrl+z" and current_polygon:
        current_polygon.pop()
        redraw()


def draw_polygons(pdf_path, dpi=300):
    global img_crop, CW, CH, fig, ax
    global current_polygon, all_polygons, colors
    global L, T, R, B  # store margins

    current_polygon = []
    all_polygons = []
    colors = []

    # Load PDF
    img = np.array(convert_from_path(pdf_path, dpi=dpi)[0])
    H, W = img.shape[:2]

    # Crop margins
    img_crop, (L, T, R, B) = detect_margins(img)
    CH, CW = img_crop.shape[:2]

    # GUI
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img_crop, extent=[0, CW, CH, 0])
    ax.set_xlim(0, CW)
    ax.set_ylim(CH, 0)
    ax.set_aspect("equal")

    fig.canvas.mpl_connect("button_press_event", onclick)
    fig.canvas.mpl_connect("key_press_event", onkey)

    plt.show()

    # Convert cropped→full coordinates
    full_polygons = []
    for poly in all_polygons:
        full_poly = [(x + L, y + T) for x, y in poly]
        full_polygons.append(full_poly)

    return full_polygons, img_crop, CW, CH, W, H
