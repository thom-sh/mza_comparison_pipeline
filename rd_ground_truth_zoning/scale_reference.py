import matplotlib.pyplot as plt
import numpy as np
from draw_polygons import draw_polygons, img_crop, CW, CH

def scale_reference(pdf_path):
    polygons, img_crop, CW, CH, W, H = draw_polygons(pdf_path)

    print("\nSelect reference line (2 clicks).")

    ref_pts = []

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img_crop, extent=[0, CW, CH, 0])
    ax.set_xlim(0, CW)
    ax.set_ylim(CH, 0)
    ax.set_aspect("equal")
    ax.set_title("Click TWO points on reference line")

    def onclick(event):
        if event.xdata:
            ref_pts.append((event.xdata, event.ydata))
            if len(ref_pts) == 2:
                plt.close()

    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()

    if len(ref_pts) != 2:
        print("ERROR: No reference selected.")
        return polygons

    p1 = np.array(ref_pts[0])
    p2 = np.array(ref_pts[1])
    d_px = np.linalg.norm(p1 - p2)

    real_len = float(input("\nEnter REAL reference length (meters): "))

    px_to_m = real_len / d_px

    real_polys = []
    for poly in polygons:
        real_polys.append([(x * px_to_m, y * px_to_m) for x, y in poly])

    return real_polys
