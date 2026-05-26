import os
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.geometry import box
from skimage import measure


def structure_mask_to_polygons(struct_file, min_area=0.0, plot=False):
    """
    Convert MSD structure mask (channel 0 of struct_in/*.npy) to polygons
    using the x/y coordinate channels.

    Parameters
    ----------
    struct_file : str
        Path to struct_in/<id>.npy
    min_area : float
        Minimum polygon area to retain
    plot : bool
        If True, show the mask and extracted polygons

    Returns
    -------
    polygons : list of shapely.geometry.Polygon
        List of extracted structure polygons
    """

    stack = np.load(struct_file)

    # channel 0 = structure mask
    # channel 1 = x coordinates
    # channel 2 = y coordinates
    struct = stack[..., 0].astype(np.uint8)
    x = stack[..., 1]
    y = stack[..., 2]

    # find contours at the boundary between 0 and 1
    contours = measure.find_contours(struct, level=0.5)

    polygons = []

    for contour in contours:
        # contour is in (row, col) image coordinates
        rows = contour[:, 0]
        cols = contour[:, 1]

        # convert from pixel coordinates to real x/y coordinates
        # use nearest-neighbour indexing
        rr = np.clip(np.round(rows).astype(int), 0, struct.shape[0]-1)
        cc = np.clip(np.round(cols).astype(int), 0, struct.shape[1]-1)

        coords = np.column_stack([x[rr, cc], y[rr, cc]])

        # make polygon
        poly = Polygon(coords)

        if poly.is_valid and not poly.is_empty and poly.area > min_area:
            polygons.append(poly)

    # optional: merge touching polygons
    polygons = [p for p in polygons if p.is_valid]
    
    if plot:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(struct, cmap="gray", origin="lower",
                  extent=[x.min(), x.max(), y.min(), y.max()],
                  alpha=0.5)

        for poly in polygons:
            x_poly, y_poly = poly.exterior.xy
            ax.plot(x_poly, y_poly, "r-", linewidth=2)

        ax.set_aspect("equal")
        ax.set_title("Extracted structure polygons")
        plt.show()

    return polygons

def structure_mask_to_filled_polygons(struct_file, min_area=0.01, simplify_tol=0.02, plot=False):
    """
    Convert structure mask to filled shapely polygons by turning each occupied pixel
    into a small box in world coordinates and unioning them.
    """

    stack = np.load(struct_file)

    raw = stack[..., 0]
    mask = raw < 128
    x = stack[..., 1]
    y = stack[..., 2]

    # estimate pixel size from coordinate channels
    dx = float(np.median(np.abs(np.diff(x[0, :]))))
    dy = float(np.median(np.abs(np.diff(y[:, 0]))))

    rr, cc = np.where(mask)

    pixel_boxes = [
        box(
            x[r, c] - dx / 2,
            y[r, c] - dy / 2,
            x[r, c] + dx / 2,
            y[r, c] + dy / 2
        )
        for r, c in zip(rr, cc)
    ]

    if not pixel_boxes:
        return []

    geom = unary_union(pixel_boxes)

    if simplify_tol > 0:
        geom = geom.simplify(simplify_tol, preserve_topology=True)

    if geom.geom_type == "Polygon":
        polygons = [geom]
    elif geom.geom_type == "MultiPolygon":
        polygons = [g for g in geom.geoms if g.area >= min_area]
    else:
        polygons = []

    if plot:
        fig, ax = plt.subplots(figsize=(8, 8))
        for poly in polygons:
            x_poly, y_poly = poly.exterior.xy
            ax.fill(x_poly, y_poly, facecolor="none", edgecolor="red", linewidth=2)
        ax.set_aspect("equal")
        ax.set_title("Filled structure polygons")
        vals, counts = np.unique(stack[..., 0], return_counts=True)
        print(list(zip(vals, counts)))
        plt.show()

    return polygons

def main():
    datapath = r"C:\WF\Thomas Sharon\Floorplan_Dataset\archive\modified-swiss-dwellings-v2\train"
    IDs = [68]

    for building_id in IDs:
        struct_file = os.path.join(datapath, "struct_in", f"{building_id}.npy")
        polygons = structure_mask_to_filled_polygons(struct_file, min_area=0.01, plot=True)

        print(f"Extracted {len(polygons)} structure polygons")
        for i, p in enumerate(polygons):
            print(i, p.area)
        
if __name__ == "__main__":
    main()

