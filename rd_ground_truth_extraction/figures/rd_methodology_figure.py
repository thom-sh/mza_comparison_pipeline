import math
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.patches import Patch
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import sys

# ============================================================
# CONFIGURATION
# ============================================================
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

BUILDING_ID = 20

PDF_PATH = PROJECT_DIR / "data" / "raw_rd" / f"{BUILDING_ID}.pdf"

OUTPUT_DIR = PROJECT_DIR / "figures" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FIGURE = OUTPUT_DIR / f"rd_groundtruth_workflow_{BUILDING_ID}.pdf"

# Indices of polygons that represent the core/stair space.
# These indices refer to the drawing order during digitisation, starting from 0.
# Example: if you digitise [core, dwelling1, dwelling2], then use [0]
CORE_POLYGON_INDICES = [0]

RENDER_DPI = 250
CROP_THRESHOLD = 250
CROP_BORDER = 0

SNAP_TO_EXISTING_POINTS = True
SNAP_DISTANCE_PX = 12

# Area labels in panel (c)
SHOW_AREA_LABELS = True
AREA_LABEL_DECIMALS = 0
MIN_AREA_LABEL_M2 = 0.0


# ============================================================
# THESIS STYLE
# ============================================================

LEGEND_BOX_EDGE_COLOR = "#bdc1c5"
POLYGON_EDGE_COLOR = "#777d84"
CORE_EDGE_COLOR = "#777d84"

DWELLING_COLORS = [
    "#DCEAF7",
    "#AFCBE3",
    "#AFCBE3",
    "#DBDBDB",
    "#BABABA",
    "#999999",
]

CORE_COLOR = "#5F666D"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIX Two Text", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",

    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,

    "axes.titleweight": "normal",
    "axes.labelweight": "normal",

    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


# ============================================================
# PDF RENDERING AND CROPPING
# ============================================================

def render_pdf_first_page(pdf_path, dpi=250):
    doc = fitz.open(pdf_path)
    page = doc[0]

    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)

    img = np.frombuffer(
        pix.samples,
        dtype=np.uint8
    ).reshape(pix.height, pix.width, pix.n)

    doc.close()

    if img.shape[2] == 4:
        img = img[:, :, :3]

    return img


def crop_white_margin(img, threshold=250, border=0):
    """
    Crop white outer margins from an RGB image.
    Returns cropped image and bbox=(left, top, right, bottom).
    """
    gray = np.mean(img, axis=2)
    mask = gray < threshold

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not np.any(rows) or not np.any(cols):
        h, w = img.shape[:2]
        return img, (0, 0, w, h)

    top = np.argmax(rows)
    bottom = len(rows) - np.argmax(rows[::-1])
    left = np.argmax(cols)
    right = len(cols) - np.argmax(cols[::-1])

    left = max(left - border, 0)
    top = max(top - border, 0)
    right = min(right + border, img.shape[1])
    bottom = min(bottom + border, img.shape[0])

    return img[top:bottom, left:right], (left, top, right, bottom)


# ============================================================
# COLOUR HELPER
# ============================================================

def get_polygon_facecolor(idx, alpha=1.0):
    """
    Returns face colour, edge colour, and alpha for a polygon based on its index.
    Core polygons use CORE_COLOR.
    Other polygons use dwelling colours.
    """
    if idx in CORE_POLYGON_INDICES:
        return CORE_COLOR, CORE_EDGE_COLOR, alpha

    dwelling_idx = 0

    for j in range(idx + 1):
        if j not in CORE_POLYGON_INDICES:
            if j == idx:
                break
            dwelling_idx += 1

    face = DWELLING_COLORS[dwelling_idx % len(DWELLING_COLORS)]
    return face, POLYGON_EDGE_COLOR, alpha


# ============================================================
# INTERACTIVE DIGITISATION
# ============================================================

def digitise_polygons_interactively(img_crop):
    """
    Controls:
      - left click  = add point
      - right click = close current polygon
      - backspace   = undo last point
      - enter       = finish all polygons

    Snap logic:
      - if the clicked point is close to an already selected point,
        the existing point is reused exactly.
    """
    print("Digitise apartment and core polygons.")
    print("Draw polygons in this order:")
    print("  - draw core/stair polygons according to CORE_POLYGON_INDICES")
    print("  - draw dwelling polygons for the remaining spaces")
    print("Controls:")
    print("  - left click = add point")
    print("  - right click = close polygon")
    print("  - backspace = undo last point")
    print("  - enter = finish")
    print(f"  - snapping enabled within {SNAP_DISTANCE_PX} pixels")

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(img_crop)
    ax.set_title("Digitise polygons")
    ax.axis("off")

    polygons = []
    current_poly = []

    def get_all_existing_points():
        pts = []
        for poly in polygons:
            pts.extend(poly)
        pts.extend(current_poly)
        return pts

    def find_nearest_existing_point(x, y):
        if not SNAP_TO_EXISTING_POINTS:
            return (x, y)

        existing_points = get_all_existing_points()

        if not existing_points:
            return (x, y)

        click_pt = np.array([x, y], dtype=float)

        best_point = None
        best_dist = float("inf")

        for pt in existing_points:
            pt_arr = np.array(pt, dtype=float)
            dist = np.linalg.norm(click_pt - pt_arr)

            if dist < best_dist:
                best_dist = dist
                best_point = pt

        if best_dist <= SNAP_DISTANCE_PX:
            return best_point

        return (x, y)

    def redraw():
        ax.clear()
        ax.imshow(img_crop)
        ax.axis("off")

        # Already closed polygons
        for i, poly in enumerate(polygons):
            xs = [p[0] for p in poly] + [poly[0][0]]
            ys = [p[1] for p in poly] + [poly[0][1]]

            facecolor, edgecolor, alpha = get_polygon_facecolor(i, alpha=0.35)

            ax.fill(
                xs,
                ys,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=1.5,
                alpha=alpha
            )
            ax.plot(xs, ys, linewidth=1.2, color=edgecolor)
            ax.scatter(xs[:-1], ys[:-1], s=18, color=edgecolor)

            cx = np.mean([p[0] for p in poly])
            cy = np.mean([p[1] for p in poly])

            ax.text(
                cx,
                cy,
                f"P{i}",
                fontsize=9,
                ha="center",
                va="center",
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.7,
                    pad=1.5
                )
            )

        # Current polygon
        if current_poly:
            xs = [p[0] for p in current_poly]
            ys = [p[1] for p in current_poly]
            ax.plot(xs, ys, "r-o", linewidth=1.2, markersize=4)

        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return

        x, y = event.xdata, event.ydata

        if event.button == 1:
            snapped_point = find_nearest_existing_point(x, y)
            current_poly.append(snapped_point)
            redraw()

        elif event.button == 3:
            if len(current_poly) >= 3:
                polygons.append(current_poly.copy())
                current_poly.clear()
                redraw()

    def on_key(event):
        if event.key == "backspace":
            if current_poly:
                current_poly.pop()
                redraw()

        elif event.key == "enter":
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show()

    return polygons


# ============================================================
# SCALING
# ============================================================

def polygon_area(poly):
    area = 0.0
    n = len(poly)

    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        area += x1 * y2 - x2 * y1

    return abs(area) / 2.0


def select_reference_line(img_crop):
    print("\nSelect reference line with TWO clicks.")

    pts = []

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img_crop)
    ax.set_title("Click two points on a reference line")
    ax.axis("off")

    def onclick(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return

        pts.append((event.xdata, event.ydata))
        ax.plot(event.xdata, event.ydata, "ro", markersize=5)

        if len(pts) == 2:
            ax.plot(
                [pts[0][0], pts[1][0]],
                [pts[0][1], pts[1][1]],
                "r-",
                linewidth=1.2
            )
            fig.canvas.draw_idle()
            plt.pause(0.3)
            plt.close(fig)
        else:
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()

    if len(pts) != 2:
        raise ValueError("Reference line was not selected correctly.")

    return pts


def scale_polygons(polygons_px, img_crop, render_dpi=250):
    print("\nChoose scaling method:")
    print("1 = Known scale, e.g. 1:100")
    print("2 = Reference line")
    print("3 = Total real area of all polygons")

    choice = input("Enter 1, 2, or 3: ").strip()

    if choice == "1":
        scale_den = float(input("Enter scale denominator, e.g. 100 for 1:100: "))
        px_to_mm_paper = 25.4 / render_dpi
        px_to_m = (px_to_mm_paper * scale_den) / 1000.0

    elif choice == "2":
        pts = select_reference_line(img_crop)

        p1 = np.array(pts[0], dtype=float)
        p2 = np.array(pts[1], dtype=float)

        d_px = np.linalg.norm(p2 - p1)

        if d_px <= 0:
            raise ValueError("Invalid reference line length.")

        real_len_m = float(input("Enter REAL reference length in meters: "))
        px_to_m = real_len_m / d_px

    elif choice == "3":
        shapely_polys = [Polygon(poly) for poly in polygons_px]
        union_poly = unary_union(shapely_polys)
        area_px = union_poly.area

        px_to_mm_paper = 25.4 / render_dpi
        px_to_m_initial = px_to_mm_paper / 1000.0

        area_drawing_m2 = area_px * (px_to_m_initial ** 2)

        print(f"Pixel union area = {area_px:.2f} px²")
        print(f"Initial paper-based drawing area estimate = {area_drawing_m2:.4f} m²")

        real_area_total = float(input("Enter TOTAL REAL AREA of all polygons combined in m²: "))

        if real_area_total <= 0:
            raise ValueError("Real area must be positive.")

        scale_factor = math.sqrt(real_area_total / area_drawing_m2)
        px_to_m = px_to_m_initial * scale_factor

        print(f"Estimated global scale factor: 1:{scale_factor:.2f}")

    else:
        raise ValueError("Invalid scaling choice.")

    polygons_m = []

    for poly in polygons_px:
        polygons_m.append([(x * px_to_m, y * px_to_m) for (x, y) in poly])

    return polygons_m


# ============================================================
# CLASSIFICATION
# ============================================================

def classify_polygons(polygons_m, core_polygon_indices):
    records = []

    core_geoms = []
    dwelling_counter = 1

    for idx, poly in enumerate(polygons_m):
        geom = Polygon(poly)

        if not geom.is_valid:
            geom = geom.buffer(0)

        if idx in core_polygon_indices:
            core_geoms.append(geom)
        else:
            records.append({
                "geometry": geom,
                "label": "dwelling",
                "name": f"Dwelling {dwelling_counter}"
            })
            dwelling_counter += 1

    if core_geoms:
        merged_core = unary_union(core_geoms)

        records.append({
            "geometry": merged_core,
            "label": "core",
            "name": "Core"
        })

    return records


# ============================================================
# PLOTTING HELPERS
# ============================================================

def iter_geoms(geom):
    if geom is None:
        return

    if isinstance(geom, Polygon):
        yield geom

    elif isinstance(geom, MultiPolygon):
        for g in geom.geoms:
            yield g


def add_polygon_patch(ax, geom, facecolor, edgecolor, alpha=1.0, linewidth=0.8):
    for g in iter_geoms(geom):
        x, y = g.exterior.xy
        coords = np.column_stack([x, y])

        patch = MplPolygon(
            coords,
            closed=True,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=alpha
        )

        ax.add_patch(patch)


def set_geometry_limits(ax, records, invert_y=True, pad_ratio=0.08):
    xs = []
    ys = []

    for rec in records:
        for g in iter_geoms(rec["geometry"]):
            x, y = g.exterior.xy
            xs.extend(x)
            ys.extend(y)

    if not xs or not ys:
        return

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    dx = xmax - xmin
    dy = ymax - ymin

    padx = dx * pad_ratio if dx > 0 else 1.0
    pady = dy * pad_ratio if dy > 0 else 1.0

    ax.set_xlim(xmin - padx, xmax + padx)

    if invert_y:
        ax.set_ylim(ymax + pady, ymin - pady)
    else:
        ax.set_ylim(ymin - pady, ymax + pady)

    ax.set_aspect("equal", adjustable="box")


def style_axis(ax, show_box=True):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")

    if show_box:
        ax.set_frame_on(True)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_edgecolor(LEGEND_BOX_EDGE_COLOR)

    else:
        ax.set_frame_on(False)

        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.patch.set_edgecolor("none")
        ax.patch.set_linewidth(0)


def add_panel_label(ax, label):
    ax.text(
        0.5,
        -0.10,
        label,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10
    )


def add_area_label(ax, geom):
    area_m2 = geom.area

    if area_m2 < MIN_AREA_LABEL_M2:
        return

    pt = geom.representative_point()
    cx, cy = pt.x, pt.y

    label = f"A = {area_m2:.{AREA_LABEL_DECIMALS}f} m²"

    ax.text(
        cx,
        cy,
        label,
        ha="center",
        va="center",
        fontsize=7,
        bbox=dict(
            facecolor="white",
            edgecolor=POLYGON_EDGE_COLOR,
            boxstyle="square,pad=0.30",
            alpha=0.95,
            linewidth=0.8
        )
    )


# ============================================================
# PANEL PLOTS
# ============================================================

def plot_raw_image(ax, img_crop, show_box=True):
    style_axis(ax, show_box=show_box)
    ax.imshow(img_crop)


def plot_digitised_overlay(ax, img_crop, polygons_px, show_box=True):
    """
    Panel (b): original PDF image with translucent digitised polygons.
    """
    style_axis(ax, show_box=show_box)
    ax.imshow(img_crop)

    for i, poly in enumerate(polygons_px):
        xs = [p[0] for p in poly] + [poly[0][0]]
        ys = [p[1] for p in poly] + [poly[0][1]]

        facecolor, edgecolor, alpha = get_polygon_facecolor(i, alpha=0.35)

        ax.fill(
            xs,
            ys,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=1.0,
            alpha=alpha
        )
        ax.plot(xs, ys, color=edgecolor, linewidth=1.0)

        cx = np.mean([p[0] for p in poly])
        cy = np.mean([p[1] for p in poly])

        ax.text(
            cx,
            cy,
            f"P{i}",
            ha="center",
            va="center",
            fontsize=8,
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.7,
                pad=1.0
            )
        )


def plot_scaled_polygons(ax, polygons_m, show_box=False, show_area=True):
    """
    Panel (c): scaled polygons as outline-only geometry,
    with optional area labels in m².
    """
    style_axis(ax, show_box=show_box)

    temp_records = []

    for i, poly in enumerate(polygons_m):
        geom = Polygon(poly)

        if not geom.is_valid:
            geom = geom.buffer(0)

        temp_records.append({
            "geometry": geom,
            "label": "core" if i in CORE_POLYGON_INDICES else "dwelling"
        })

        if i in CORE_POLYGON_INDICES:
            edge = CORE_EDGE_COLOR
        else:
            edge = POLYGON_EDGE_COLOR

        add_polygon_patch(
            ax,
            geom,
            facecolor="none",
            edgecolor=edge,
            alpha=1.0,
            linewidth=1.1
        )

        if show_area:
            add_area_label(ax, geom)

    set_geometry_limits(ax, temp_records, invert_y=True)


def plot_classified(ax, records, add_legend=True, show_box=False):
    """
    Panel (d): classified dwelling/core geometry.
    """
    style_axis(ax, show_box=show_box)

    dwelling_i = 0
    legend_handles = []

    for rec in records:
        geom = rec["geometry"]

        if rec["label"] == "core":
            face = CORE_COLOR
            edge = CORE_EDGE_COLOR

        else:
            face = DWELLING_COLORS[dwelling_i % len(DWELLING_COLORS)]
            edge = POLYGON_EDGE_COLOR

            if add_legend:
                legend_handles.append(
                    Patch(
                        facecolor=face,
                        edgecolor="none",
                        label=f"Dwelling {dwelling_i + 1}"
                    )
                )

            dwelling_i += 1

        add_polygon_patch(
            ax,
            geom,
            facecolor=face,
            edgecolor=edge,
            alpha=1.0,
            linewidth=0.8
        )

    if add_legend and any(rec["label"] == "core" for rec in records):
        legend_handles.append(
            Patch(
                facecolor=CORE_COLOR,
                edgecolor="none",
                label="Stairwell"
            )
        )

    set_geometry_limits(ax, records, invert_y=True)

    if add_legend and legend_handles:
        leg = ax.legend(
            handles=legend_handles,
            loc="upper right",
            bbox_to_anchor=(1.18, 1.0),
            frameon=True,
            fancybox=False,
            framealpha=1.0,
            borderpad=0.4,
            handlelength=1.2,
            handletextpad=0.6
        )

        leg.get_frame().set_edgecolor(LEGEND_BOX_EDGE_COLOR)
        leg.get_frame().set_linewidth(0.8)
        leg.get_frame().set_facecolor("white")


# ============================================================
# FINAL WORKFLOW FIGURE
# ============================================================

def create_workflow_figure(img_crop, polygons_px, polygons_m, classified_records, output_path):
    """
    Panels:
      (a) Original PDF floor plan
      (b) Digitised polygons
      (c) Scaled polygons with area labels
      (d) Classified dwellings and core
    """
    fig = plt.figure(figsize=(6.14, 3.5))
    gs = fig.add_gridspec(2, 2, wspace=0.12, hspace=0.28)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    plot_raw_image(ax_a, img_crop, show_box=True)
    plot_digitised_overlay(ax_b, img_crop, polygons_px, show_box=True)

    plot_scaled_polygons(
        ax_c,
        polygons_m,
        show_box=False,
        show_area=SHOW_AREA_LABELS
    )

    plot_classified(ax_d, classified_records, add_legend=True, show_box=False)

    add_panel_label(ax_a, "(a)")
    add_panel_label(ax_b, "(b)")
    add_panel_label(ax_c, "(c)")
    add_panel_label(ax_d, "(d)")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    png_path = output_path.with_suffix(".png")

    plt.savefig(
        output_path,
        format="pdf",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.03
    )

    # plt.savefig(
    #     png_path,
    #     format="png",
    #     dpi=300,
    #     bbox_inches="tight",
    #     pad_inches=0.03
    # )

    plt.show()
    plt.close()

    print(f"Saved PDF figure: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    pdf_path = Path(PDF_PATH)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print("Rendering PDF...")

    img_full = render_pdf_first_page(pdf_path, dpi=RENDER_DPI)

    img_crop, _ = crop_white_margin(
        img_full,
        threshold=CROP_THRESHOLD,
        border=CROP_BORDER
    )

    polygons_px = digitise_polygons_interactively(img_crop)

    if not polygons_px:
        raise ValueError("No polygons were digitised.")

    print(f"Digitised {len(polygons_px)} polygon(s).")

    print("Scaling polygons...")

    polygons_m = scale_polygons(
        polygons_px=polygons_px,
        img_crop=img_crop,
        render_dpi=RENDER_DPI
    )

    print("Classifying dwellings and core...")

    classified_records = classify_polygons(
        polygons_m=polygons_m,
        core_polygon_indices=CORE_POLYGON_INDICES
    )

    print("Creating workflow figure...")

    create_workflow_figure(
        img_crop=img_crop,
        polygons_px=polygons_px,
        polygons_m=polygons_m,
        classified_records=classified_records,
        output_path=OUTPUT_FIGURE
    )


if __name__ == "__main__":
    main()