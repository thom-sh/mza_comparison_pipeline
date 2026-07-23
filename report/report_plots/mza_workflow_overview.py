from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import fitz

from matplotlib.patches import (
    Rectangle,
    FancyBboxPatch,
    FancyArrowPatch,
    Circle,
    Polygon,
)


# ============================================================
# Style
# ============================================================

def setup_style():
    plt.rcParams.update({
        "font.family": "cmr10",
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
        "font.size": 15,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# ============================================================
# Configuration
# ============================================================

# PDF thumbnails used in the input-source cards.
# Replace these placeholder paths with the actual PDF files on your system.
CITYGML_THUMBNAIL_PDF = Path(
    r"C:\WF\Thomas Sharon\Master_Thesis_Report\for_plotting_reference\mza_workflow_logos\CityGML_Berlin.pdf"
)
OSM_THUMBNAIL_PDF = Path(
    r"C:\WF\Thomas Sharon\Master_Thesis_Report\for_plotting_reference\mza_workflow_logos\Picture1.pdf"
)
BUILDING_STOCK_THUMBNAIL_PDF = Path(
    r"C:\WF\Thomas Sharon\Master_Thesis_Report\for_plotting_reference\mza_workflow_logos\stock.pdf"
)
TABULA_LOGO_PDF = Path(
    r"C:\WF\Thomas Sharon\Master_Thesis_Report\for_plotting_reference\mza_workflow_logos\tabula.pdf"
)

LPG_LOGO_PDF = Path(
    r"C:\WF\Thomas Sharon\Master_Thesis_Report\for_plotting_reference\mza_workflow_logos\logo_1.pdf"
)

TEASER_LOGO_PDF = Path(
    r"C:\WF\Thomas Sharon\Master_Thesis_Report\for_plotting_reference\mza_workflow_logos\teaser.pdf"
)

MODELICA_LOGO_PDF = Path(
    r"C:\WF\Thomas Sharon\Master_Thesis_Report\for_plotting_reference\mza_workflow_logos\modelica.pdf"
)

OUTPUT_GRAPH_PDF = Path(
    r"C:\WF\Thomas Sharon\Master_Thesis_Report\for_plotting_reference\mza_workflow_logos\graph.pdf"
)


# ============================================================
# Basic drawing helpers
# ============================================================

def add_header(ax, x, y, w, text):
    ax.add_patch(
        Rectangle(
            (x, y),
            w,
            0.42,
            facecolor="#e9ecef",
            edgecolor="none",
            zorder=0,
        )
    )
    ax.text(
        x + w / 2,
        y + 0.21,
        text,
        ha="center",
        va="center",
        fontsize=16,
    )


def add_arrow(ax, x1, y1, x2, y2, lw=1.5):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=lw,
            color="0.45",
        )
    )



def data_extent_to_pdf_rect(fig, ax, page, extent):
    """
    Convert a Matplotlib data-coordinate extent to a PyMuPDF page rectangle.
    extent = (left, right, bottom, top) in ax data coordinates.
    """
    fig.canvas.draw()

    left, right, bottom, top = extent

    x0_px, y0_px = ax.transData.transform((left, bottom))
    x1_px, y1_px = ax.transData.transform((right, top))

    x0_pt = x0_px / fig.dpi * 72.0
    x1_pt = x1_px / fig.dpi * 72.0
    y0_pt = y0_px / fig.dpi * 72.0
    y1_pt = y1_px / fig.dpi * 72.0

    return fitz.Rect(
        x0_pt,
        page.rect.height - y1_pt,
        x1_pt,
        page.rect.height - y0_pt,
    )


def insert_pdf_thumbnails(fig, ax, base_pdf, final_pdf, thumbnail_placements):
    """
    Insert PDF thumbnails as vector PDF objects into a saved Matplotlib PDF.
    thumbnail_placements is a list of (pdf_path, extent) tuples.
    """
    doc = fitz.open(base_pdf)
    page = doc[0]

    for pdf_path, extent in thumbnail_placements:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"Missing thumbnail PDF: {pdf_path}")

        rect = data_extent_to_pdf_rect(fig, ax, page, extent)
        thumb_doc = fitz.open(pdf_path)
        page.show_pdf_page(
            rect,
            thumb_doc,
            0,
            overlay=True,
            keep_proportion=True,
        )
        thumb_doc.close()

    doc.save(final_pdf, garbage=4, deflate=True)
    doc.close()



# ============================================================
# Stage 1: Data aggregation and preprocessing
# ============================================================

def draw_input_sources(ax, x, y, scale=1.0, thumbnail_placements=None):
    """
    Draw three enlarged input-source cards.
    PDF thumbnails are inserted later into the saved PDF as vector PDF objects.
    """

    card_w = 2.35 * scale
    card_h = 0.58 * scale
    gap = 0.22 * scale

    edge_color = "black"
    fill_color = "#f7f7f7"

    sources = [
        ("CityGML / LoD2", "geometry", CITYGML_THUMBNAIL_PDF),
        ("OpenStreetMap", "address / access", OSM_THUMBNAIL_PDF),
        ("Building stock data", "apartment size, period", BUILDING_STOCK_THUMBNAIL_PDF),
    ]

    for i, (title, subtitle, thumb_pdf) in enumerate(sources):
        yy = y + (2 - i) * (card_h + gap)

        ax.add_patch(
            FancyBboxPatch(
                (x, yy),
                card_w,
                card_h,
                boxstyle="round,pad=0.035,rounding_size=0.045",
                facecolor=fill_color,
                edgecolor=edge_color,
                linewidth=0.9,
                zorder=2,
            )
        )

        thumb_extent = (
            x + 0.01 * scale,
            x + 0.6 * scale,
            yy + 0.04 * scale,
            yy + card_h - 0.04 * scale,
        )

        if thumbnail_placements is not None:
            thumbnail_placements.append((thumb_pdf, thumb_extent))

        ax.text(
            x + 0.76 * scale,
            yy + 0.37 * scale,
            title,
            ha="left",
            va="center",
            fontsize=15,
            zorder=4,
        )

        ax.text(
            x + 0.76 * scale,
            yy + 0.17 * scale,
            subtitle,
            ha="left",
            va="center",
            fontsize=12,
            color="0.35",
            zorder=4,
        )

    ax.text(
        x + card_w / 2,
        y - 0.16 * scale,
        "Urban input data",
        ha="center",
        va="top",
        fontsize=15,
    )


def draw_footprint(ax, x, y, scale=1.0):
    w = 1.55 * scale
    h = 1.05 * scale

    ax.add_patch(
        Rectangle(
            (x, y),
            w,
            h,
            facecolor="none",
            edgecolor="black",
            linewidth=1.0,
        )
    )

    ax.text(
        x + w / 2,
        y - 0.18 * scale,
        "Building footprint",
        ha="center",
        va="top",
        fontsize=15,
    )


# Backward-compatible name if older calls are used.
draw_footprint_with_core = draw_footprint


# ============================================================
# Stage 2: Automatic zoning
# ============================================================

def draw_bsp_tree_final_logic(ax, x, y, scale=1.0):
    """
    Final BSP tree logic:
        A -> B + C
        B -> D + E
        D -> F + G
    """
    nodes = {
        "A": (x + 0.95 * scale, y + 1.75 * scale),
        "B": (x + 0.58 * scale, y + 1.25 * scale),
        "C": (x + 1.32 * scale, y + 1.25 * scale),
        "D": (x + 0.35 * scale, y + 0.72 * scale),
        "E": (x + 0.82 * scale, y + 0.72 * scale),
        "F": (x + 0.18 * scale, y + 0.18 * scale),
        "G": (x + 0.52 * scale, y + 0.18 * scale),
    }

    edges = [
        ("A", "B"),
        ("A", "C"),
        ("B", "D"),
        ("B", "E"),
        ("D", "F"),
        ("D", "G"),
    ]

    for a, b in edges:
        ax.plot(
            [nodes[a][0], nodes[b][0]],
            [nodes[a][1], nodes[b][1]],
            color="0.35",
            linewidth=1.0,
            zorder=1,
        )

    for label, (xx, yy) in nodes.items():
        ax.add_patch(
            Circle(
                (xx, yy),
                0.14 * scale,
                facecolor="white",
                edgecolor="#777d84",
                linewidth=0.9,
                zorder=2,
            )
        )
        ax.text(xx, yy, label, ha="center", va="center", fontsize=12, zorder=3)

    ax.text(
        x + 0.78 * scale,
        y + 1.98 * scale,
        "BSP tree",
        ha="center",
        va="bottom",
        fontsize=15,
    )


def draw_leaf_based_apartment_core_layout(ax, x, y, scale=1.0):
    """
    Leaf-based apartment/core layout based on the previous BSP figure logic.
    Two dwelling regions are arranged around a central shared core.
    """
    unit = 0.19 * scale

    building_width = 12.0 * unit
    building_height = 6.0 * unit

    core_width = 2.0 * unit
    core_depth = 3.0 * unit

    core_x0 = x + (building_width - core_width) / 2.0
    core_x1 = core_x0 + core_width
    core_y0 = y
    core_y1 = y + core_depth

    centre_x = (core_x0 + core_x1) / 2.0

    dwelling_color = "#F0F0F0"
    core_color = "#BFC3C7"
    edge_color = "#777d84"
    leaf_color = "#bdc1c5"

    left_apartment = [
        (x, y),
        (core_x0, y),
        (core_x0, core_y1),
        (centre_x, core_y1),
        (centre_x, y + building_height),
        (x, y + building_height),
    ]

    right_apartment = [
        (core_x1, y),
        (x + building_width, y),
        (x + building_width, y + building_height),
        (centre_x, y + building_height),
        (centre_x, core_y1),
        (core_x1, core_y1),
    ]

    core = [
        (core_x0, core_y0),
        (core_x1, core_y0),
        (core_x1, core_y1),
        (core_x0, core_y1),
    ]

    ax.add_patch(
        Polygon(
            left_apartment,
            closed=True,
            facecolor=dwelling_color,
            edgecolor=edge_color,
            linewidth=0.9,
            zorder=1,
        )
    )

    ax.add_patch(
        Polygon(
            right_apartment,
            closed=True,
            facecolor=dwelling_color,
            edgecolor=edge_color,
            linewidth=0.9,
            zorder=1,
        )
    )

    ax.add_patch(
        Polygon(
            core,
            closed=True,
            facecolor=core_color,
            edgecolor=edge_color,
            linewidth=0.9,
            alpha=0.65,
            zorder=2,
        )
    )

    xx = x
    while xx <= x + building_width + 1e-9:
        ax.plot(
            [xx, xx],
            [y, y + building_height],
            color=leaf_color,
            linewidth=0.35,
            zorder=3,
        )
        xx += unit

    yy = y
    while yy <= y + building_height + 1e-9:
        ax.plot(
            [x, x + building_width],
            [yy, yy],
            color=leaf_color,
            linewidth=0.35,
            zorder=3,
        )
        yy += unit

    for pts in [left_apartment, right_apartment, core]:
        closed = pts + [pts[0]]
        ax.plot(
            [p[0] for p in closed],
            [p[1] for p in closed],
            color=edge_color,
            linewidth=1.0,
            zorder=4,
        )

    ax.add_patch(
        Rectangle(
            (x, y),
            building_width,
            building_height,
            facecolor="none",
            edgecolor=edge_color,
            linewidth=1.2,
            zorder=5,
        )
    )

    ax.text(
        x + building_width / 1.5,
        y - 0.18 * scale,
        "Dwelling and stairwell zone layout",
        ha="center",
        va="top",
        fontsize=15,
    )


def draw_3d_building(ax, x, y, scale=1.0):
    """
    Three-floor multizone building.
    Lower two floors are shown as closed facades.
    The third floor is shown as an open wireframe floor with the final
    dwelling/core partition layout.
    """

    front_w = 1.95 * scale
    front_h = 1.55 * scale
    dx = 0.52 * scale
    dy = 0.34 * scale

    edge_color = "black"
    front_color = "#f2f2f2"
    side_color = "#d9d9d9"
    dwelling_color = "#f2f2f2"
    slab_color = "#f7f7f7"
    window_color = "#7f7f7f"

    top_y = y + front_h

    ax.add_patch(
        Rectangle(
            (x, y),
            front_w,
            front_h,
            facecolor=front_color,
            edgecolor=edge_color,
            linewidth=1.0,
            zorder=2,
        )
    )

    ax.add_patch(
        Polygon(
            [
                (x + front_w, y),
                (x + front_w + dx, y + dy),
                (x + front_w + dx, y + front_h + dy),
                (x + front_w, y + front_h),
            ],
            closed=True,
            facecolor=side_color,
            edgecolor=edge_color,
            linewidth=1.0,
            zorder=1,
        )
    )

    yy = y + front_h / 2.0
    ax.plot([x, x + front_w], [yy, yy], color="0.55", linewidth=0.8, zorder=4)
    ax.plot([x + front_w, x + front_w + dx], [yy, yy + dy], color="0.55", linewidth=0.8, zorder=4)

    core_front_x = x + 0.83 * scale
    core_front_w = 0.26 * scale

    # Core strip is transparent; only the core boundary is retained.
    ax.add_patch(
        Rectangle(
            (core_front_x, y),
            core_front_w,
            front_h,
            facecolor="none",
            edgecolor=edge_color,
            linewidth=0.8,
            zorder=5,
        )
    )

    ax.add_patch(
        Rectangle(
            (core_front_x + 0.04 * scale, y),
            0.18 * scale,
            0.32 * scale,
            facecolor="#7f7f7f",
            edgecolor=edge_color,
            linewidth=0.6,
            zorder=6,
        )
    )

    window_w = 0.22 * scale
    window_h = 0.18 * scale

    left_cols = [x + 0.18 * scale, x + 0.48 * scale]
    right_cols = [x + 1.22 * scale, x + 1.52 * scale]
    rows = [y + 0.28 * scale, y + 1.02 * scale]

    for yy in rows:
        for xx in left_cols + right_cols:
            ax.add_patch(
                Rectangle(
                    (xx, yy),
                    window_w,
                    window_h,
                    facecolor=window_color,
                    edgecolor="none",
                    zorder=6,
                )
            )

    def top_point(px, py):
        return (
            x + px * front_w + py * dx,
            top_y + py * dy,
        )

    def add_top_polygon(points, facecolor, zorder):
        mapped = [top_point(px, py) for px, py in points]
        ax.add_patch(
            Polygon(
                mapped,
                closed=True,
                facecolor=facecolor,
                edgecolor=edge_color,
                linewidth=0.9,
                zorder=zorder,
            )
        )

    slab = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    add_top_polygon(slab, slab_color, zorder=7)

    core_x0 = 0.42
    core_x1 = 0.58
    core_y0 = 0.00
    core_y1 = 0.50
    centre_x = 0.50

    left_dwelling = [
        (0.00, 0.00),
        (core_x0, 0.00),
        (core_x0, core_y1),
        (centre_x, core_y1),
        (centre_x, 1.00),
        (0.00, 1.00),
    ]

    right_dwelling = [
        (core_x1, 0.00),
        (1.00, 0.00),
        (1.00, 1.00),
        (centre_x, 1.00),
        (centre_x, core_y1),
        (core_x1, core_y1),
    ]

    core = [
        (core_x0, core_y0),
        (core_x1, core_y0),
        (core_x1, core_y1),
        (core_x0, core_y1),
    ]

    add_top_polygon(left_dwelling, dwelling_color, zorder=8)
    add_top_polygon(right_dwelling, dwelling_color, zorder=8)
    add_top_polygon(core, "none", zorder=9)

    third_floor_h = 0.62 * scale

    def top_point_3d(px, py, pz=0.0):
        return (
            x + px * front_w + py * dx,
            top_y + py * dy + pz * third_floor_h,
        )

    def draw_3d_line(p1, p2, lw=0.9, color=edge_color, zorder=11):
        x1, y1 = top_point_3d(*p1)
        x2, y2 = top_point_3d(*p2)
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, zorder=zorder)

    wire_vertices = [
        (0.00, 0.00), (1.00, 0.00), (1.00, 1.00), (0.00, 1.00),
        (core_x0, core_y0), (core_x1, core_y0),
        (core_x0, core_y1), (core_x1, core_y1),
        (centre_x, core_y1), (centre_x, 1.00),
    ]

    for px, py in wire_vertices:
        draw_3d_line((px, py, 0.0), (px, py, 1.0), lw=0.85, color=edge_color, zorder=12)

    draw_3d_line((0.00, 0.00, 1.0), (1.00, 0.00, 1.0), lw=1.0, color=edge_color)
    draw_3d_line((1.00, 0.00, 1.0), (1.00, 1.00, 1.0), lw=1.0, color=edge_color)
    draw_3d_line((1.00, 1.00, 1.0), (0.00, 1.00, 1.0), lw=1.0, color=edge_color)
    draw_3d_line((0.00, 1.00, 1.0), (0.00, 0.00, 1.0), lw=1.0, color=edge_color)

    draw_3d_line((core_x0, core_y0, 1.0), (core_x0, core_y1, 1.0), lw=0.9, color=edge_color)
    draw_3d_line((core_x1, core_y0, 1.0), (core_x1, core_y1, 1.0), lw=0.9, color=edge_color)
    draw_3d_line((core_x0, core_y1, 1.0), (core_x1, core_y1, 1.0), lw=0.9, color=edge_color)
    draw_3d_line((centre_x, core_y1, 1.0), (centre_x, 1.00, 1.0), lw=0.9, color=edge_color)

    draw_3d_line((core_x0, core_y0, 0.0), (core_x0, core_y1, 0.0), lw=0.85, color=edge_color)
    draw_3d_line((core_x1, core_y0, 0.0), (core_x1, core_y1, 0.0), lw=0.85, color=edge_color)
    draw_3d_line((core_x0, core_y1, 0.0), (core_x1, core_y1, 0.0), lw=0.85, color=edge_color)
    draw_3d_line((centre_x, core_y1, 0.0), (centre_x, 1.00, 0.0), lw=0.85, color=edge_color)

    ax.text(
        x + front_w / 2 + 0.12 * scale,
        y - 0.18 * scale,
        "Multi-zone building model",
        ha="center",
        va="top",
        fontsize=15,
    )


# ============================================================
# Stage 3: Thermal enrichment and simulation
# ============================================================

def draw_teaser(ax, x, y, scale=1.0):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            1.25 * scale,
            0.58 * scale,
            boxstyle="round,pad=0.04,rounding_size=0.06",
            facecolor="#f4f4f4",
            edgecolor="0.35",
            linewidth=1.0,
        )
    )
    ax.text(
        x + 0.625 * scale,
        y + 0.36 * scale,
        "TEASER",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color="0.35",
    )
    ax.text(
        x + 0.625 * scale,
        y + 0.17 * scale,
        "VDI 6007",
        ha="center",
        va="center",
        fontsize=15,
        color="0.35",
    )


def draw_lpg_bars(ax, x, y, scale=1.0):
    values = np.array([0.35, 0.90, 0.55, 1.05, 0.70, 0.45])
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2"]

    for i, val in enumerate(values):
        ax.add_patch(
            Rectangle(
                (x + i * 0.13 * scale, y),
                0.08 * scale,
                val * 0.58 * scale,
                facecolor=colors[i],
                edgecolor="none",
            )
        )

    ax.text(
        x + 0.34 * scale,
        y - 0.12 * scale,
        "LPG\nprofiles",
        ha="center",
        va="top",
        fontsize=15,
    )


def draw_tabula(ax, x, y, scale=1.0):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            1.10 * scale,
            0.55 * scale,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            facecolor="#e8f0e8",
            edgecolor="#3a6b35",
            linewidth=1.2,
        )
    )
    ax.text(
        x + 0.55 * scale,
        y + 0.28 * scale,
        "TABULA",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
    )


def draw_simulation_box(ax, x, y, scale=1.0):
    w = 2.55 * scale
    h = 1.55 * scale

    ax.add_patch(
        Rectangle(
            (x, y),
            w,
            h,
            facecolor="#f9f9f9",
            edgecolor="black",
            linewidth=1.0,
        )
    )

    ax.add_patch(
        Rectangle(
            (x + 0.18 * scale, y + 0.85 * scale),
            0.78 * scale,
            0.48 * scale,
            facecolor="#eef3ff",
            edgecolor="#4c78a8",
            linewidth=0.9,
        )
    )
    ax.text(
        x + 0.57 * scale,
        y + 1.09 * scale,
        "AixLib /\nModelica",
        ha="center",
        va="center",
        fontsize=15,
    )

    xx = np.linspace(0, 1, 100)
    yy1 = 0.25 + 0.23 * np.sin(2 * np.pi * xx) + 0.08 * np.sin(6 * np.pi * xx)
    yy2 = yy1 + 0.07 * np.cos(3 * np.pi * xx)

    ax.plot(
        x + 1.10 * scale + xx * 1.15 * scale,
        y + 0.34 * scale + yy1 * scale,
        color="black",
        linewidth=1.1,
    )
    ax.plot(
        x + 1.10 * scale + xx * 1.15 * scale,
        y + 0.34 * scale + yy2 * scale,
        color="#b22222",
        linewidth=1.1,
    )

    ax.text(
        x + 1.68 * scale,
        y + 1.24 * scale,
        "simulation\noutputs",
        ha="center",
        va="center",
        fontsize=15,
    )

    ax.text(
        x + 1.68 * scale,
        y - 0.16 * scale,
        "heat load, temperatures,\ncomfort indicators",
        ha="center",
        va="top",
        fontsize=15,
    )


# ============================================================
# Bottom workflow line
# ============================================================

def draw_bottom_process_line(ax):
    y = 2.3

    ax.plot([0.65, 4.95], [y, y], color="0.55", linewidth=2.0)
    ax.plot([5.15, 9.75], [y, y], color="0.55", linewidth=2.0)
    ax.plot([9.95, 14.70], [y, y], color="0.55", linewidth=2.0)

    add_arrow(ax, 4.95, y, 5.05, y, lw=1.3)
    add_arrow(ax, 9.75, y, 9.85, y, lw=1.3)

    ax.text(2.80, y - 0.1, "Input data", ha="center", va="top", fontsize=15)
    ax.text(7.45, y - 0.1, "Python", ha="center", va="top", fontsize=15)
    ax.text(12.35, y - 0.1, "TEASER -- AixLib / Modelica", ha="center", va="top", fontsize=15)


# ============================================================
# Main figure
# ============================================================

def create_mza_workflow_figure(output_dir: Path, output_name: str):
    setup_style()

    # Compact one-row version for LaTeX insertion with width=\textwidth.
    # The figure ratio and x/y axis ratio are matched to avoid visual stretching.
    fig, ax = plt.subplots(figsize=(12, 3), dpi=300)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    thumbnail_placements = []

    ax.set_xlim(0.3, 15.)
    ax.set_ylim(2.0, 6.0)
    ax.axis("off")

    add_header(ax, 0.45, 5.45, 4.55, "Data aggregation and preprocessing")
    add_header(ax, 5.15, 5.45, 4.55, "Automatic zoning")
    add_header(ax, 9.95, 5.45, 4.75, "Thermal enrichment and simulation")

    for x_sep in [5.09, 9.85]:
        ax.plot(
            [x_sep, x_sep],
            [2.4, 5.25],
            color="0.84",
            linewidth=1.0,
            linestyle="--",
        )

    # --------------------------------------------------------
    # Stage 1: data aggregation and preprocessing
    # --------------------------------------------------------
    draw_input_sources(
        ax,
        0.55,
        2.90,
        scale=0.88,
        thumbnail_placements=thumbnail_placements,
    )
    add_arrow(ax, 2.80, 3.95, 3.45, 3.95)
    draw_footprint(ax, 3.60, 3.48, scale=0.90)

    # --------------------------------------------------------
    # Stage 2: automatic zoning
    # --------------------------------------------------------
    zoning_group_center_x = 6.15

    bsp_tree_scale = 0.60
    bsp_tree_x = zoning_group_center_x - 0.78 * bsp_tree_scale
    bsp_tree_y = 3.90

    draw_bsp_tree_final_logic(ax, bsp_tree_x, bsp_tree_y, scale=bsp_tree_scale)

    layout_scale = 0.75
    layout_width = 12.0 * 0.19 * layout_scale
    layout_x = zoning_group_center_x - layout_width / 2
    layout_y = 2.80

    draw_leaf_based_apartment_core_layout(ax, layout_x, layout_y, scale=layout_scale)

    building_x = 7.75
    building_y = 2.82
    building_scale = 0.78

    add_arrow(ax, 7.2, 3.95, 7.65, 3.95)
    draw_3d_building(ax, building_x, building_y, scale=building_scale)

    # --------------------------------------------------------
    # Stage 3: thermal enrichment and simulation
    # --------------------------------------------------------
    # ax.text(
    #     10.65,
    #     5.2,
    #     "Thermal enrichment",
    #     ha="center",
    #     va="center",
    #     fontsize=10,
    # )

    # TABULA and LPG PDF logos
    tabula_extent = (9.75, 11.20, 4.30, 4.95)
    thumbnail_placements.append((TABULA_LOGO_PDF, tabula_extent))

    lpg_extent = (9.75, 11.20, 3.20, 3.85)
    thumbnail_placements.append((LPG_LOGO_PDF, lpg_extent))

    ax.text(10.45, 3.90, "+", ha="center", va="center", fontsize=13, color="0.35")

    # Optional labels below the logos
    ax.text(
        10.5,
        4.20,
        "Typology / Envelope",
        ha="center",
        va="top",
        fontsize=9,
    )

    ax.text(
        10.5,
        3.15,
        "LPG usage profiles",
        ha="center",
        va="top",
        fontsize=9,
    )

    # Arrow from TABULA/LPG to TEASER
    add_arrow(ax, 11.1, 3.95, 11.45, 3.95)

    # TEASER parameterisation
    teaser_extent = (11.0, 12.80, 3.75, 4.55)
    thumbnail_placements.append((TEASER_LOGO_PDF, teaser_extent))

    ax.text(
        11.9,
        3.62,
        "TEASER\nparameterisation",
        ha="center",
        va="top",
        fontsize=9,
    )

    # Arrow from TEASER to simulation box
    add_arrow(ax, 12.45, 3.95, 12.80, 3.95)

    # Modelica / AixLib model thumbnail
    modelica_extent = (12.95, 14.71, 3.95, 5.55)
    thumbnail_placements.append((MODELICA_LOGO_PDF, modelica_extent))

    ax.text(
        13.78,
        4.08,
        "AixLib-based\nModelica model",
        ha="center",
        va="top",
        fontsize=9,
    )

    # Simulation output graph thumbnail
    graph_extent = (13.1, 14.56, 2.36, 3.75)
    thumbnail_placements.append((OUTPUT_GRAPH_PDF, graph_extent))

    ax.text(
        13.78,
        2.50,
        "simulation outputs",
        ha="center",
        va="top",
        fontsize=9,
    )

    draw_bottom_process_line(ax)

    output_dir.mkdir(parents=True, exist_ok=True)

    base_pdf_path = output_dir / f"{output_name}_base.pdf"
    pdf_path = output_dir / f"{output_name}.pdf"

    # Save the Matplotlib figure first. Do not use bbox_inches="tight" here,
    # because the PDF-thumbnail placement uses the original figure coordinates.
    fig.savefig(base_pdf_path)

    insert_pdf_thumbnails(
        fig=fig,
        ax=ax,
        base_pdf=base_pdf_path,
        final_pdf=pdf_path,
        thumbnail_placements=thumbnail_placements,
    )

    try:
        preview_final_pdf_in_jupyter(pdf_path)
    except Exception:
        pass

    base_pdf_path.unlink(missing_ok=True)
    plt.close(fig)

    print(f"Saved PDF: {pdf_path}")



def preview_final_pdf_in_jupyter(pdf_path, dpi=200):
    """
    Render the final PDF page to a PNG preview and display it in Jupyter.
    This preview is raster only; the saved PDF keeps the inserted PDF thumbnails.
    """
    from IPython.display import display, Image

    pdf_path = Path(pdf_path)
    preview_png = pdf_path.with_suffix(".preview.png")

    doc = fitz.open(pdf_path)
    page = doc[0]
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    pix.save(preview_png)
    doc.close()

    display(Image(filename=str(preview_png)))


# ============================================================
# Run
# ============================================================

def main():
    script_dir = Path(r"C:\WF\Thomas Sharon\Master_Thesis_Report\figures")

    output_dir = script_dir
    output_name = "mza_workflow_overview"

    create_mza_workflow_figure(
        output_dir=output_dir,
        output_name=output_name,
    )


if __name__ == "__main__":
    main()
