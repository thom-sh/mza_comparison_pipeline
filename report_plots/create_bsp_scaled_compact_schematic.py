"""
Create a compact schematic figure for BSP partitioning and a simple dwelling-core layout.

Updated BSP logic:
    Stage 1: A
    Stage 2: A -> B + C
    Stage 3: B -> D + E   (horizontal split inside B)
    Stage 4: D -> F + G   (vertical split inside D)

The script saves one PDF and displays the figure after saving.

Edit only the CONFIGURATION section if needed.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, FancyArrowPatch, Rectangle
from matplotlib.patches import Polygon as MplPolygon


# =========================
# CONFIGURATION
# =========================
OUTPUT_DIR = r"C:\WF\Thomas Sharon\Master_Thesis_Report\figures"
OUTPUT_NAME = "bsp_partitioning_updated_logic_horizontal_B_vertical_D.pdf"

FIGSIZE = (7.2, 2.4)
DPI = 300
SHOW_AFTER_SAVE = True

# Simple representative building dimensions for panel (b)
BUILDING_WIDTH = 12.0
BUILDING_HEIGHT = 6.0

# Core dimensions in the final plan
CORE_WIDTH = 2.0
CORE_DEPTH = 3.0

CORE_X0 = (BUILDING_WIDTH - CORE_WIDTH) / 2.0
CORE_X1 = CORE_X0 + CORE_WIDTH
CORE_Y0 = 0.0
CORE_Y1 = CORE_DEPTH

# Representative BSP leaf size for panel (b)
LEAF_SIZE = 1.0

# Visual scale for panel (b)
PLAN_SCALE = 0.5


# =========================
# THESIS COLOR PALETTE
# =========================
LEGEND_EDGE_COLOR = "#bdc1c5"

POLYGON_EDGE_COLOR = "#777d84"
CORE_EDGE_COLOR = "#777d84"
LEAF_EDGE_COLOR = "#bdc1c5"

TREE_EDGE_COLOR = "#777d84"
TREE_FILL_COLOR = "white"

DWELLING_COLOR = "#F0F0F0"
CORE_COLOR = "#BFC3C7"
CORE_ALPHA = 0.55


# =========================
# MATPLOTLIB STYLE
# =========================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": [
        "Times New Roman",
        "STIX Two Text",
        "STIXGeneral",
        "DejaVu Serif",
    ],
    "mathtext.fontset": "stix",

    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,

    "axes.titleweight": "normal",
    "axes.labelweight": "normal",

    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


# =========================
# GENERAL HELPERS
# =========================
def setup_axis(ax):
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)


def add_panel_label(ax, label, y=-0.08):
    ax.text(
        0.5,
        y,
        label,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
    )


# =========================
# TREE DRAWING HELPERS
# =========================
def draw_tree_node(ax, x, y, label):
    circle = Circle(
        (x, y),
        0.14,
        facecolor=TREE_FILL_COLOR,
        edgecolor=TREE_EDGE_COLOR,
        linewidth=0.85,
        zorder=5,
    )
    ax.add_patch(circle)
    ax.text(
        x,
        y,
        label,
        ha="center",
        va="center",
        fontsize=7.5,
        zorder=6,
    )


def draw_node_role_label(ax, x, y, text, ha="left"):
    ax.text(
        x,
        y,
        text,
        ha=ha,
        va="center",
        fontsize=6,
        fontstyle="italic",
        color="#4c5055",
        zorder=7,
    )

def draw_tree_edge(ax, p1, p2):
    ax.plot(
        [p1[0], p2[0]],
        [p1[1], p2[1]],
        color=TREE_EDGE_COLOR,
        linewidth=0.75,
        zorder=4,
    )


def draw_small_tree(ax, cx, y0, stage):
    """
    Draw the BSP tree below one rectangle stage.
    """
    A = (cx, y0 + 0.72)

    if stage == 1:
        draw_tree_node(ax, *A, "A")
        draw_node_role_label(ax, A[0] + 0.22, A[1] + 0.02, "Root\nnode")
        return

    B = (cx - 0.42, y0 + 0.34)
    C = (cx + 0.42, y0 + 0.34)

    draw_tree_edge(ax, A, B)
    draw_tree_edge(ax, A, C)

    draw_tree_node(ax, *A, "A")
    draw_tree_node(ax, *B, "B")
    draw_tree_node(ax, *C, "C")

    if stage == 2:
        draw_node_role_label(ax, A[0] + 0.22, A[1] + 0.02, "Parent\nnode")
        draw_node_role_label(ax, B[0] - 0.18, B[1] - 0.28, "Child\nnode", ha="right")
        draw_node_role_label(ax, C[0] + 0.18, C[1] - 0.28, "Child\nnode")
        return

    D = (cx - 0.62, y0 - 0.04)
    E = (cx - 0.22, y0 - 0.04)

    draw_tree_edge(ax, B, D)
    draw_tree_edge(ax, B, E)

    draw_tree_node(ax, *D, "D")
    draw_tree_node(ax, *E, "E")

    if stage == 3:
        return

    F = (cx - 0.78, y0 - 0.42)
    G = (cx - 0.46, y0 - 0.42)

    draw_tree_edge(ax, D, F)
    draw_tree_edge(ax, D, G)

    draw_tree_node(ax, *F, "F")
    draw_tree_node(ax, *G, "G")

    if stage == 4:
        draw_node_role_label(ax, C[0] + 0.20, C[1], "Leaf\nnode")
        draw_node_role_label(ax, F[0] - 0.18, F[1], "Leaf\nnode", ha="right")
        draw_node_role_label(ax, G[0] + 0.20, G[1], "Leaf\nnode")



# =========================
# PANEL (a): BSP PROCESS
# =========================
def draw_bsp_stage(ax, x0, y0, scale, stage):
    """
    Draw one rectangular BSP stage.

    Logic:
        Stage 1: A
        Stage 2: A -> B + C               (vertical split)
        Stage 3: B -> D + E              (horizontal split inside B)
        Stage 4: D -> F + G              (vertical split inside D)

    Final stage:
        left upper region  -> F | G
        left lower region  -> E
        right full region  -> C
    """
    width = 2.0 * scale
    height = 1.0 * scale

    # Outer rectangle
    ax.add_patch(
        Rectangle(
            (x0, y0),
            width,
            height,
            facecolor="#F4F4F4",
            edgecolor=POLYGON_EDGE_COLOR,
            linewidth=0.9,
            zorder=1,
        )
    )

    # Split locations
    x_bc = x0 + 1.0 * scale          # A -> B + C
    y_de = y0 + 0.5 * height         # B -> D + E (horizontal)
    x_fg = x0 + 0.5 * scale          # D -> F + G (vertical within top-left block)

    # Stage 2: A split into B and C
    if stage >= 2:
        ax.plot(
            [x_bc, x_bc],
            [y0, y0 + height],
            color=LEAF_EDGE_COLOR,
            linewidth=0.9,
            linestyle="--",
            zorder=2,
        )

    # Stage 3: B split horizontally into D and E
    if stage >= 3:
        ax.plot(
            [x0, x_bc],
            [y_de, y_de],
            color=LEAF_EDGE_COLOR,
            linewidth=0.9,
            linestyle="--",
            zorder=2,
        )

    # Stage 4: D split vertically into F and G
    if stage >= 4:
        ax.plot(
            [x_fg, x_fg],
            [y_de, y0 + height],
            color=LEAF_EDGE_COLOR,
            linewidth=0.9,
            linestyle="--",
            zorder=2,
        )

    # Labels
    label_kwargs = dict(
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        zorder=3,
    )

    if stage == 1:
        ax.text(x0 + width / 2, y0 + height / 2, "A", **label_kwargs)

    elif stage == 2:
        ax.text(x0 + 0.5 * scale, y0 + 0.5 * height, "B", **label_kwargs)
        ax.text(x0 + 1.5 * scale, y0 + 0.5 * height, "C", **label_kwargs)

    elif stage == 3:
        ax.text(x0 + 0.5 * scale, y0 + 0.75 * height, "D", **label_kwargs)
        ax.text(x0 + 0.5 * scale, y0 + 0.25 * height, "E", **label_kwargs)
        ax.text(x0 + 1.5 * scale, y0 + 0.5 * height, "C", **label_kwargs)

    elif stage == 4:
        ax.text(x0 + 0.25 * scale, y0 + 0.75 * height, "F", **label_kwargs)
        ax.text(x0 + 0.75 * scale, y0 + 0.75 * height, "G", **label_kwargs)
        ax.text(x0 + 0.5 * scale, y0 + 0.25 * height, "E", **label_kwargs)
        ax.text(x0 + 1.5 * scale, y0 + 0.5 * height, "C", **label_kwargs)


def draw_bsp_process(ax):
    ax.set_xlim(0, 10.0)
    ax.set_ylim(0, 3.45)
    ax.axis("off")
    ax.set_aspect("equal", adjustable="box")

    stage_x = [0.20, 2.60, 5.00, 7.40]
    rect_y = 2.20
    rect_scale = 0.95

    # Draw rectangles
    for i, stage in enumerate([1, 2, 3, 4]):
        draw_bsp_stage(
            ax=ax,
            x0=stage_x[i],
            y0=rect_y,
            scale=rect_scale,
            stage=stage,
        )

    # Draw corresponding trees below rectangle centres
    tree_y = 0.95
    rect_centres = [x + rect_scale * 1.0 for x in stage_x]

    draw_small_tree(ax, rect_centres[0], tree_y, stage=1)
    draw_small_tree(ax, rect_centres[1], tree_y, stage=2)
    draw_small_tree(ax, rect_centres[2], tree_y, stage=3)
    draw_small_tree(ax, rect_centres[3], tree_y, stage=4)

    # Direction arrow
    arrow = FancyArrowPatch(
        (0.20, 0.08),
        (9.55, 0.08),
        arrowstyle="-|>",
        mutation_scale=9,
        linewidth=0.9,
        color=TREE_EDGE_COLOR,
    )
    ax.add_patch(arrow)

    ax.text(
        0.20,
        0.24,
        "Partitioning process",
        ha="left",
        va="bottom",
        fontsize=8,
        fontstyle="italic",
    )


# =========================
# PANEL (b): SIMPLE PLAN
# =========================
def draw_simple_plan(ax):
    setup_axis(ax)

    s = PLAN_SCALE

    width = BUILDING_WIDTH * s
    height = BUILDING_HEIGHT * s

    core_x0 = CORE_X0 * s
    core_x1 = CORE_X1 * s
    core_y0 = CORE_Y0 * s
    core_y1 = CORE_Y1 * s

    leaf_size = LEAF_SIZE * s
    centre_x = (core_x0 + core_x1) / 2.0

    left_apartment = [
        (0.0, 0.0),
        (core_x0, 0.0),
        (core_x0, core_y1),
        (centre_x, core_y1),
        (centre_x, height),
        (0.0, height),
    ]

    right_apartment = [
        (core_x1, 0.0),
        (width, 0.0),
        (width, height),
        (centre_x, height),
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
        MplPolygon(
            left_apartment,
            closed=True,
            facecolor=DWELLING_COLOR,
            edgecolor=POLYGON_EDGE_COLOR,
            linewidth=0.9,
            zorder=1,
        )
    )

    ax.add_patch(
        MplPolygon(
            right_apartment,
            closed=True,
            facecolor=DWELLING_COLOR,
            edgecolor=POLYGON_EDGE_COLOR,
            linewidth=0.9,
            zorder=1,
        )
    )

    ax.add_patch(
        MplPolygon(
            core,
            closed=True,
            facecolor=CORE_COLOR,
            edgecolor=CORE_EDGE_COLOR,
            linewidth=0.9,
            alpha=CORE_ALPHA,
            zorder=2,
        )
    )

    # BSP leaf boundaries
    x = 0.0
    while x <= width + 1e-9:
        ax.plot(
            [x, x],
            [0.0, height],
            color=LEAF_EDGE_COLOR,
            linewidth=0.35,
            zorder=3,
        )
        x += leaf_size

    y = 0.0
    while y <= height + 1e-9:
        ax.plot(
            [0.0, width],
            [y, y],
            color=LEAF_EDGE_COLOR,
            linewidth=0.35,
            zorder=3,
        )
        y += leaf_size

    # Final zone boundaries
    for pts, edge_color in [
        (left_apartment, POLYGON_EDGE_COLOR),
        (right_apartment, POLYGON_EDGE_COLOR),
        (core, CORE_EDGE_COLOR),
    ]:
        closed = pts + [pts[0]]
        ax.plot(
            [p[0] for p in closed],
            [p[1] for p in closed],
            color=edge_color,
            linewidth=1.0,
            zorder=4,
        )

    outline = [
        (0.0, 0.0),
        (width, 0.0),
        (width, height),
        (0.0, height),
        (0.0, 0.0),
    ]
    ax.plot(
        [p[0] for p in outline],
        [p[1] for p in outline],
        color=POLYGON_EDGE_COLOR,
        linewidth=1.2,
        zorder=5,
    )

    ax.set_xlim(-0.3, width + 0.3)
    ax.set_ylim(-0.3, height + 0.3)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=LEAF_EDGE_COLOR,
            lw=0.7,
            label="BSP leaf boundary",
        ),
        Patch(
            facecolor=DWELLING_COLOR,
            edgecolor="none",
            label="Apartment",
        ),
        Patch(
            facecolor=CORE_COLOR,
            edgecolor="none",
            alpha=CORE_ALPHA,
            label="Core",
        ),
    ]

    # legend = ax.legend(
    #     handles=legend_handles,
    #     loc="lower center",
    #     bbox_to_anchor=(0.5, -0.37),
    #     ncol=3,
    #     frameon=True,
    #     framealpha=1.0,
    #     borderpad=0.35,
    #     handlelength=1.2,
    #     columnspacing=1.1,
    # )
    # legend.get_frame().set_edgecolor(LEGEND_EDGE_COLOR)
    # legend.get_frame().set_linewidth(0.8)


# =========================
# MAIN
# =========================
def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / OUTPUT_NAME

    fig, axes = plt.subplots(
        1,
        2,
        figsize=FIGSIZE,
        dpi=DPI,
        gridspec_kw={"width_ratios": [2.00, 0.60]},
    )

    draw_bsp_process(axes[0])
    axes[0].set_title("BSP partitioning process", pad=3)
    # add_panel_label(axes[0], "(a)", y=-0.03)

    draw_simple_plan(axes[1])
    axes[1].set_title("Leaf based apartment layout", pad=3)
    # add_panel_label(axes[1], "(b)", y=-0.34)

    plt.subplots_adjust(
        left=0.02,
        right=0.98,
        top=0.86,
        bottom=0.14,
        wspace=0.08,
    )

    fig.savefig(
        output_path,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.01,
    )
    print(f"Saved PDF: {output_path}")

    if SHOW_AFTER_SAVE:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()