import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def draw_workflow_box(ax, x, y, width, height, text, fontsize=11):
    box = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.03,rounding_size=0.04",
        linewidth=0.8,
        edgecolor="black",
        facecolor="white",
    )
    ax.add_patch(box)

    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        linespacing=1.15,
    )


def draw_arrow(ax, x, y_start, y_end):
    arrow = FancyArrowPatch(
        (x, y_start),
        (x, y_end),
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=0.8,
        color="black",
    )
    ax.add_patch(arrow)


def create_sensitivity_workflow_figure(output_dir, output_name):
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    plt.rcParams["mathtext.fontset"] = "stix"

    fig, ax = plt.subplots(figsize=(7.2, 6.2))

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    steps = [
        {
            "y": 9.1,
            "width": 8.9,
            "height": 0.68,
            "text": "Determine variations or probability distributions\nof input factors",
        },
        {
            "y": 7.55,
            "width": 7.9,
            "height": 0.68,
            "text": "Create building energy models\nbased on input variations",
        },
        {
            "y": 6.0,
            "width": 3.6,
            "height": 0.60,
            "text": "Run energy models",
        },
        {
            "y": 4.45,
            "width": 4.6,
            "height": 0.60,
            "text": "Collect simulation results",
        },
        {
            "y": 2.9,
            "width": 4.6,
            "height": 0.60,
            "text": "Run sensitivity analysis",
        },
        {
            "y": 1.35,
            "width": 6.4,
            "height": 0.62,
            "text": "Present sensitivity analysis results",
        },
    ]

    x_center = 5.0

    for step in steps:
        draw_workflow_box(
            ax=ax,
            x=x_center,
            y=step["y"],
            width=step["width"],
            height=step["height"],
            text=step["text"],
            fontsize=11,
        )

    for i in range(len(steps) - 1):
        y_start = steps[i]["y"] - steps[i]["height"] / 2
        y_end = steps[i + 1]["y"] + steps[i + 1]["height"] / 2
        draw_arrow(ax, x_center, y_start, y_end)

    os.makedirs(output_dir, exist_ok=True)

    # png_path = os.path.join(output_dir, f"{output_name}.png")
    pdf_path = os.path.join(output_dir, f"{output_name}.pdf")
    # svg_path = os.path.join(output_dir, f"{output_name}.svg")

    # plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    # plt.savefig(svg_path, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    # print(f"Saved:\n{png_path}\n{pdf_path}\n{svg_path}")


def main():
    output_dir = r"C:\WF\Thomas Sharon\Master_Thesis_Report\figures"
    output_name = "sa_workflow_tian_adapted"

    create_sensitivity_workflow_figure(
        output_dir=output_dir,
        output_name=output_name,
    )


if __name__ == "__main__":
    main()