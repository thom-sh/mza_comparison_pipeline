import fitz  # PyMuPDF
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# Thesis visual standards
# ============================================================

LEGEND_BOX_EDGE_COLOR = "#bdc1c5"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIX Two Text", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",

    "font.size": 7,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 7,

    "axes.titleweight": "normal",
    "axes.labelweight": "normal",

    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def render_pdf_first_page(pdf_path, dpi=250):
    doc = fitz.open(pdf_path)
    page = doc[0]
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img


def load_plan(path, dpi=250):
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return render_pdf_first_page(path, dpi=dpi)
    else:
        return Image.open(path).convert("RGB")


def crop_white_margin(img, threshold=250, border=0):
    gray = img.convert("L")
    mask = gray.point(lambda p: 255 if p < threshold else 0)
    bbox = mask.getbbox()

    if bbox is None:
        return img

    left, top, right, bottom = bbox
    left = max(left - border, 0)
    top = max(top - border, 0)
    right = min(right + border, img.width)
    bottom = min(bottom + border, img.height)

    return img.crop((left, top, right, bottom))


def create_2x3_figure(plan_paths, output_pdf):
    images = []

    for path in plan_paths:
        img = load_plan(path)
        img = crop_white_margin(img, threshold=250, border=0)
        images.append(img)

    fig, axes = plt.subplots(2, 3, figsize=(8.27, 5.0))
    axes = axes.flatten()

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("white")

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_edgecolor(LEGEND_BOX_EDGE_COLOR)

    for i, img in enumerate(images):
        axes[i].imshow(img)
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    # Hide unused axes if fewer than 6 images are used
    for ax in axes[len(images):]:
        ax.set_visible(False)

    plt.subplots_adjust(
        left=0.02,
        right=0.98,
        top=0.98,
        bottom=0.02,
        wspace=0.08,
        hspace=0.10
    )

    plt.savefig(output_pdf, format="pdf", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.show()
    plt.close()

    print(f"Saved figure to: {output_pdf}")


def main():
    from pathlib import Path

    # If this script is inside:
    # rd_ground_truth_extraction/figures/
    PROJECT_DIR = Path(__file__).resolve().parents[1]

    INPUT_DIR = PROJECT_DIR / "data" / "raw_rd"
    OUTPUT_DIR = PROJECT_DIR / "figures" / "output"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plan_paths = [
        INPUT_DIR / "1.pdf",
        INPUT_DIR / "3.pdf",
        INPUT_DIR / "4.pdf",
        INPUT_DIR / "2.pdf",
        INPUT_DIR / "5.pdf",
        INPUT_DIR / "6.pdf",
    ]

    output_pdf = OUTPUT_DIR / "real_estate_floorplan_examples_2.pdf"

    for path in plan_paths:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

    create_2x3_figure(
        plan_paths=[str(path) for path in plan_paths],
        output_pdf=str(output_pdf),
    )


if __name__ == "__main__":
    main()