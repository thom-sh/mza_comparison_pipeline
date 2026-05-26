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
    plan_paths = [
        r"C:\WF\Thomas Sharon\Floorplan_Dataset\Real_estate_data\Footprints_final\1.pdf",
        r"C:\Users\thom_sh\Desktop\3.pdf",
        r"C:\Users\thom_sh\Desktop\4.png",
        r"C:\Users\thom_sh\Desktop\2.png",
        r"C:\Users\thom_sh\Desktop\5.png",
        r"C:\WF\Thomas Sharon\Floorplan_Dataset\Real_estate_data\Footprints_final\6.pdf",
    ]

    output_pdf = r"C:\WF\Thomas Sharon\Floorplan_Dataset\Real_estate_data\real_estate_floorplan_examples_2.pdf"

    for path in plan_paths:
        if not Path(path).exists():
            raise FileNotFoundError(f"File not found: {path}")

    create_2x3_figure(plan_paths, output_pdf)


if __name__ == "__main__":
    main()