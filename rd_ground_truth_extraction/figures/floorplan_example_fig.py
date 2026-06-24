import fitz  # PyMuPDF
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path


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
    """
    Crop outer white margins.
    """
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

    # 2 rows, 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(6.14, 3.8))
    axes = axes.flatten()

    # Turn all axes off first
    for ax in axes:
        ax.axis("off")

    # Plot all images
    for i, img in enumerate(images):
        axes[i].imshow(img)
        axes[i].axis("off")

    plt.subplots_adjust(
        left=0.01,
        right=0.99,
        top=0.96,
        bottom=0.02,
        wspace=0.02,
        hspace=0.06
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

    output_pdf = OUTPUT_DIR / "real_estate_floorplan_examples.pdf"

    for path in plan_paths:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

    create_2x3_figure(
        plan_paths=[str(path) for path in plan_paths],
        output_pdf=str(output_pdf),
    )


if __name__ == "__main__":
    main()