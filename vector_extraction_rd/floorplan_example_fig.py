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
    fig, axes = plt.subplots(2, 3, figsize=(8.27, 4.0))
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
    plan_paths = [
        r"C:\WF\Thomas Sharon\Floorplan_Dataset\Real_estate_data\Footprints_final\1.pdf",
        r"C:\Users\thom_sh\Desktop\3.pdf",
        r"C:\Users\thom_sh\Desktop\4.png",
        r"C:\Users\thom_sh\Desktop\2.png",
        r"C:\Users\thom_sh\Desktop\5.png",
        r"C:\WF\Thomas Sharon\Floorplan_Dataset\Real_estate_data\Footprints_final\6.pdf",
    ]

    output_pdf = r"C:\WF\Thomas Sharon\Floorplan_Dataset\Real_estate_data\real_estate_floorplan_examples.pdf"

    for path in plan_paths:
        if not Path(path).exists():
            raise FileNotFoundError(f"File not found: {path}")

    create_2x3_figure(plan_paths, output_pdf)


if __name__ == "__main__":
    main()