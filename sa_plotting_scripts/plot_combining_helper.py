#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from pypdf import PdfReader, PdfWriter, PageObject, Transformation
from pypdf.generic import RectangleObject


# ============================================================
# CONFIGURATION
# ============================================================
INPUT_PDFS = [
    Path(r"../new_plots/output/matrices/peak_heating_kW_pairwise_percent_heatmap_P5_all_runs.pdf"),
    Path(r"../new_plots/output/matrices/peak_heating_kW_pairwise_percent_heatmap_P50_all_runs.pdf"),
    Path(r"../new_plots/output/matrices/peak_heating_kW_pairwise_percent_heatmap_P95_all_runs.pdf"),
]



OUTPUT_PDF = Path(r"../new_plots/output/matrices/combined_graphs_peak_heating.pdf")

# layout
N_COLS = 3
N_ROWS = 1

# spacing between graphs
GAP_X = 12
GAP_Y = 18

# outer margins
MARGIN_LEFT = 10
MARGIN_RIGHT = 10
MARGIN_TOP = 10
MARGIN_BOTTOM = 10

# label settings
ADD_PANEL_LABELS = True
PANEL_LABELS = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
LABEL_FONT_SIZE = 12
LABEL_OFFSET_X = 4
LABEL_OFFSET_Y = 4
LABEL_BOX_HEIGHT = 18   # reserved space above each graph if labels are enabled

# crop settings (points)
# positive values trim inward from each side
CROP_LEFT = 0
CROP_RIGHT = 0
CROP_BOTTOM = 0
CROP_TOP = 0

USE_ALL_PAGES = False
# ============================================================


def iter_input_pages(pdf_paths: Iterable[Path], use_all_pages: bool) -> list[PageObject]:
    pages: list[PageObject] = []

    for pdf_path in pdf_paths:
        if not pdf_path.exists():
            raise FileNotFoundError(f"Input PDF not found: {pdf_path}")

        reader = PdfReader(str(pdf_path))
        if len(reader.pages) == 0:
            continue

        if use_all_pages:
            for page in reader.pages:
                pages.append(page)
        else:
            pages.append(reader.pages[0])

    if not pages:
        raise RuntimeError("No input pages found.")

    return pages


def clone_page(page: PageObject) -> PageObject:
    # safe independent copy
    return PageObject.create_blank_page(
        width=float(page.mediabox.width),
        height=float(page.mediabox.height),
    ).merge_page(page)


def crop_page(page: PageObject, left: float, right: float, bottom: float, top: float) -> PageObject:
    src = page

    x0 = float(src.mediabox.left) + left
    y0 = float(src.mediabox.bottom) + bottom
    x1 = float(src.mediabox.right) - right
    y1 = float(src.mediabox.top) - top

    if x1 <= x0 or y1 <= y0:
        raise ValueError("Crop values are too large and removed the full page.")

    src.cropbox = RectangleObject((x0, y0, x1, y1))
    src.mediabox = RectangleObject((x0, y0, x1, y1))
    return src


def get_page_size(page: PageObject) -> tuple[float, float]:
    return float(page.mediabox.width), float(page.mediabox.height)


def make_blank_page(width: float, height: float) -> PageObject:
    return PageObject.create_blank_page(width=width, height=height)


def draw_label_box_text_placeholder(
    out_page: PageObject,
    label: str,
    x: float,
    y: float,
) -> None:
    """
    pypdf does not directly support easy text drawing like matplotlib/reportlab.
    So this function is left as a placeholder for layout spacing only.

    The page still reserves label space above each graph.
    """
    _ = (out_page, label, x, y)


def build_output_page(
    pages: list[PageObject],
    n_cols: int,
    n_rows: int,
    gap_x: float,
    gap_y: float,
    margin_left: float,
    margin_right: float,
    margin_top: float,
    margin_bottom: float,
    add_panel_labels: bool,
) -> PageObject:
    if not pages:
        raise RuntimeError("No pages supplied for output page.")

    # assume all graphs should stay at their own native size
    widths = []
    heights = []
    for p in pages:
        w, h = get_page_size(p)
        widths.append(w)
        heights.append(h)

    # grid-wise max sizes per column/row
    col_widths = [0.0] * n_cols
    row_heights = [0.0] * n_rows

    for i, p in enumerate(pages):
        r = i // n_cols
        c = i % n_cols
        w, h = get_page_size(p)
        if add_panel_labels:
            h += LABEL_BOX_HEIGHT
        col_widths[c] = max(col_widths[c], w)
        row_heights[r] = max(row_heights[r], h)

    total_width = margin_left + sum(col_widths) + gap_x * (n_cols - 1) + margin_right
    total_height = margin_bottom + sum(row_heights) + gap_y * (n_rows - 1) + margin_top

    out_page = make_blank_page(total_width, total_height)

    # place from top to bottom
    current_top = total_height - margin_top

    for r in range(n_rows):
        row_h = row_heights[r]
        row_bottom = current_top - row_h

        current_left = margin_left
        for c in range(n_cols):
            idx = r * n_cols + c
            if idx >= len(pages):
                break

            p = pages[idx]
            pw, ph = get_page_size(p)

            label_space = LABEL_BOX_HEIGHT if add_panel_labels else 0.0

            # left-aligned in each cell, graph below label area
            x = current_left
            y = row_bottom

            graph_y = y
            if add_panel_labels:
                graph_y = y
                label_x = x + LABEL_OFFSET_X
                label_y = y + ph + LABEL_OFFSET_Y
                if idx < len(PANEL_LABELS):
                    draw_label_box_text_placeholder(out_page, PANEL_LABELS[idx], label_x, label_y)

            transform = Transformation().translate(
                tx=x - float(p.mediabox.left),
                ty=graph_y - float(p.mediabox.bottom),
            )
            out_page.merge_transformed_page(p, transform)

            current_left += col_widths[c] + gap_x

        current_top = row_bottom - gap_y

    return out_page


def combine_pdfs() -> None:
    input_pages = iter_input_pages(INPUT_PDFS, USE_ALL_PAGES)

    processed_pages: list[PageObject] = []
    for p in input_pages:
        page = p
        if any(v != 0 for v in [CROP_LEFT, CROP_RIGHT, CROP_BOTTOM, CROP_TOP]):
            page = crop_page(page, CROP_LEFT, CROP_RIGHT, CROP_BOTTOM, CROP_TOP)
        processed_pages.append(page)

    writer = PdfWriter()
    per_page = N_COLS * N_ROWS

    for start in range(0, len(processed_pages), per_page):
        chunk = processed_pages[start:start + per_page]
        out_page = build_output_page(
            pages=chunk,
            n_cols=N_COLS,
            n_rows=N_ROWS,
            gap_x=GAP_X,
            gap_y=GAP_Y,
            margin_left=MARGIN_LEFT,
            margin_right=MARGIN_RIGHT,
            margin_top=MARGIN_TOP,
            margin_bottom=MARGIN_BOTTOM,
            add_panel_labels=ADD_PANEL_LABELS,
        )
        writer.add_page(out_page)

    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PDF, "wb") as f:
        writer.write(f)

    print(f"Saved: {OUTPUT_PDF}")


if __name__ == "__main__":
    combine_pdfs()