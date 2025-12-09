# scale_area.py
# ----------------------------------------------------------
# Method 3: Scale floorplan using TOTAL REAL area of ALL polygons
# ----------------------------------------------------------

import math
from shapely.geometry import Polygon
from shapely.ops import unary_union
from draw_polygons import draw_polygons   # your polygon drawing tool


def ask_real_area_total():
    """Ask user for total real area in m²."""
    while True:
        try:
            val = float(input("\nEnter TOTAL REAL AREA of all polygons combined (m²): "))
            if val > 0:
                return val
            print("Area must be positive.")
        except:
            print("Invalid number. Try again.")


def scale_area(pdf_path):
    """
    NEW LOGIC:
    1. Draw ALL polygons
    2. Compute UNION area of all polygons (avoids overlap double-counting)
    3. Ask user for TOTAL real-world area (m²)
    4. Compute global scale factor from area ratio
    5. Convert ALL polygons to real-world coordinates
    """
    print("\n=== AREA-BASED SCALING (TOTAL AREA) ===")
    print("Draw ALL polygons on the floorplan...")

    # Step 1: Get polygons (pixel coordinates)
    polys, img_crop, CW, CH, W, H = draw_polygons(pdf_path)

    if not polys:
        print("No polygons drawn.")
        return []

    # Step 2: compute area of union (in pixel units)
    shapely_polys = [Polygon(p) for p in polys]
    union_poly = unary_union(shapely_polys)
    area_px = union_poly.area

    print(f"\nPixel UNION area = {area_px:.2f} px²")

    # Step 3: estimate pixel→mm based on PDF physical size
    import fitz
    doc = fitz.open(pdf_path)
    page = doc[0]

    pt_to_mm = 0.352778
    pdf_w_mm = page.rect.width * pt_to_mm
    pdf_h_mm = page.rect.height * pt_to_mm

    px_to_mm_x = pdf_w_mm / W
    px_to_mm_y = pdf_h_mm / H

    # mm → meter
    px_to_m_initial = (px_to_mm_x + px_to_mm_y) / 2000.0

    print(f"PDF paper-based px→m estimate = {px_to_m_initial:.7f} m/px")

    # Step 4: compute drawing-area estimate in m²
    area_drawing_m2 = area_px * (px_to_m_initial ** 2)
    print(f"Drawing UNION area estimate = {area_drawing_m2:.4f} m²")

    # Step 5: ask user for total REAL area
    real_area_total = ask_real_area_total()

    # Step 6: compute global scale factor
    S = math.sqrt(real_area_total / area_drawing_m2)
    print(f"\nEstimated global scale factor: 1:{S:.2f}")

    # Step 7: corrected pixel→meter factor
    px_to_m_corrected = px_to_m_initial * S
    print(f"Corrected px→m = {px_to_m_corrected:.7f} m/px")

    # Step 8: scale ALL polygons
    real_polygons = []
    for poly in polys:
        real_poly = [(x * px_to_m_corrected, y * px_to_m_corrected) for (x, y) in poly]
        real_polygons.append(real_poly)

    print("\n✔ Conversion complete.")
    return real_polygons
