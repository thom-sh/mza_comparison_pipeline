from draw_polygons import draw_polygons

def scale_known(pdf_path):
    polygons, img_crop, CW, CH, W, H = draw_polygons(pdf_path)

    scale = float(input("\nEnter scale denominator (1:50 → 50): "))
    dpi = 300

    # mm per pixel on paper
    px_to_mm = 25.4 / dpi

    # real-world mm per pixel
    px_to_mm_real = px_to_mm * scale

    # convert to meters
    px_to_m = px_to_mm_real / 1000

    real_polygons = []

    for poly in polygons:
        real_poly = [(x * px_to_m, y * px_to_m) for (x, y) in poly]
        real_polygons.append(real_poly)

    return real_polygons
