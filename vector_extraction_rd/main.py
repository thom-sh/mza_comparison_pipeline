from scale_known import scale_known
from scale_reference import scale_reference
from scale_area import scale_area
from display_polygons import display_polygons
from save_pickle import classify_and_save_pickle   # <-- NEW

import os

print("\n=== FLOORPLAN SCALING TOOL ===")

# --------------------------------------
# CONFIGURATION
# --------------------------------------

# Folder containing PDFs (each named buildingID.pdf)
PDF_FOLDER = r"C:\WF\Thomas Sharon\Floorplan_Dataset\Real_estate_data\Footprints_final"

# OUTPUT folder for pickles
OUTPUT_PICKLE_FOLDER = r"C:\WF\Thomas Sharon\Floorplan_Dataset\rd_pickle"

# Ask user for building ID
building_id = 33

# Compose PDF path automatically
pdf_path = os.path.join(PDF_FOLDER, f"{building_id}.pdf")

if not os.path.exists(pdf_path):
    print(f"❌ PDF not found:\n{pdf_path}")
    exit()

print(f"\nUsing PDF: {pdf_path}")

# --------------------------------------
# CHOOSE SCALING METHOD
# --------------------------------------

print("\nChoose scaling method:")
print("1 = Known scale (1:50, 1:100, etc.)")
print("2 = Reference line (click 2 points)")
print("3 = Area-based scaling (enter real area of a polygon)")

choice = input("Enter 1, 2, or 3: ").strip()


# --------------------------------------
# RUN SELECTED SCALING METHOD
# --------------------------------------

if choice == "1":
    real_polys = scale_known(pdf_path)

elif choice == "2":
    real_polys = scale_reference(pdf_path)

elif choice == "3":
    real_polys = scale_area(pdf_path)

else:
    print("Invalid choice.")
    exit()


# --------------------------------------
# PRINT REAL-WORLD POLYGONS
# --------------------------------------

print("\n=== REAL-WORLD POLYGONS (meters) ===")
for i, poly in enumerate(real_polys, 1):
    print(f"\nPolygon {i}:")
    for p in poly:
        print(p)


# --------------------------------------
# AREA COMPUTATION (in m²)
# --------------------------------------

def polygon_area(poly):
    area = 0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        area += x1 * y2 - y1 * x2
    return abs(area) / 2


print("\n=== POLYGON AREAS (m²) ===")

for i, poly in enumerate(real_polys, 1):
    A = polygon_area(poly)
    print(f"Polygon {i} --> Area: {A:.3f} m²")


# --------------------------------------
# DISPLAY POLYGONS
# --------------------------------------

display_polygons(real_polys)


# --------------------------------------
# SAVE AS SWISS-FORMAT PICKLE
# --------------------------------------

output_pickle = os.path.join(OUTPUT_PICKLE_FOLDER, f"{building_id}.pickle")

classify_and_save_pickle(real_polys, output_pickle)

print("\n🎉 All Done!")
