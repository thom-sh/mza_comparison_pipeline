from PIL import Image
from pathlib import Path

ids = [53, 54, 55, 56, 57, 58]

input_folder = Path(r"C:\WF\Thomas Sharon\Floorplan_Dataset\Real_estate_data\Footprints")
output_folder = Path(r"C:\WF\Thomas Sharon\Floorplan_Dataset\Real_estate_data\Footprints_final")

output_folder.mkdir(parents=True, exist_ok=True)

for i in ids:
    input_file = input_folder / str(i)          # file has no extension
    output_file = output_folder / f"{i}.pdf"

    img = Image.open(input_file)
    img = img.convert("RGB")                   # required for PDF export
    img.save(output_file, "PDF")

    print("PDF created:", output_file)