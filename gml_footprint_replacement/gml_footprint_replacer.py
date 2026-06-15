import os
import json
import numpy as np
from lxml import etree
from shapely.geometry import Polygon, shape
from shapely.ops import unary_union
from shapely.affinity import rotate as shapely_rotate, translate as shapely_translate

# ============================================================
#                 GLOBAL SETTINGS / CONSTANTS
# ============================================================

DEFAULT_SOURCE_EPSG = 25833

NS = {
    "gml":  "http://www.opengis.net/gml",
    "core": "http://www.opengis.net/citygml/1.0",
    "bldg": "http://www.opengis.net/citygml/building/1.0",
    "app":  "http://www.opengis.net/citygml/appearance/1.0",
    "gen":  "http://www.opengis.net/citygml/generics/1.0",
    "xlink":"http://www.w3.org/1999/xlink",
    "xsi":  "http://www.w3.org/2001/XMLSchema-instance",
}

# Fixed rotation angle for building DEBE01YYK0002Uqm:
FIXED_ANGLE = -164.22  # degrees


# ============================================================
#               BASIC GEOMETRY HELPERS (GML)
# ============================================================

def ring_is_ccw(coords2d: np.ndarray) -> bool:
    x, y = coords2d[:, 0], coords2d[:, 1]
    return np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)) > 0

def close_ring(coords2d: np.ndarray) -> np.ndarray:
    if not np.allclose(coords2d[0], coords2d[-1]):
        coords2d = np.vstack([coords2d, coords2d[0]])
    return coords2d

def get_poslist_xyz(el_poslist_text: str) -> np.ndarray:
    vals = [float(v) for v in el_poslist_text.strip().split()]
    return np.array(vals, dtype=float).reshape((-1, 3))

def polygon_to_linear_ring_poslist(coords3d: np.ndarray) -> str:
    return " ".join(f"{x:.3f} {y:.3f} {z:.3f}" for x, y, z in coords3d)


# ------------------------------------------------------------

def extract_ground_roof_info(bldg_el):
    srsName = None
    first_ring = bldg_el.find(".//gml:LinearRing", namespaces=NS)
    if first_ring is not None:
        srsName = first_ring.get("{http://www.opengis.net/gml}srsName", None)

    ground_pos, roof_pos = [], []

    for poly in bldg_el.findall(".//bldg:GroundSurface//gml:Polygon", namespaces=NS):
        pos = poly.find(".//gml:posList", namespaces=NS)
        if pos is not None and pos.text:
            ground_pos.append(get_poslist_xyz(pos.text))

    for poly in bldg_el.findall(".//bldg:RoofSurface//gml:Polygon", namespaces=NS):
        pos = poly.find(".//gml:posList", namespaces=NS)
        if pos is not None and pos.text:
            roof_pos.append(get_poslist_xyz(pos.text))

    groundZ = np.median(np.vstack(ground_pos)[:, 2]) if ground_pos else 0.0

    if roof_pos:
        roofZ = np.median(np.vstack(roof_pos)[:, 2])
        height = roofZ - groundZ
    else:
        height = 3.0
        roofZ = groundZ + height

    return groundZ, roofZ, height, srsName


# ------------------------------------------------------------

def get_ground_polygon(bldg_el) -> Polygon | None:
    polys = []
    for poly in bldg_el.findall(".//bldg:GroundSurface//gml:Polygon", namespaces=NS):
        pos = poly.find(".//gml:posList", namespaces=NS)
        if pos is not None and pos.text:
            arr = get_poslist_xyz(pos.text)
            pts2d = close_ring(arr[:, :2])
            p = Polygon(pts2d)
            if p.is_valid and not p.is_empty:
                polys.append(p)

    if not polys:
        return None

    return unary_union(polys)


# ------------------------------------------------------------

def make_gml_polygon(coords3d, srsName=None):
    gml_pol = etree.Element("{%s}Polygon" % NS["gml"])
    if srsName:
        gml_pol.set("{%s}srsName" % NS["gml"], srsName)
    exterior = etree.SubElement(gml_pol, "{%s}exterior" % NS["gml"])
    lr = etree.SubElement(exterior, "{%s}LinearRing" % NS["gml"])
    pos = etree.SubElement(lr, "{%s}posList" % NS["gml"])
    pos.text = polygon_to_linear_ring_poslist(coords3d)
    return gml_pol


# ============================================================
#                     LOD2 GEOMETRY BUILD
# ============================================================

def rebuild_lod2_solid(bldg_el, footprint2d, groundZ, roofZ, srsName):
    for elem in bldg_el.findall("./bldg:lod2Solid", namespaces=NS):
        bldg_el.remove(elem)

    ext2d = close_ring(np.array(footprint2d.exterior.coords))
    if not ring_is_ccw(ext2d):
        ext2d = ext2d[::-1]

    bottom = np.column_stack([ext2d[:,0], ext2d[:,1], np.full(len(ext2d), groundZ)])
    top    = np.column_stack([ext2d[:,0], ext2d[:,1], np.full(len(ext2d), roofZ)])

    lod2 = etree.SubElement(bldg_el, "{%s}lod2Solid" % NS["bldg"])
    solid = etree.SubElement(lod2, "{%s}Solid" % NS["gml"])
    exterior = etree.SubElement(solid, "{%s}exterior" % NS["gml"])
    comp = etree.SubElement(exterior, "{%s}CompositeSurface" % NS["gml"])

    def add(coords): 
        sm = etree.SubElement(comp, "{%s}surfaceMember" % NS["gml"])
        sm.append(make_gml_polygon(coords, srsName))

    add(top)
    add(bottom[::-1])

    for i in range(len(ext2d)-1):
        wall = np.array([
            bottom[i], bottom[i+1],
            top[i+1], top[i],
            bottom[i]
        ])
        add(wall)


# ------------------------------------------------------------

def rebuild_bounded_by_surfaces(bldg_el, footprint2d, groundZ, roofZ, srsName):
    for bb in bldg_el.findall("./bldg:boundedBy", namespaces=NS):
        bldg_el.remove(bb)

    ext2d = close_ring(np.array(footprint2d.exterior.coords))
    if not ring_is_ccw(ext2d):
        ext2d = ext2d[::-1]

    bottom = np.column_stack([ext2d[:,0], ext2d[:,1], np.full(len(ext2d), groundZ)])
    top    = np.column_stack([ext2d[:,0], ext2d[:,1], np.full(len(ext2d), roofZ)])

    # Ground
    bb = etree.SubElement(bldg_el, "{%s}boundedBy" % NS["bldg"])
    gs = etree.SubElement(bb, "{%s}GroundSurface" % NS["bldg"])
    ms = etree.SubElement(etree.SubElement(etree.SubElement(gs, "{%s}lod2MultiSurface" % NS["bldg"]), "{%s}MultiSurface" % NS["gml"]), "{%s}surfaceMember" % NS["gml"])
    ms.append(make_gml_polygon(bottom[::-1], srsName))

    # Roof
    bb = etree.SubElement(bldg_el, "{%s}boundedBy" % NS["bldg"])
    rs = etree.SubElement(bb, "{%s}RoofSurface" % NS["bldg"])
    ms = etree.SubElement(etree.SubElement(etree.SubElement(rs, "{%s}lod2MultiSurface" % NS["bldg"]), "{%s}MultiSurface" % NS["gml"]), "{%s}surfaceMember" % NS["gml"])
    ms.append(make_gml_polygon(top, srsName))

    # Walls
    for i in range(len(ext2d)-1):
        bb = etree.SubElement(bldg_el, "{%s}boundedBy" % NS["bldg"])
        ws = etree.SubElement(bb, "{%s}WallSurface" % NS["bldg"])
        ms = etree.SubElement(etree.SubElement(etree.SubElement(ws, "{%s}lod2MultiSurface" % NS["bldg"]), "{%s}MultiSurface" % NS["gml"]), "{%s}surfaceMember" % NS["gml"])
        wall = np.array([
            bottom[i], bottom[i+1],
            top[i+1], top[i],
            bottom[i]
        ])
        ms.append(make_gml_polygon(wall, srsName))


# ============================================================
#                   GEOJSON FOOTPRINT READER
# ============================================================

def read_footprint_geojson(path: str) -> Polygon:
    with open(path, "r", encoding="utf-8") as f:
        gj = json.load(f)
    geom = gj["geometry"] if "geometry" in gj else gj["features"][0]["geometry"]
    shp = shape(geom)
    if shp.geom_type == "MultiPolygon":
        shp = max(shp.geoms, key=lambda g: g.area)
    return shp


# ============================================================
#              MAIN: REPLACE BUILDING FOOTPRINT
# ============================================================

def replace_building_footprint(input_gml, output_gml, target_id, geojson_fp, source_epsg=DEFAULT_SOURCE_EPSG):

    tree = etree.parse(input_gml)
    root = tree.getroot()

    # Find building
    bldg_el = root.find(f".//bldg:Building[@gml:id='{target_id}']", namespaces=NS)
    if bldg_el is None:
        raise ValueError(f"Building {target_id} not found")

    print(f"🏗️ Found building: {target_id}")

    # Extract heights
    groundZ, roofZ, height, srsName = extract_ground_roof_info(bldg_el)
    if not srsName:
        srsName = f"urn:ogc:def:crs:EPSG::{source_epsg}"

    old_fp = get_ground_polygon(bldg_el)
    old_centroid = old_fp.centroid
    print(f"🏁 Original centroid: ({old_centroid.x:.2f}, {old_centroid.y:.2f})")

    # Load new footprint
    new_fp = read_footprint_geojson(geojson_fp)
    if not new_fp.is_valid:
        new_fp = new_fp.buffer(0)

    # --- FIXED ORIENTATION ---
    print(f"📐 Using FIXED rotation angle: {FIXED_ANGLE}°")

    # Rotate
    new_fp_rot = shapely_rotate(new_fp, FIXED_ANGLE, origin="centroid", use_radians=False)

    # Translate
    dx = old_centroid.x - new_fp_rot.centroid.x
    dy = old_centroid.y - new_fp_rot.centroid.y
    new_fp_aligned = shapely_translate(new_fp_rot, xoff=dx, yoff=dy)

    print(f"🔄 Translation: dx={dx:.2f}, dy={dy:.2f}")

    # Replace geometry
    rebuild_lod2_solid(bldg_el, new_fp_aligned, groundZ, roofZ, srsName)
    rebuild_bounded_by_surfaces(bldg_el, new_fp_aligned, groundZ, roofZ, srsName)

    # Write output (NO pretty_print → fast!)
    tree.write(output_gml, xml_declaration=True, encoding="utf-8")

    print(f"✅ Saved updated GML:\n{output_gml}")
    return output_gml
