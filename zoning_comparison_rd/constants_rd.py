import matplotlib.colors as mcolors
from matplotlib.cm import get_cmap

# ============================================================
#     SIMPLE TWO-CLASS MAPPING (Your New Logic)
# ============================================================

ROOM_TYPE_NAMES = {
    1: "Stair",
    0: "Apartment"
}

# Colors for visualization
ROOM_TYPE_COLORS = {
    1: "#1f77b4",   # Stair → Blue
    0: "#ff7f0e"    # Apartment → Orange
}

COLOR_MAP_SIMPLE = mcolors.ListedColormap([
    ROOM_TYPE_COLORS[0],   # Apartment
    ROOM_TYPE_COLORS[1]    # Stair
])

CMAP_SIMPLE = get_cmap(COLOR_MAP_SIMPLE)

