import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===============================================================
# CONFIG
# ===============================================================
GLOBAL_CSV = r"C:\WF\Thomas Sharon\Floorplan_Dataset\comparison_logic\all_global_stats_selected_200.csv"

# Thresholds for quadrant view (adjust if you want)
IOU_REF = 0.70
RECALL_REF = 0.90  # use 0.90 as a more meaningful split than 1.0; set to 1.0 if you prefer

# Jitter settings for recall pile-up at 1.0
JITTER_EPS = 0.008  # small horizontal jitter

# ===============================================================
# LOAD DATA
# ===============================================================
df = pd.read_csv(GLOBAL_CSV)

# Basic columns used
x = df["region_recall"].astype(float).to_numpy()
y = df["mean_iou_matched"].astype(float).to_numpy()

# ===============================================================
# 1) DENSITY-AWARE SCATTER (alpha + jitter at recall=1.0)
# ===============================================================
x_j = x.copy()

# Add tiny jitter only to points that are exactly 1.0 (or extremely close)
mask_one = np.isclose(x_j, 1.0)
x_j[mask_one] = np.clip(
    x_j[mask_one] + np.random.uniform(-JITTER_EPS, JITTER_EPS, size=mask_one.sum()),
    0, 1
)

plt.figure(figsize=(6, 6))
plt.scatter(x_j, y, alpha=0.6)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("Region Recall (completeness)")
plt.ylabel("Mean IoU (matched regions)")
plt.title("Recall vs Mean Matched IoU (density-aware)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ===============================================================
# 2) QUADRANT PLOT (explicit regimes)
# ===============================================================
plt.figure(figsize=(6, 6))
plt.scatter(x_j, y, alpha=0.6)

# Reference lines
plt.axvline(RECALL_REF, linestyle="--", linewidth=1)
plt.axhline(IOU_REF, linestyle="--", linewidth=1)

# Quadrant labels (placed roughly in centers)
plt.text(0.05, 0.95, "Low recall\nHigh IoU", va="top", fontsize=10)
plt.text(0.95, 0.95, "High recall\nHigh IoU", va="top", ha="right", fontsize=10)
plt.text(0.05, 0.05, "Low recall\nLow IoU", va="bottom", fontsize=10)
plt.text(0.95, 0.05, "High recall\nLow IoU", va="bottom", ha="right", fontsize=10)

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("Region Recall (completeness)")
plt.ylabel("Mean IoU (matched regions)")
plt.title("Geometric Accuracy vs Zoning Completeness (quadrants)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ===============================================================
# 3) BUBBLE PLOT (3rd variable = GT region count, if available)
# ===============================================================
# We need a column that represents GT region count. If not present, we skip gracefully.
# If you DO have 'false_negatives' and recall, you can reconstruct n_gt approximately:
# recall = matched / n_gt and false_negatives = n_gt - matched  -> n_gt = false_negatives / (1 - recall)
# This is valid when recall < 1. For recall==1, false_negatives=0 but n_gt unknown.
# So bubble plot is best if you have an explicit column, e.g. 'n_gt_regions'.

bubble_col = None
for candidate in ["n_gt_regions", "num_gt_regions", "gt_region_count", "n_regions_gt"]:
    if candidate in df.columns:
        bubble_col = candidate
        break

if bubble_col is not None:
    s_raw = df[bubble_col].astype(float).to_numpy()
    # Scale bubble sizes to a visually reasonable range
    s = 30 + 15 * (s_raw - np.min(s_raw)) / (np.max(s_raw) - np.min(s_raw) + 1e-9)

    plt.figure(figsize=(6, 6))
    plt.scatter(x_j, y, s=s, alpha=0.5)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Region Recall (completeness)")
    plt.ylabel("Mean IoU (matched regions)")
    plt.title(f"Recall vs Mean Matched IoU (bubble size = {bubble_col})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
else:
    print("[INFO] No explicit GT region-count column found (e.g., 'n_gt_regions'). Skipping bubble plot.")

# ===============================================================
# 4) SCATTER WITH MARGINAL HISTOGRAMS (FIXED: constrained_layout)
# ===============================================================

fig = plt.figure(figsize=(7, 7), constrained_layout=True)

# Grid specification
gs = fig.add_gridspec(
    2, 2,
    width_ratios=[4, 1.3],
    height_ratios=[1.3, 4]
)

ax_histx = fig.add_subplot(gs[0, 0])
ax_scatter = fig.add_subplot(gs[1, 0])
ax_histy = fig.add_subplot(gs[1, 1])

# ---- Scatter ----
ax_scatter.scatter(x_j, y, alpha=0.6)
ax_scatter.set_xlim(0, 1)
ax_scatter.set_ylim(0, 1)
ax_scatter.set_xlabel("Region Recall (completeness)")
ax_scatter.set_ylabel("Mean IoU (matched regions)")
ax_scatter.grid(True, alpha=0.3)

# ---- Top histogram (Recall) ----
ax_histx.hist(x, bins=15, edgecolor="black")
ax_histx.set_xlim(0, 1)
ax_histx.set_xticks([])
ax_histx.set_ylabel("Count")

# ---- Right histogram (IoU) ----
ax_histy.hist(y, bins=15, orientation="horizontal", edgecolor="black")
ax_histy.set_ylim(0, 1)
ax_histy.set_yticks([])
ax_histy.set_xlabel("Count")

fig.suptitle(
    "Recall vs Mean Matched IoU with Marginal Distributions",
    fontsize=12
)

plt.show()
