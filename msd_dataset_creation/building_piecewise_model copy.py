import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ============================================================
# USER SETTINGS (EDIT HERE)
# ============================================================
USE_TARGET = "mean"   # "mean" or "median"
K_FORCED   = 6          # <--- force number of regimes (K=1 means single linear)
MIN_PER_REGIME = 30     # safety: minimum train samples per regime
RANDOM_STATE = 42

# ============================================================
# 0) LOAD + BUILD BUILDING-LEVEL DATASET (choose target)
# ============================================================

csv_path = r"C:\WF\Thomas Sharon\Floorplan_Dataset\msd_dataset_creation\msd_apt_list.csv"
df = pd.read_csv(csv_path)

# Identify apartment area columns
apt_cols = [c for c in df.columns if c.startswith("apt_") and c.endswith("_area")]

# Convert to numeric
for c in apt_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["footprint_area"] = pd.to_numeric(df["footprint_area"], errors="coerce")

# Extract valid apartment areas
apt_values = df[apt_cols].to_numpy(dtype=float)
valid = np.where((apt_values > 0) & np.isfinite(apt_values), apt_values, np.nan)

df["n_valid_apts"] = np.sum(np.isfinite(valid), axis=1)

# Choose target definition
if USE_TARGET == "mean":
    df["y_target"] = np.nanmean(valid, axis=1)
elif USE_TARGET == "median":
    df["y_target"] = np.nanmedian(valid, axis=1)
else:
    raise ValueError("USE_TARGET must be 'mean' or 'median'")

# Quality filtering
df_clean = df.dropna(subset=["building_id", "footprint_area", "y_target"]).copy()
df_clean = df_clean[df_clean["footprint_area"] > 0]
df_clean = df_clean[df_clean["n_valid_apts"] >= 2]

# Final building-level table
g = df_clean[["building_id", "footprint_area", "y_target", "n_valid_apts"]].copy()
g = g.rename(columns={"footprint_area": "A", "y_target": "y"}).reset_index(drop=True)

print("Buildings after cleaning:", len(g))
print(g.head())

# ============================================================
# Metrics
# ============================================================

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_pred - y_true)))

def bias(y_true, y_pred):
    return float(np.mean(y_pred - y_true))

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

# ============================================================
# STEP 1) OUTER split: TRAIN / TEST (no leakage)
# ============================================================

g["A_bin"] = pd.qcut(g["A"], q=4, labels=False, duplicates="drop")

train_idx, test_idx = train_test_split(
    g.index,
    test_size=0.3,
    random_state=RANDOM_STATE,
    stratify=g["A_bin"]
)

train = g.loc[train_idx].copy().reset_index(drop=True)
test  = g.loc[test_idx].copy().reset_index(drop=True)

print("OUTER TRAIN:", len(train), "OUTER TEST:", len(test))

# ============================================================
# Piecewise helpers (K regimes, K-1 breakpoints) — FORCED K
# ============================================================

def fit_piecewise_linear(df_, breaks):
    """Fit separate LinearRegression per regime. returns list of models + breaks."""
    breaks = tuple(sorted(breaks))
    models = []
    for r in range(len(breaks) + 1):
        if len(breaks) == 0:
            seg = df_
        elif r == 0:
            seg = df_[df_["A"] <= breaks[0]]
        elif r == len(breaks):
            seg = df_[df_["A"] > breaks[-1]]
        else:
            seg = df_[(df_["A"] > breaks[r-1]) & (df_["A"] <= breaks[r])]

        if len(seg) < 2:
            raise ValueError(f"Segment {r+1} has too few samples ({len(seg)}) to fit a regression.")

        m = LinearRegression()
        m.fit(seg[["A"]], seg["y"])
        models.append(m)
    return models, breaks

def predict_piecewise(A_values, models, breaks):
    """Vectorized prediction."""
    A_values = np.asarray(A_values, dtype=float)
    yhat = np.empty_like(A_values, dtype=float)

    if len(breaks) == 0:
        yhat[:] = models[0].predict(pd.DataFrame({"A": A_values}))
        return yhat

    masks = []
    masks.append(A_values <= breaks[0])
    for i in range(1, len(breaks)):
        masks.append((A_values > breaks[i-1]) & (A_values <= breaks[i]))
    masks.append(A_values > breaks[-1])

    for r, mask in enumerate(masks):
        if mask.any():
            yhat[mask] = models[r].predict(pd.DataFrame({"A": A_values[mask]}))
    return yhat

def counts_per_regime(df_, breaks):
    breaks = tuple(sorted(breaks))
    if len(breaks) == 0:
        return [len(df_)]
    counts = []
    counts.append((df_["A"] <= breaks[0]).sum())
    for i in range(1, len(breaks)):
        counts.append(((df_["A"] > breaks[i-1]) & (df_["A"] <= breaks[i])).sum())
    counts.append((df_["A"] > breaks[-1]).sum())
    return counts

# ============================================================
# STEP 2) FORCE K regimes by quantile breakpoints on TRAIN
# ============================================================

if K_FORCED < 1:
    raise ValueError("K_FORCED must be >= 1")

if K_FORCED == 1:
    breaks_star = tuple()
else:
    # K regimes -> K-1 breakpoints at equally spaced quantiles (train only)
    qs = np.linspace(0, 1, K_FORCED + 1)[1:-1]  # exclude 0 and 1
    breaks_star = tuple(np.quantile(train["A"].values, qs))

# Safety: ensure each regime has enough samples
cnt_train = counts_per_regime(train, breaks_star)
cnt_test  = counts_per_regime(test, breaks_star)

print("\n=== FORCED PIECEWISE SETUP ===")
print("K_FORCED:", K_FORCED)
print("Breakpoints:", breaks_star)
print(f"Counts per regime (TRAIN): {cnt_train}  (total={sum(cnt_train)})")
print(f"Counts per regime (TEST) : {cnt_test}  (total={sum(cnt_test)})")

if any(c < MIN_PER_REGIME for c in cnt_train):
    print(f"[WARN] Some TRAIN regimes have < {MIN_PER_REGIME} samples. "
          f"Consider lowering K_FORCED or reducing MIN_PER_REGIME.")

# ============================================================
# STEP 3) Fit on TRAIN, evaluate on TEST
# ============================================================

models_star, breaks_star = fit_piecewise_linear(train, breaks_star)
test["yhat_opt"] = predict_piecewise(test["A"].values, models_star, breaks_star)

print("\n=== FORCED PIECEWISE MODEL (fit on TRAIN) ===")
print("K regimes:", K_FORCED)
print("Breakpoints:", breaks_star)
print("TEST MAE :", mae(test["y"].values, test["yhat_opt"].values))
print("TEST Bias:", bias(test["y"].values, test["yhat_opt"].values))
print("TEST RMSE:", rmse(test["y"].values, test["yhat_opt"].values))

# ============================================================
# STEP 4) Report coefficients per regime
# ============================================================

print("\nCoefficients per regime (y = alpha + beta*A):")
for i, m in enumerate(models_star, start=1):
    alpha = float(m.intercept_)
    beta  = float(m.coef_[0])

    if len(breaks_star) == 0:
        seg_txt = "all A"
    elif i == 1:
        seg_txt = f"A <= {breaks_star[0]:.2f}"
    elif i == len(models_star):
        seg_txt = f"A > {breaks_star[-1]:.2f}"
    else:
        seg_txt = f"{breaks_star[i-2]:.2f} < A <= {breaks_star[i-1]:.2f}"

    print(f"  Regime {i}: {seg_txt:>24} | alpha={alpha:.3f}, beta={beta:.6f}")

# ============================================================
# STEP 5) Diagnostics plots (TEST)
# ============================================================

res = test["yhat_opt"] - test["y"]

# # True vs Pred
# plt.figure(figsize=(6,6))
# plt.scatter(test["y"], test["yhat_opt"], alpha=0.6)
# mn = min(test["y"].min(), test["yhat_opt"].min())
# mx = max(test["y"].max(), test["yhat_opt"].max())
# plt.plot([mn, mx], [mn, mx], linestyle="--")
# plt.xlabel(f"True {USE_TARGET} apartment size y [m²]")
# plt.ylabel("Predicted ŷ [m²]")
# plt.title(f"TEST: True vs Predicted (Forced K={K_FORCED})")
# plt.tight_layout()
# plt.show()

# # Residual vs A
# plt.figure(figsize=(7,5))
# plt.scatter(test["A"], res, alpha=0.6)
# plt.axhline(0, linestyle="--")
# for b in breaks_star:
#     plt.axvline(b, linestyle="--")
# plt.xlabel("Footprint area A [m²]")
# plt.ylabel("Residual (ŷ − y) [m²]")
# plt.title(f"TEST: Residual vs A (Forced K={K_FORCED})")
# plt.tight_layout()
# plt.show()

# # Histogram
# plt.figure(figsize=(6,5))
# plt.hist(res, bins=25)
# plt.axvline(0, linestyle="--")
# plt.xlabel("Residual (ŷ − y) [m²]")
# plt.ylabel("Count")
# plt.title(f"TEST: Residual Distribution (Forced K={K_FORCED})")
# plt.tight_layout()
# plt.show()

# Piecewise curve on all data
A_plot = np.linspace(g["A"].min(), g["A"].max(), 600)
y_plot = predict_piecewise(A_plot, models_star, breaks_star)

plt.figure(figsize=(8,6))
plt.scatter(g["A"], g["y"], alpha=0.30)
plt.plot(A_plot, y_plot, linewidth=2)
for b in breaks_star:
    plt.axvline(b, linestyle="--")
plt.xlabel("Footprint area A [m²]")
plt.ylabel(f"{USE_TARGET} apartment size y [m²]")
plt.title(f"Forced Piecewise Fit (K={K_FORCED}, breakpoints from TRAIN quantiles)")
plt.tight_layout()
plt.show()
