import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression

# ============================================================
# 0) LOAD + BUILD BUILDING-LEVEL DATASET (choose target)
# ============================================================

import pandas as pd
import numpy as np

# ---- Load CSV ----
csv_path = r"C:\WF\Thomas Sharon\Floorplan_Dataset\msd_dataset_creation\msd_apt_list.csv"
df = pd.read_csv(csv_path)

# ---- Identify apartment area columns ----
apt_cols = [c for c in df.columns if c.startswith("apt_") and c.endswith("_area")]

# ---- Convert to numeric ----
for c in apt_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["footprint_area"] = pd.to_numeric(df["footprint_area"], errors="coerce")

# ---- Extract valid apartment areas ----
apt_values = df[apt_cols].to_numpy(dtype=float)
valid = np.where((apt_values > 0) & np.isfinite(apt_values), apt_values, np.nan)

df["n_valid_apts"] = np.sum(np.isfinite(valid), axis=1)

# ---- Choose target definition ----
USE_TARGET = "median"   # options: "mean" or "median"

if USE_TARGET == "mean":
    df["y_target"] = np.nanmean(valid, axis=1)
elif USE_TARGET == "median":
    df["y_target"] = np.nanmedian(valid, axis=1)
else:
    raise ValueError("USE_TARGET must be 'mean' or 'median'")

# ---- Quality filtering ----
df_clean = df.dropna(subset=["building_id", "footprint_area", "y_target"]).copy()
df_clean = df_clean[df_clean["footprint_area"] > 0]
df_clean = df_clean[df_clean["n_valid_apts"] >= 2]

# ---- Final building-level table ----
g = df_clean[["building_id", "footprint_area", "y_target", "n_valid_apts"]].copy()
g = g.rename(columns={
    "footprint_area": "A",
    "y_target": "y"
})

g = g.reset_index(drop=True)

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
    test_size=0.5,
    random_state=42,
    stratify=g["A_bin"]
)

train = g.loc[train_idx].copy().reset_index(drop=True)
test  = g.loc[test_idx].copy().reset_index(drop=True)

print("OUTER TRAIN:", len(train), "OUTER TEST:", len(test))

# ============================================================
# Piecewise helpers (K regimes, K-1 breakpoints)
# ============================================================

def _assign_regime(A, breaks):
    # breaks sorted, regimes: (-inf,b1], (b1,b2], ..., (b_{K-1}, +inf)
    for i, b in enumerate(breaks):
        if A <= b:
            return i
    return len(breaks)

def fit_piecewise_linear(df_, breaks):
    """Fit separate LinearRegression per regime. returns list of models + breaks."""
    breaks = tuple(sorted(breaks))
    models = []
    for r in range(len(breaks) + 1):
        if r == 0:
            seg = df_[df_["A"] <= breaks[0]] if breaks else df_
        elif r == len(breaks):
            seg = df_[df_["A"] > breaks[-1]]
        else:
            seg = df_[(df_["A"] > breaks[r-1]) & (df_["A"] <= breaks[r])]

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

    # masks for each segment
    masks = []
    masks.append(A_values <= breaks[0])
    for i in range(1, len(breaks)):
        masks.append((A_values > breaks[i-1]) & (A_values <= breaks[i]))
    masks.append(A_values > breaks[-1])

    for r, mask in enumerate(masks):
        if mask.any():
            yhat[mask] = models[r].predict(pd.DataFrame({"A": A_values[mask]}))
    return yhat

def valid_breaks(df_, breaks, min_per_regime=30):
    """Check each regime has at least min_per_regime rows."""
    breaks = tuple(sorted(breaks))
    counts = []
    if len(breaks) == 0:
        return len(df_) >= min_per_regime

    counts.append((df_["A"] <= breaks[0]).sum())
    for i in range(1, len(breaks)):
        counts.append(((df_["A"] > breaks[i-1]) & (df_["A"] <= breaks[i])).sum())
    counts.append((df_["A"] > breaks[-1]).sum())
    return all(c >= min_per_regime for c in counts)

# ============================================================
# STEP 2) INNER CV search on TRAIN ONLY:
#         choose best K regimes and optimized breakpoints
# ============================================================

# candidate breakpoint grid from TRAIN (quantiles -> stable search grid)
# Increase n_grid for finer search (cost increases)
n_grid = 19
q = np.linspace(0.05, 0.95, n_grid)
cand = np.unique(np.quantile(train["A"].values, q))
cand = np.sort(cand)

K_max = 5           # try K=1..4 regimes
min_per_regime = 30 # avoid tiny segments
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=123)

def cv_score_for_breaks(df_train, breaks):
    fold_mae = []
    for tr_i, va_i in kf.split(df_train):
        tr = df_train.iloc[tr_i]
        va = df_train.iloc[va_i]

        # Ensure regime counts are OK in this fold too (optional but recommended)
        if not valid_breaks(tr, breaks, min_per_regime=min_per_regime):
            return np.inf

        models, br = fit_piecewise_linear(tr, breaks)
        yhat = predict_piecewise(va["A"].values, models, br)
        fold_mae.append(mae(va["y"].values, yhat))
    return float(np.mean(fold_mae))

results = []

# K=1 has no breaks
print("\n[SEARCH] Starting CV search...")
best_overall = None

for K in range(1, K_max + 1):
    n_breaks = K - 1
    if n_breaks == 0:
        br_list = [tuple()]
    else:
        br_list = list(itertools.combinations(cand, n_breaks))

    best_K = {"K": K, "breaks": None, "cv_mae": np.inf}
    for breaks in br_list:
        # check full TRAIN has enough per regime
        if not valid_breaks(train, breaks, min_per_regime=min_per_regime):
            continue

        score = cv_score_for_breaks(train, breaks)
        if score < best_K["cv_mae"]:
            best_K = {"K": K, "breaks": tuple(sorted(breaks)), "cv_mae": score}

    results.append(best_K)
    print(f"[SEARCH] K={K} best CV-MAE={best_K['cv_mae']:.3f} breaks={best_K['breaks']}")

# choose best K by CV-MAE; tie-break -> smaller K
results_sorted = sorted(results, key=lambda d: (d["cv_mae"], d["K"]))
chosen = results_sorted[0]
print("\n[CHOSEN] ", chosen)

# ============================================================
# STEP 3) Refit chosen model on ALL TRAIN, evaluate on TEST
# ============================================================

K_star = chosen["K"]
breaks_star = chosen["breaks"] if chosen["breaks"] is not None else tuple()

models_star, breaks_star = fit_piecewise_linear(train, breaks_star)
test["yhat_opt"] = predict_piecewise(test["A"].values, models_star, breaks_star)

print("\n=== OPTIMIZED PIECEWISE MODEL (refit on TRAIN) ===")
print("K regimes:", K_star)
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
    print(f"  Regime {i}: {seg_txt:>20} | alpha={alpha:.3f}, beta={beta:.6f}")

# ============================================================
# STEP 5) Diagnostics plots (TEST)
# ============================================================

res = test["yhat_opt"] - test["y"]

# True vs Pred
plt.figure(figsize=(6,6))
plt.scatter(test["y"], test["yhat_opt"], alpha=0.6)
mn = min(test["y"].min(), test["yhat_opt"].min())
mx = max(test["y"].max(), test["yhat_opt"].max())
plt.plot([mn, mx], [mn, mx], linestyle="--")
plt.xlabel(f"True {USE_TARGET} apartment size y [m²]")
plt.ylabel(f"Predicted ŷ [m²]")
plt.title(f"TEST: True vs Predicted (Optimized K={K_star})")
plt.tight_layout()
plt.show()

# Residual vs A
plt.figure(figsize=(7,5))
plt.scatter(test["A"], res, alpha=0.6)
plt.axhline(0, linestyle="--")
for b in breaks_star:
    plt.axvline(b, linestyle="--")
plt.xlabel("Footprint area A [m²]")
plt.ylabel("Residual (ŷ − y) [m²]")
plt.title(f"TEST: Residual vs A (Optimized K={K_star})")
plt.tight_layout()
plt.show()

# Histogram
plt.figure(figsize=(6,5))
plt.hist(res, bins=25)
plt.axvline(0, linestyle="--")
plt.xlabel("Residual (ŷ − y) [m²]")
plt.ylabel("Count")
plt.title(f"TEST: Residual Distribution (Optimized K={K_star})")
plt.tight_layout()
plt.show()

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
plt.title(f"Optimized Piecewise Fit (K={K_star}, TRAIN-chosen)")
plt.tight_layout()
plt.show()
