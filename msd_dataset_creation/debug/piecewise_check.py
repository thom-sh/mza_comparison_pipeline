import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.linear_model import LinearRegression


# ============================================================
# USER SETTINGS
# ============================================================

TRAIN_CSV = Path(
    r"C:\WF\Thomas Sharon\Floorplan_Dataset\msd_dataset_creation\new_method\msd_apt_list_non_selected_200.csv"
)


OUTPUT_DIR = Path(
    r"C:\WF\Thomas Sharon\Floorplan_Dataset\msd_dataset_creation\new_method\model_selection_two_csv_no_refit"
)

TARGET_TYPE = "mean"
K_REGIMES = 6

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# THESIS PLOT STYLE
# ============================================================

LEGEND_EDGE_COLOR = "#bdc1c5"
POLYGON_EDGE_COLOR = "#777d84"
CORE_COLOR = "#5F666D"

DWELLING_COLORS = [
    "#DCEAF7",
    "#AFCBE3",
    "#7FA6C9",
    "#DBDBDB",
    "#BABABA",
    "#999999",
]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIX Two Text", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",

    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,

    "axes.titleweight": "normal",
    "axes.labelweight": "normal",

    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_area_dataset(input_csv, target_type="mean"):
    df = pd.read_csv(input_csv)

    apt_cols = [
        c for c in df.columns
        if c.startswith("apt_") and c.endswith("_area")
    ]

    if not apt_cols:
        raise ValueError(
            "No dwelling-area columns found. Expected columns such as apt_1_area, apt_2_area, ..."
        )

    if "building_id" not in df.columns:
        raise ValueError("Input CSV must contain a 'building_id' column.")

    if "footprint_area" not in df.columns:
        raise ValueError("Input CSV must contain a 'footprint_area' column.")

    for c in apt_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["footprint_area"] = pd.to_numeric(df["footprint_area"], errors="coerce")

    apt_values = df[apt_cols].to_numpy(dtype=float)
    valid = np.where((apt_values > 0) & np.isfinite(apt_values), apt_values, np.nan)

    df["n_valid_dwellings"] = np.sum(np.isfinite(valid), axis=1)

    if target_type == "mean":
        df["y_target"] = np.nanmean(valid, axis=1)
    elif target_type == "median":
        df["y_target"] = np.nanmedian(valid, axis=1)
    else:
        raise ValueError("target_type must be either 'mean' or 'median'.")

    df_clean = df.dropna(subset=["building_id", "footprint_area", "y_target"]).copy()
    df_clean = df_clean[df_clean["footprint_area"] > 0]
    df_clean = df_clean[df_clean["n_valid_dwellings"] >= 2]

    g = df_clean[
        ["building_id", "footprint_area", "y_target", "n_valid_dwellings"]
    ].copy()

    g = g.rename(columns={
        "footprint_area": "A",
        "y_target": "y",
    })

    g = g.reset_index(drop=True)

    return g


# ============================================================
# PIECEWISE LINEAR MODEL
# ============================================================

def make_quantile_breaks(A_values, K):
    """
    K regimes means K - 1 quantile breakpoints.
    """
    qs = np.linspace(0, 1, K + 1)[1:-1]
    return tuple(np.quantile(A_values, qs))


def fit_piecewise_linear(df_train, breaks):
    """
    Fit one linear regression model per footprint-area regime.
    """
    breaks = tuple(sorted(breaks))
    models = []

    for r in range(len(breaks) + 1):
        if r == 0:
            segment = df_train[df_train["A"] <= breaks[0]]
        elif r == len(breaks):
            segment = df_train[df_train["A"] > breaks[-1]]
        else:
            segment = df_train[
                (df_train["A"] > breaks[r - 1]) &
                (df_train["A"] <= breaks[r])
            ]

        if len(segment) < 2:
            raise ValueError(
                f"Regime {r + 1} has too few samples: {len(segment)}"
            )

        model = LinearRegression()
        model.fit(segment[["A"]], segment["y"])
        models.append(model)

    return models, breaks


def predict_piecewise(A_values, models, breaks):
    A_values = np.asarray(A_values, dtype=float)
    yhat = np.empty_like(A_values, dtype=float)

    masks = [A_values <= breaks[0]]

    for i in range(1, len(breaks)):
        masks.append((A_values > breaks[i - 1]) & (A_values <= breaks[i]))

    masks.append(A_values > breaks[-1])

    for r, mask in enumerate(masks):
        if mask.any():
            yhat[mask] = models[r].predict(
                pd.DataFrame({"A": A_values[mask]})
            )

    return yhat


# ============================================================
# METRICS
# ============================================================

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_pred - y_true)))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def bias(y_true, y_pred):
    return float(np.mean(y_pred - y_true))


def r2_score_manual(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return np.nan

    return float(1 - ss_res / ss_tot)


# ============================================================
# EXPORT EQUATIONS
# ============================================================

def export_k6_equations(models, breaks, output_dir):
    rows = []

    for i, model in enumerate(models, start=1):
        alpha = float(model.intercept_)
        beta = float(model.coef_[0])

        if i == 1:
            lower = -np.inf
            upper = breaks[0]
            interval = f"A_fp <= {upper:.4f}"
        elif i == len(models):
            lower = breaks[-1]
            upper = np.inf
            interval = f"A_fp > {lower:.4f}"
        else:
            lower = breaks[i - 2]
            upper = breaks[i - 1]
            interval = f"{lower:.4f} < A_fp <= {upper:.4f}"

        equation = f"A_dw_hat = {alpha:.4f} + {beta:.6f} * A_fp"

        rows.append({
            "model": "piecewise_linear_K6",
            "regime": i,
            "lower_bound_A_fp": lower,
            "upper_bound_A_fp": upper,
            "interval": interval,
            "alpha_intercept": alpha,
            "beta_slope": beta,
            "equation": equation,
        })

    equations_df = pd.DataFrame(rows)

    equations_path = output_dir / "piecewise_linear_K6_equations_training_only.csv"
    equations_df.to_csv(equations_path, index=False)

    return equations_df, equations_path


# ============================================================
# PLOT
# ============================================================

def style_legend(legend):
    if legend is not None:
        legend.get_frame().set_edgecolor(LEGEND_EDGE_COLOR)
        legend.get_frame().set_linewidth(0.8)
        legend.get_frame().set_facecolor("white")


def plot_k6_model_fit(g, models, breaks, output_dir):
    fig, ax = plt.subplots(figsize=(6.2, 4.2))

    ax.scatter(
        g["A"],
        g["y"],
        s=18,
        alpha=0.65,
        facecolor=DWELLING_COLORS[1],
        edgecolor=POLYGON_EDGE_COLOR,
        linewidth=0.35,
        label="Training cases",
    )

    A_plot = np.linspace(g["A"].min(), g["A"].max(), 600)
    y_plot = predict_piecewise(A_plot, models, breaks)

    ax.plot(
        A_plot,
        y_plot,
        color=CORE_COLOR,
        linewidth=1.8,
        label="Six-regime piecewise model",
    )

    for i, b in enumerate(breaks):
        ax.axvline(
            b,
            color=POLYGON_EDGE_COLOR,
            linestyle="--",
            linewidth=0.8,
            alpha=0.75,
            label="Regime breakpoint" if i == 0 else None,
        )

    ax.set_xlabel(r"External footprint area $A_\mathrm{fp}$ [m²]")
    ax.set_ylabel(r"Mean dwelling area $A_\mathrm{dw}$ [m²]")
    ax.set_title("Six-regime mean dwelling-area estimation model")

    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)

    legend = ax.legend(loc="best", frameon=True)
    style_legend(legend)

    fig.tight_layout()

    pdf_path = output_dir / "piecewise_linear_K6_model_fit_training_only.pdf"
    png_path = output_dir / "piecewise_linear_K6_model_fit_training_only.png"

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")

    print(f"Saved plot: {pdf_path}")
    print(f"Saved plot: {png_path}")

    plt.show()
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():
    print("Loading and cleaning training dataset...")
    g = prepare_area_dataset(TRAIN_CSV, target_type=TARGET_TYPE)

    print(f"Cleaned training cases: {len(g)}")

    if len(g) < K_REGIMES * 2:
        raise ValueError(
            f"Too few cases for {K_REGIMES} regimes. "
            f"Only {len(g)} cleaned cases available."
        )

    cleaned_path = OUTPUT_DIR / "cleaned_training_dataset_used_for_K6.csv"
    g.to_csv(cleaned_path, index=False)
    print(f"Cleaned training dataset saved to: {cleaned_path}")

    print("\nCreating six-regime quantile breakpoints from training dataset only...")
    breaks = make_quantile_breaks(g["A"].values, K_REGIMES)

    print("Breakpoints:")
    for b in breaks:
        print(f"  {b:.4f}")

    print("\nFitting six-regime piecewise linear model...")
    models, breaks = fit_piecewise_linear(g, breaks)

    print("\nExporting equations...")
    equations_df, equations_path = export_k6_equations(models, breaks, OUTPUT_DIR)

    print("\n=== SIX-REGIME EQUATIONS ===")
    print(equations_df.to_string(index=False))
    print(f"\nEquations saved to: {equations_path}")

    print("\nCalculating training-set fitted values for cross-check...")
    g_predictions = g.copy()
    g_predictions["yhat_K6"] = predict_piecewise(g_predictions["A"].values, models, breaks)
    g_predictions["residual_K6"] = g_predictions["yhat_K6"] - g_predictions["y"]

    predictions_path = OUTPUT_DIR / "piecewise_linear_K6_training_predictions.csv"
    g_predictions.to_csv(predictions_path, index=False)

    metrics = {
        "model": "piecewise_linear_K6",
        "training_cases": len(g),
        "MAE_m2": mae(g_predictions["y"].values, g_predictions["yhat_K6"].values),
        "RMSE_m2": rmse(g_predictions["y"].values, g_predictions["yhat_K6"].values),
        "Bias_m2": bias(g_predictions["y"].values, g_predictions["yhat_K6"].values),
        "R2": r2_score_manual(g_predictions["y"].values, g_predictions["yhat_K6"].values),
    }

    metrics_df = pd.DataFrame([metrics])
    metrics_path = OUTPUT_DIR / "piecewise_linear_K6_training_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    print("\n=== TRAINING-SET METRICS FOR CROSS-CHECK ===")
    print(metrics_df.to_string(index=False))
    print(f"\nTraining predictions saved to: {predictions_path}")
    print(f"Training metrics saved to: {metrics_path}")

    print("\nCreating plot...")
    plot_k6_model_fit(g, models, breaks, OUTPUT_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()