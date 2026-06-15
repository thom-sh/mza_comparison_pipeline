import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


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


def evaluate_model(name, y_true, y_pred):
    return {
        "model": name,
        "MAE_m2": mae(y_true, y_pred),
        "RMSE_m2": rmse(y_true, y_pred),
        "Bias_m2": bias(y_true, y_pred),
        "R2": r2_score_manual(y_true, y_pred),
    }


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_area_dataset(input_csv, target_type="mean"):
    """
    Load one CSV and create a cleaned building-level dataset.

    Expected columns:
    - building_id
    - footprint_area
    - apt_1_area, apt_2_area, ... or similar columns ending in _area
    """

    df = pd.read_csv(input_csv)

    apt_cols = [
        c for c in df.columns
        if c.startswith("apt_") and c.endswith("_area")
    ]

    if not apt_cols:
        raise ValueError(
            f"No apartment/dwelling area columns found in {input_csv}. "
            "Expected columns like apt_1_area, apt_2_area, ..."
        )

    if "building_id" not in df.columns:
        raise ValueError(f"Input CSV must contain a 'building_id' column: {input_csv}")

    if "footprint_area" not in df.columns:
        raise ValueError(f"Input CSV must contain a 'footprint_area' column: {input_csv}")

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

    g = df_clean[[
        "building_id",
        "footprint_area",
        "y_target",
        "n_valid_dwellings",
    ]].copy()

    g = g.rename(columns={
        "footprint_area": "A",
        "y_target": "y",
    })

    g = g.reset_index(drop=True)

    return g


# ============================================================
# PIECEWISE LINEAR MODEL HELPERS
# ============================================================

def make_quantile_breaks(A_values, K):
    """
    K regimes -> K - 1 quantile breakpoints.
    Breakpoints are always created from the training CSV only.
    """
    qs = np.linspace(0, 1, K + 1)[1:-1]
    return tuple(np.quantile(A_values, qs))


def fit_piecewise_linear(df_train, breaks):
    """
    Fit one LinearRegression model per footprint-area regime.
    """
    breaks = tuple(sorted(breaks))
    models = []

    for r in range(len(breaks) + 1):
        if len(breaks) == 0:
            seg = df_train
        elif r == 0:
            seg = df_train[df_train["A"] <= breaks[0]]
        elif r == len(breaks):
            seg = df_train[df_train["A"] > breaks[-1]]
        else:
            seg = df_train[
                (df_train["A"] > breaks[r - 1]) &
                (df_train["A"] <= breaks[r])
            ]

        if len(seg) < 2:
            raise ValueError(f"Regime {r + 1} has too few samples: {len(seg)}")

        model = LinearRegression()
        model.fit(seg[["A"]], seg["y"])
        models.append(model)

    return models, breaks


def predict_piecewise(A_values, models, breaks):
    """
    Predict using the correct linear model for each footprint-area regime.
    """
    A_values = np.asarray(A_values, dtype=float)
    yhat = np.empty_like(A_values, dtype=float)

    if len(breaks) == 0:
        yhat[:] = models[0].predict(pd.DataFrame({"A": A_values}))
        return yhat

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
# PLOTTING HELPERS
# ============================================================

def save_figure(fig, output_dir, filename_base):
    pdf_path = output_dir / f"{filename_base}.pdf"
    png_path = output_dir / f"{filename_base}.png"

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")

    print(f"Saved plot: {pdf_path}")
    print(f"Saved plot: {png_path}")


def style_legend(legend):
    if legend is not None:
        legend.get_frame().set_edgecolor(LEGEND_EDGE_COLOR)
        legend.get_frame().set_linewidth(0.8)
        legend.get_frame().set_facecolor("white")


def pretty_model_label(model_name):
    if model_name == "B0_constant_training_mean":
        return "Constant mean baseline"
    if model_name == "B1_simple_linear_regression":
        return "Simple linear regression"
    if model_name.startswith("M2_piecewise_linear_K"):
        K = model_name.split("K")[-1]
        return f"Piecewise linear, {K} regimes"
    return model_name


def plot_model_comparison(results_df, output_dir):
    df_plot = results_df.sort_values("MAE_m2", ascending=True).copy()
    df_plot["label"] = df_plot["model"].apply(pretty_model_label)

    fig, ax = plt.subplots(figsize=(7.0, 3.6))

    colors = []
    for i, model in enumerate(df_plot["model"]):
        if i == 0:
            colors.append(CORE_COLOR)
        elif model.startswith("M2"):
            colors.append(DWELLING_COLORS[2])
        elif model.startswith("B1"):
            colors.append(DWELLING_COLORS[1])
        else:
            colors.append(DWELLING_COLORS[3])

    bars = ax.bar(
        range(len(df_plot)),
        df_plot["MAE_m2"],
        color=colors,
        edgecolor=POLYGON_EDGE_COLOR,
        linewidth=0.7,
    )

    ax.set_ylabel("Mean absolute error [m²]")
    ax.set_xlabel("Candidate model")
    ax.set_title("Model comparison for mean dwelling-area estimation")
    ax.set_xticks(range(len(df_plot)))
    ax.set_xticklabels(df_plot["label"], rotation=35, ha="right")

    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, df_plot["MAE_m2"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    fig.tight_layout()
    save_figure(fig, output_dir, "plot_01_model_comparison_mae")
    plt.show()
    plt.close(fig)


def plot_final_fit(train_df, test_50_df, final_model_name, final_model_object, final_breaks, output_dir):
    """
    Final selected model trained only on the 200 training cases.
    No refit is performed.
    """

    fig, ax = plt.subplots(figsize=(6.2, 4.2))

    ax.scatter(
        train_df["A"],
        train_df["y"],
        s=18,
        alpha=0.65,
        facecolor=DWELLING_COLORS[1],
        edgecolor=POLYGON_EDGE_COLOR,
        linewidth=0.35,
        label="Training cases",
    )

    ax.scatter(
        test_50_df["A"],
        test_50_df["y"],
        s=22,
        alpha=0.75,
        facecolor="white",
        edgecolor=CORE_COLOR,
        linewidth=0.7,
        label="Hold-out test cases",
    )

    A_min = min(train_df["A"].min(), test_50_df["A"].min())
    A_max = max(train_df["A"].max(), test_50_df["A"].max())
    A_plot = np.linspace(A_min, A_max, 600)

    if final_model_name.startswith("M2"):
        y_plot = predict_piecewise(A_plot, final_model_object, final_breaks)

        ax.plot(
            A_plot,
            y_plot,
            color=CORE_COLOR,
            linewidth=1.8,
            label="Selected model",
        )

        for i, b in enumerate(final_breaks):
            ax.axvline(
                b,
                color=POLYGON_EDGE_COLOR,
                linestyle="--",
                linewidth=0.8,
                alpha=0.75,
                label="Regime breakpoint" if i == 0 else None,
            )

    elif final_model_name.startswith("B1"):
        y_plot = final_model_object.predict(pd.DataFrame({"A": A_plot}))

        ax.plot(
            A_plot,
            y_plot,
            color=CORE_COLOR,
            linewidth=1.8,
            label="Selected linear model",
        )

    elif final_model_name.startswith("B0"):
        ax.axhline(
            final_model_object,
            color=CORE_COLOR,
            linewidth=1.8,
            label="Selected constant baseline",
        )

    ax.set_xlabel(r"External footprint area $A_\mathrm{fp}$ [m²]")
    ax.set_ylabel(r"Mean dwelling area $A_\mathrm{dw}$ [m²]")
    ax.set_title("Selected mean dwelling-area estimation model")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)

    legend = ax.legend(loc="best", frameon=True)
    style_legend(legend)

    fig.tight_layout()
    save_figure(fig, output_dir, "plot_02_selected_model_fit_no_refit")
    plt.show()
    plt.close(fig)


def plot_predicted_vs_observed(test_best, output_dir):
    y = test_best["y"].values
    yhat = test_best["yhat_best"].values

    mn = min(np.min(y), np.min(yhat))
    mx = max(np.max(y), np.max(yhat))

    fig, ax = plt.subplots(figsize=(4.8, 4.8))

    ax.scatter(
        y,
        yhat,
        s=22,
        alpha=0.70,
        facecolor=DWELLING_COLORS[1],
        edgecolor=POLYGON_EDGE_COLOR,
        linewidth=0.35,
        label="Hold-out test cases",
    )

    ax.plot(
        [mn, mx],
        [mn, mx],
        color=CORE_COLOR,
        linestyle="--",
        linewidth=1.1,
        label="Perfect prediction",
    )

    ax.set_xlabel("Observed mean dwelling area [m²]")
    ax.set_ylabel("Predicted mean dwelling area [m²]")
    ax.set_title("Predicted vs observed mean dwelling area")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)

    legend = ax.legend(loc="best", frameon=True)
    style_legend(legend)

    fig.tight_layout()
    save_figure(fig, output_dir, "plot_03_predicted_vs_observed_holdout_50")
    plt.show()
    plt.close(fig)


def plot_residual_vs_footprint(test_best, output_dir):
    fig, ax = plt.subplots(figsize=(6.2, 3.8))

    ax.scatter(
        test_best["A"],
        test_best["residual_best"],
        s=22,
        alpha=0.70,
        facecolor=DWELLING_COLORS[1],
        edgecolor=POLYGON_EDGE_COLOR,
        linewidth=0.35,
        label="Hold-out test cases",
    )

    ax.axhline(
        0,
        color=CORE_COLOR,
        linestyle="--",
        linewidth=1.1,
        label="Zero residual",
    )

    ax.set_xlabel(r"External footprint area $A_\mathrm{fp}$ [m²]")
    ax.set_ylabel("Residual [m²]")
    ax.set_title("Residuals across footprint area")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)

    legend = ax.legend(loc="best", frameon=True)
    style_legend(legend)

    fig.tight_layout()
    save_figure(fig, output_dir, "plot_04_residual_vs_footprint_holdout_50")
    plt.show()
    plt.close(fig)


# ============================================================
# EQUATION EXPORT
# ============================================================

def export_selected_model_parameters(
    selected_model_name,
    selected_model_object,
    selected_breaks,
    target_type,
    output_dir,
):
    """
    Export the actual equations from the selected model.
    This uses the model trained on the 200 training cases.
    It does not refit on the test data.
    """

    params = []

    if selected_model_name.startswith("M2_piecewise_linear"):
        for i, model in enumerate(selected_model_object, start=1):
            alpha = float(model.intercept_)
            beta = float(model.coef_[0])

            if i == 1:
                lower = -np.inf
                upper = selected_breaks[0]
                interval = f"A_fp <= {upper:.4f}"
            elif i == len(selected_model_object):
                lower = selected_breaks[-1]
                upper = np.inf
                interval = f"A_fp > {lower:.4f}"
            else:
                lower = selected_breaks[i - 2]
                upper = selected_breaks[i - 1]
                interval = f"{lower:.4f} < A_fp <= {upper:.4f}"

            equation = f"A_dw_hat = {alpha:.4f} + {beta:.6f} * A_fp"

            params.append({
                "selected_model": selected_model_name,
                "target_type": target_type,
                "regime": i,
                "lower_bound_A_fp": lower,
                "upper_bound_A_fp": upper,
                "interval": interval,
                "alpha_intercept": alpha,
                "beta_slope": beta,
                "equation": equation,
            })

    elif selected_model_name == "B1_simple_linear_regression":
        alpha = float(selected_model_object.intercept_)
        beta = float(selected_model_object.coef_[0])
        equation = f"A_dw_hat = {alpha:.4f} + {beta:.6f} * A_fp"

        params.append({
            "selected_model": selected_model_name,
            "target_type": target_type,
            "regime": 1,
            "lower_bound_A_fp": -np.inf,
            "upper_bound_A_fp": np.inf,
            "interval": "all A_fp",
            "alpha_intercept": alpha,
            "beta_slope": beta,
            "equation": equation,
        })

    elif selected_model_name == "B0_constant_training_mean":
        value = float(selected_model_object)
        equation = f"A_dw_hat = {value:.4f}"

        params.append({
            "selected_model": selected_model_name,
            "target_type": target_type,
            "regime": 1,
            "lower_bound_A_fp": -np.inf,
            "upper_bound_A_fp": np.inf,
            "interval": "all A_fp",
            "alpha_intercept": value,
            "beta_slope": 0.0,
            "equation": equation,
        })

    else:
        raise ValueError(f"Unknown selected model: {selected_model_name}")

    params_df = pd.DataFrame(params)

    params_path = output_dir / "selected_model_equations_no_refit.csv"
    params_df.to_csv(params_path, index=False)

    return params_df, params_path


# ============================================================
# MAIN SCRIPT
# ============================================================

def main():
    # ------------------------------------------------------------
    # USER SETTINGS
    # ------------------------------------------------------------

    TRAIN_CSV = Path(
        r"C:\WF\Thomas Sharon\Floorplan_Dataset\msd_dataset_creation\new_method\msd_apt_list_non_selected_200.csv"
    )

    TEST_SOURCE_CSV = Path(
        r"C:\WF\Thomas Sharon\Floorplan_Dataset\msd_dataset_creation\new_method\msd_apt_list_selected_200.csv"
    )

    OUTPUT_DIR = Path(
        r"C:\WF\Thomas Sharon\Floorplan_Dataset\msd_dataset_creation\new_method\model_selection_two_csv_no_refit"
    )

    TARGET_TYPE = "mean"
    TEST_SAMPLE_SIZE = 50
    RANDOM_STATE = 42
    K_VALUES = [2, 3, 4, 5, 6]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # LOAD AND CLEAN TWO SEPARATE DATASETS
    # ------------------------------------------------------------

    train_df = prepare_area_dataset(TRAIN_CSV, target_type=TARGET_TYPE)
    test_source_df = prepare_area_dataset(TEST_SOURCE_CSV, target_type=TARGET_TYPE)

    print(f"Training CSV cleaned cases: {len(train_df)}")
    print(f"Testing-source CSV cleaned cases: {len(test_source_df)}")

    if len(train_df) < 2:
        raise ValueError("Training dataset has too few cleaned cases.")

    if len(test_source_df) < TEST_SAMPLE_SIZE:
        raise ValueError(
            f"Testing-source dataset has only {len(test_source_df)} cleaned cases, "
            f"but TEST_SAMPLE_SIZE is {TEST_SAMPLE_SIZE}."
        )

    train_df.to_csv(OUTPUT_DIR / "cleaned_training_dataset.csv", index=False)
    test_source_df.to_csv(OUTPUT_DIR / "cleaned_testing_source_dataset.csv", index=False)

    # ------------------------------------------------------------
    # SELECT 50 HOLD-OUT TEST CASES FROM THE SECOND CSV
    # Stratified by footprint-area quartiles
    # ------------------------------------------------------------

    test_source_df = test_source_df.copy()
    test_source_df["A_bin"] = pd.qcut(
        test_source_df["A"],
        q=4,
        labels=False,
        duplicates="drop",
    )

    holdout_50_idx, remaining_idx = train_test_split(
        test_source_df.index,
        train_size=TEST_SAMPLE_SIZE,
        random_state=RANDOM_STATE,
        stratify=test_source_df["A_bin"],
    )

    holdout_50 = test_source_df.loc[holdout_50_idx].copy().reset_index(drop=True)
    remaining_test_source = test_source_df.loc[remaining_idx].copy().reset_index(drop=True)

    holdout_50 = holdout_50.drop(columns=["A_bin"])
    remaining_test_source = remaining_test_source.drop(columns=["A_bin"])

    holdout_50.to_csv(OUTPUT_DIR / "holdout_50_cases_used_for_model_selection.csv", index=False)
    remaining_test_source.to_csv(OUTPUT_DIR / "remaining_test_source_cases_not_used_for_selection.csv", index=False)

    print(f"Hold-out test cases used for model selection: {len(holdout_50)}")
    print(f"Remaining second-CSV cases not used for model selection: {len(remaining_test_source)}")

    # ------------------------------------------------------------
    # TRAIN CANDIDATE MODELS ON TRAINING CSV ONLY
    # EVALUATE ON 50 HOLD-OUT CASES ONLY
    # ------------------------------------------------------------

    results = []
    prediction_tables = []
    fitted_models = {}

    # -------------------------
    # B0: Constant baseline
    # -------------------------
    b0_value = float(train_df["y"].mean())
    yhat_b0 = np.full(len(holdout_50), b0_value)

    results.append(
        evaluate_model(
            "B0_constant_training_mean",
            holdout_50["y"].values,
            yhat_b0,
        )
    )

    prediction_tables.append(
        holdout_50[["building_id", "A", "y"]]
        .assign(yhat=yhat_b0, model="B0_constant_training_mean")
    )

    fitted_models["B0_constant_training_mean"] = {
        "object": b0_value,
        "breaks": tuple(),
    }

    # -------------------------
    # B1: Simple linear model
    # -------------------------
    b1 = LinearRegression()
    b1.fit(train_df[["A"]], train_df["y"])
    yhat_b1 = b1.predict(holdout_50[["A"]])

    results.append(
        evaluate_model(
            "B1_simple_linear_regression",
            holdout_50["y"].values,
            yhat_b1,
        )
    )

    prediction_tables.append(
        holdout_50[["building_id", "A", "y"]]
        .assign(yhat=yhat_b1, model="B1_simple_linear_regression")
    )

    fitted_models["B1_simple_linear_regression"] = {
        "object": b1,
        "breaks": tuple(),
    }

    # -------------------------
    # M2: Piecewise linear models
    # -------------------------
    for K in K_VALUES:
        breaks = make_quantile_breaks(train_df["A"].values, K)

        try:
            models, breaks = fit_piecewise_linear(train_df, breaks)
            yhat = predict_piecewise(holdout_50["A"].values, models, breaks)

            model_name = f"M2_piecewise_linear_K{K}"

            results.append(
                evaluate_model(
                    model_name,
                    holdout_50["y"].values,
                    yhat,
                )
            )

            prediction_tables.append(
                holdout_50[["building_id", "A", "y"]]
                .assign(yhat=yhat, model=model_name)
            )

            fitted_models[model_name] = {
                "object": models,
                "breaks": breaks,
            }

        except ValueError as e:
            print(f"Skipping piecewise model with K={K}: {e}")

    # ------------------------------------------------------------
    # MODEL SELECTION BASED ON HOLD-OUT 50 CASES
    # ------------------------------------------------------------

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(["MAE_m2", "RMSE_m2"]).reset_index(drop=True)

    best_model_name = str(results_df.iloc[0]["model"])
    selected_model_object = fitted_models[best_model_name]["object"]
    selected_breaks = fitted_models[best_model_name]["breaks"]

    print("\n=== MODEL COMPARISON RESULTS ON 50 HOLD-OUT CASES ===")
    print(results_df.to_string(index=False))

    print("\nSelected model based on lowest hold-out MAE:")
    print(best_model_name)

    results_df.to_csv(OUTPUT_DIR / "model_comparison_results_holdout_50.csv", index=False)

    all_predictions = pd.concat(prediction_tables, ignore_index=True)
    all_predictions.to_csv(OUTPUT_DIR / "all_model_predictions_holdout_50.csv", index=False)

    test_best = all_predictions[all_predictions["model"] == best_model_name].copy()
    test_best = test_best.rename(columns={"yhat": "yhat_best"})
    test_best["residual_best"] = test_best["yhat_best"] - test_best["y"]
    test_best.to_csv(OUTPUT_DIR / "best_model_predictions_holdout_50.csv", index=False)

    # ------------------------------------------------------------
    # EXPORT SELECTED MODEL EQUATIONS
    # IMPORTANT: NO FINAL REFIT
    # ------------------------------------------------------------

    selected_params_df, params_path = export_selected_model_parameters(
        selected_model_name=best_model_name,
        selected_model_object=selected_model_object,
        selected_breaks=selected_breaks,
        target_type=TARGET_TYPE,
        output_dir=OUTPUT_DIR,
    )

    print("\n=== SELECTED MODEL EQUATIONS, TRAINED ON TRAINING CSV ONLY ===")
    print(selected_params_df.to_string(index=False))
    print(f"\nSelected model equations saved to: {params_path}")

    # ------------------------------------------------------------
    # OPTIONAL: APPLY SELECTED MODEL TO ALL 200 CASES IN SECOND CSV
    # This is not used for model selection.
    # ------------------------------------------------------------

    all_test_predictions = test_source_df.drop(columns=["A_bin"]).copy()

    if best_model_name.startswith("M2"):
        all_test_predictions["yhat_selected_model"] = predict_piecewise(
            all_test_predictions["A"].values,
            selected_model_object,
            selected_breaks,
        )
    elif best_model_name.startswith("B1"):
        all_test_predictions["yhat_selected_model"] = selected_model_object.predict(
            all_test_predictions[["A"]]
        )
    elif best_model_name.startswith("B0"):
        all_test_predictions["yhat_selected_model"] = float(selected_model_object)

    all_test_predictions["residual_selected_model"] = (
        all_test_predictions["yhat_selected_model"] - all_test_predictions["y"]
    )

    all_test_predictions.to_csv(
        OUTPUT_DIR / "selected_model_predictions_all_second_csv_cases_not_for_selection.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # SUMMARY FILE
    # ------------------------------------------------------------

    summary_path = OUTPUT_DIR / "selected_model_summary_no_refit.txt"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Mean dwelling-area estimation using two separate CSV files\n")
        f.write("=========================================================\n\n")
        f.write(f"Training CSV: {TRAIN_CSV}\n")
        f.write(f"Testing-source CSV: {TEST_SOURCE_CSV}\n")
        f.write(f"Cleaned training cases: {len(train_df)}\n")
        f.write(f"Cleaned testing-source cases: {len(test_source_df)}\n")
        f.write(f"Hold-out test cases used for model selection: {len(holdout_50)}\n")
        f.write(f"Remaining second-CSV cases not used for selection: {len(remaining_test_source)}\n")
        f.write(f"Target type: {TARGET_TYPE}\n")
        f.write(f"Random state: {RANDOM_STATE}\n\n")

        f.write("Important methodological note:\n")
        f.write(
            "Candidate models were trained only on the training CSV. "
            "They were evaluated only on the 50 hold-out cases sampled from the second CSV. "
            "After model selection, no final refit was performed.\n\n"
        )

        f.write("Selected model:\n")
        f.write(f"{best_model_name}\n\n")

        f.write("Model comparison results on 50 hold-out cases:\n")
        f.write(results_df.to_string(index=False))

        f.write("\n\nSelected model equations:\n")
        f.write(selected_params_df.to_string(index=False))

    print(f"Summary saved to: {summary_path}")

    # ------------------------------------------------------------
    # PLOTS
    # ------------------------------------------------------------

    plot_model_comparison(results_df, OUTPUT_DIR)
    plot_final_fit(
        train_df=train_df,
        test_50_df=holdout_50,
        final_model_name=best_model_name,
        final_model_object=selected_model_object,
        final_breaks=selected_breaks,
        output_dir=OUTPUT_DIR,
    )
    plot_predicted_vs_observed(test_best, OUTPUT_DIR)
    plot_residual_vs_footprint(test_best, OUTPUT_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()