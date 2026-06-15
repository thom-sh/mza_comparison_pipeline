import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# ============================================================
# THESIS PLOT STYLE
# ============================================================

LEGEND_EDGE_COLOR = "#BDC1C5"
POLYGON_EDGE_COLOR = "#777D84"

# Teal graph family
TEAL_LIGHT = "#D7ECEA"
TEAL_MID_LIGHT = "#A9D1CE"
TEAL_MID = "#6FA9A6"
TEAL_DARK = "#3F7D7A"
TEAL_DEEP = "#285452"

# Neutral greys
GREY_DARK = "#5F666D"
GREY_MID = "#999999"
GREY_LIGHT = "#BABABA"
GREY_VERY_LIGHT = "#DBDBDB"

GRID_COLOR = "#E6E8EA"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIX Two Text", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",

    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,

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
# PIECEWISE LINEAR MODEL HELPERS
# ============================================================

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


def make_quantile_breaks(A_values, K):
    """
    K regimes -> K - 1 quantile breakpoints.
    """
    qs = np.linspace(0, 1, K + 1)[1:-1]
    return tuple(np.quantile(A_values, qs))


# ============================================================
# PLOTTING HELPERS
# ============================================================

def save_figure(fig, output_dir, filename_base):
    # png_path = output_dir / f"{filename_base}.png"
    pdf_path = output_dir / f"{filename_base}.pdf"

    # fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    # print(f"Saved plot: {png_path}")
    print(f"Saved plot: {pdf_path}")


def style_legend(legend):
    if legend is not None:
        legend.get_frame().set_edgecolor(LEGEND_EDGE_COLOR)
        legend.get_frame().set_linewidth(0.8)
        legend.get_frame().set_facecolor("white")


def plot_model_comparison(results_df, output_dir):
    """
    Plot 1: Bar plot of MAE for all candidate models.
    """
    df_plot = results_df.sort_values("MAE_m2", ascending=True).copy()

    fig, ax = plt.subplots(figsize=(5.8, 2.8))

    colors = []
    for model in df_plot["model"]:
        if model == df_plot.iloc[0]["model"]:
            colors.append(TEAL_DEEP)          # selected / best model
        elif model.startswith("M2"):
            colors.append(TEAL_MID)           # other piecewise models
        elif model.startswith("B1"):
            colors.append(GREY_LIGHT)         # simple linear baseline
        else:
            colors.append(GREY_VERY_LIGHT)    # constant baseline

    bars = ax.bar(
        range(len(df_plot)),
        df_plot["MAE_m2"],
        width=0.62,
        color=colors,
        edgecolor=POLYGON_EDGE_COLOR,
        linewidth=0.7,
    )

    ax.set_ylabel("MAE [m²]")
    ax.set_xlabel("Candidate model")
    # ax.set_title("Model comparison for dwelling-area estimation")
    ax.set_xticks(range(len(df_plot)))
    ax.set_xticklabels(df_plot["model"], rotation=35, ha="right")

    ax.grid(axis="y", color=GRID_COLOR, alpha=0.8, linewidth=0.5)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, df_plot["MAE_m2"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_ylim(0, 11.5)
    fig.tight_layout()
    save_figure(fig, output_dir, "plot_01_model_comparison_mae")
    plt.show()
    plt.close(fig)


def plot_final_fit(g, final_model_name, final_model_object, final_breaks, output_dir):
    """
    Plot 2: Footprint area vs observed target with final refitted model.
    """
    fig, ax = plt.subplots(figsize=(5.8, 2.8))

    ax.scatter(
        g["A"],
        g["y"],
        s=18,
        alpha=0.65,
        facecolor=TEAL_MID_LIGHT,
        edgecolor=POLYGON_EDGE_COLOR,
        linewidth=0.35,
        label="MSD cases",
    )

    A_plot = np.linspace(g["A"].min(), g["A"].max(), 600)

    if final_model_name.startswith("M2"):
        y_plot = predict_piecewise(A_plot, final_model_object, final_breaks)

        ax.plot(
            A_plot,
            y_plot,
            color=TEAL_DEEP,
            linewidth=1.8,
            label="Final piecewise model",
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
            color=TEAL_DEEP,
            linewidth=1.8,
            label="Final linear model",
        )

    elif final_model_name.startswith("B0"):
        constant_value = final_model_object
        ax.axhline(
            constant_value,
            color=TEAL_DEEP,
            linewidth=1.8,
            label="Final constant baseline",
        )

    ax.set_xlabel(r"External footprint area $A_\mathrm{fp}$ [m²]")
    ax.set_ylabel(r"Mean apartment area $A_\mathrm{apt}$ [m²]")
    # ax.set_title("Final dwelling-area estimation model")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)

    legend = ax.legend(loc="best", frameon=True)
    style_legend(legend)

    fig.tight_layout()
    save_figure(fig, output_dir, "plot_02_final_model_fit")
    plt.show()
    plt.close(fig)


def plot_predicted_vs_observed(test_best, output_dir):
    """
    Plot 3: Predicted vs observed apartment area for the hold-out test set.
    """
    y = test_best["y"].values
    yhat = test_best["yhat_best"].values

    mn = min(np.min(y), np.min(yhat))
    mx = max(np.max(y), np.max(yhat))

    fig, ax = plt.subplots(figsize=(5.8, 3.6))

    ax.scatter(
        y,
        yhat,
        s=22,
        alpha=0.70,
        facecolor=TEAL_MID_LIGHT,
        edgecolor=POLYGON_EDGE_COLOR,
        linewidth=0.35,
        label="Test buildings",
    )

    ax.plot(
        [mn, mx],
        [mn, mx],
        color=TEAL_DEEP,
        linestyle="--",
        linewidth=1.1,
        label="Perfect prediction",
    )

    ax.set_xlabel("Observed mean apartment area [m²]")
    ax.set_ylabel("Predicted mean apartment area [m²]")
    # ax.set_title("Predicted vs observed values")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)

    legend = ax.legend(loc="best", frameon=True)
    style_legend(legend)

    fig.tight_layout()
    save_figure(fig, output_dir, "plot_03_predicted_vs_observed_test")
    plt.show()
    plt.close(fig)


def plot_residual_vs_footprint(test_best, output_dir):
    """
    Plot 4: Residual vs footprint area for the hold-out test set.
    """
    fig, ax = plt.subplots(figsize=(5.8, 2.8))

    ax.scatter(
        test_best["A"],
        test_best["residual_best"],
        s=22,
        alpha=0.70,
        facecolor=TEAL_MID_LIGHT,
        edgecolor=POLYGON_EDGE_COLOR,
        linewidth=0.35,
        label="Test buildings",
    )

    ax.axhline(
        0,
        color=TEAL_DEEP,
        linestyle="--",
        linewidth=1.1,
        label="Zero residual",
    )

    ax.set_xlabel(r"External footprint area $A_\mathrm{fp}$ [m²]")
    ax.set_ylabel("Residual [m²]")
    # ax.set_title("Residuals across footprint area")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)

    legend = ax.legend(loc="best", frameon=True)
    style_legend(legend)

    fig.tight_layout()
    save_figure(fig, output_dir, "plot_04_residual_vs_footprint_test")
    plt.show()
    plt.close(fig)


# ============================================================
# MAIN SCRIPT
# ============================================================

def main():
    # ------------------------------------------------------------
    # USER SETTINGS
    # ------------------------------------------------------------
    input_csv = Path(r"C:\WF\Thomas Sharon\Floorplan_Dataset\msd_dataset_creation\new_method\model_comparison_results_outcome\msd_area_estimation_400_train_test_geometry_topology_matched_no_validation_overlap.csv")

    output_dir = Path(r"C:\WF\Thomas Sharon\Floorplan_Dataset\msd_dataset_creation\new_method\model_comparison_results_outcome")

    target_type = "mean"      # "mean" is recommended because MZA uses this for dwelling-count estimation
    test_size = 0.30          # 70% train, 30% test
    random_state = 42

    k_values = [2, 3, 4, 5, 6]

    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # LOAD DATA
    # ------------------------------------------------------------
    df = pd.read_csv(input_csv)

    apt_cols = [
        c for c in df.columns
        if c.startswith("apt_") and c.endswith("_area")
    ]

    if not apt_cols:
        raise ValueError("No apartment area columns found. Expected columns like apt_1_area.")

    for c in apt_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["footprint_area"] = pd.to_numeric(df["footprint_area"], errors="coerce")

    if "building_id" not in df.columns:
        raise ValueError("Input CSV must contain a 'building_id' column.")

    # ------------------------------------------------------------
    # TARGET VARIABLE
    # ------------------------------------------------------------
    apt_values = df[apt_cols].to_numpy(dtype=float)
    valid = np.where((apt_values > 0) & np.isfinite(apt_values), apt_values, np.nan)

    df["n_valid_apts"] = np.sum(np.isfinite(valid), axis=1)

    if target_type == "mean":
        df["y_target"] = np.nanmean(valid, axis=1)
    elif target_type == "median":
        df["y_target"] = np.nanmedian(valid, axis=1)
    else:
        raise ValueError("target_type must be either 'mean' or 'median'.")

    # ------------------------------------------------------------
    # CLEAN DATA
    # ------------------------------------------------------------
    df_clean = df.dropna(subset=["building_id", "footprint_area", "y_target"]).copy()
    df_clean = df_clean[df_clean["footprint_area"] > 0]
    df_clean = df_clean[df_clean["n_valid_apts"] >= 2]

    g = df_clean[["building_id", "footprint_area", "y_target", "n_valid_apts"]].copy()
    g = g.rename(columns={"footprint_area": "A", "y_target": "y"})
    g = g.reset_index(drop=True)

    print(f"Buildings after cleaning: {len(g)}")

    # Save cleaned dataset used for model comparison
    cleaned_path = output_dir / "cleaned_dwelling_area_estimation_dataset.csv"
    g.to_csv(cleaned_path, index=False)

    # ------------------------------------------------------------
    # TRAIN / TEST SPLIT
    # Stratified by footprint-area quartiles
    # ------------------------------------------------------------
    g["A_bin"] = pd.qcut(g["A"], q=4, labels=False, duplicates="drop")

    train_idx, test_idx = train_test_split(
        g.index,
        test_size=test_size,
        random_state=random_state,
        stratify=g["A_bin"],
    )

    train = g.loc[train_idx].copy().reset_index(drop=True)
    test = g.loc[test_idx].copy().reset_index(drop=True)

    print(f"Training buildings: {len(train)}")
    print(f"Test buildings: {len(test)}")

    split_df = pd.concat(
        [
            train.assign(split="train"),
            test.assign(split="test"),
        ],
        ignore_index=True,
    )
    split_path = output_dir / "train_test_split_buildings.csv"
    split_df.to_csv(split_path, index=False)

    results = []
    prediction_tables = []

    # ============================================================
    # B0: CONSTANT BASELINE
    # ============================================================
    b0_value = float(train["y"].mean())
    test["yhat_B0"] = b0_value

    results.append(
        evaluate_model(
            "B0_constant_training_mean",
            test["y"].values,
            test["yhat_B0"].values,
        )
    )

    prediction_tables.append(
        test[["building_id", "A", "y", "yhat_B0"]]
        .rename(columns={"yhat_B0": "yhat"})
        .assign(model="B0_constant_training_mean")
    )

    # ============================================================
    # B1: SIMPLE LINEAR REGRESSION
    # ============================================================
    b1 = LinearRegression()
    b1.fit(train[["A"]], train["y"])

    test["yhat_B1"] = b1.predict(test[["A"]])

    results.append(
        evaluate_model(
            "B1_simple_linear_regression",
            test["y"].values,
            test["yhat_B1"].values,
        )
    )

    prediction_tables.append(
        test[["building_id", "A", "y", "yhat_B1"]]
        .rename(columns={"yhat_B1": "yhat"})
        .assign(model="B1_simple_linear_regression")
    )

    # ============================================================
    # M2: QUANTILE-BASED PIECEWISE LINEAR REGRESSION
    # ============================================================
    for K in k_values:
        breaks = make_quantile_breaks(train["A"].values, K)

        try:
            models, breaks = fit_piecewise_linear(train, breaks)
            yhat = predict_piecewise(test["A"].values, models, breaks)

            model_name = f"M2_piecewise_linear_K{K}"

            results.append(
                evaluate_model(
                    model_name,
                    test["y"].values,
                    yhat,
                )
            )

            pred_df = test[["building_id", "A", "y"]].copy()
            pred_df["yhat"] = yhat
            pred_df["model"] = model_name
            prediction_tables.append(pred_df)

        except ValueError as e:
            print(f"Skipping M2 K={K}: {e}")

    # ============================================================
    # SAVE MODEL COMPARISON RESULTS
    # ============================================================
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(["MAE_m2", "RMSE_m2"]).reset_index(drop=True)

    print("\n=== MODEL COMPARISON RESULTS ===")
    print(results_df.to_string(index=False))

    best_model_name = str(results_df.iloc[0]["model"])

    print("\nBest model based on lowest test MAE:")
    print(best_model_name)

    results_path = output_dir / "model_comparison_results.csv"
    predictions_path = output_dir / "model_predictions_test_set.csv"

    results_df.to_csv(results_path, index=False)

    all_predictions = pd.concat(prediction_tables, ignore_index=True)
    all_predictions.to_csv(predictions_path, index=False)

    print(f"\nResults saved to: {results_path}")
    print(f"Predictions saved to: {predictions_path}")

    # Save best model test predictions
    test_best = all_predictions[all_predictions["model"] == best_model_name].copy()
    test_best = test_best.rename(columns={"yhat": "yhat_best"})
    test_best["residual_best"] = test_best["yhat_best"] - test_best["y"]

    best_predictions_path = output_dir / "best_model_predictions_test_set.csv"
    test_best.to_csv(best_predictions_path, index=False)

    # ============================================================
    # FINAL REFIT ON FULL CLEANED DATASET
    # ============================================================
    final_params = []
    final_model_object = None
    final_breaks = tuple()

    if best_model_name.startswith("M2_piecewise_linear"):
        final_K = int(best_model_name.split("K")[-1])

        final_breaks = make_quantile_breaks(g["A"].values, final_K)
        final_models, final_breaks = fit_piecewise_linear(g, final_breaks)
        final_model_object = final_models

        print("\n=== FINAL REFIT MODEL ON FULL DATASET ===")
        print(f"Selected model: M2_piecewise_linear_K{final_K}")
        print("Final breakpoints based on full cleaned dataset:")

        for b in final_breaks:
            print(f"  {b:.4f}")

        print("\nFinal piecewise equations:")

        for i, model in enumerate(final_models, start=1):
            alpha = float(model.intercept_)
            beta = float(model.coef_[0])

            if i == 1:
                lower = -np.inf
                upper = final_breaks[0]
                interval = f"A_fp <= {upper:.4f}"
            elif i == len(final_models):
                lower = final_breaks[-1]
                upper = np.inf
                interval = f"A_fp > {lower:.4f}"
            else:
                lower = final_breaks[i - 2]
                upper = final_breaks[i - 1]
                interval = f"{lower:.4f} < A_fp <= {upper:.4f}"

            equation = f"y_hat = {alpha:.4f} + {beta:.6f} * A_fp"

            print(f"  Regime {i}: {interval}")
            print(f"    {equation}")

            final_params.append({
                "selected_model": best_model_name,
                "target_type": target_type,
                "regime": i,
                "lower_bound_A_fp": lower,
                "upper_bound_A_fp": upper,
                "alpha_intercept": alpha,
                "beta_slope": beta,
                "equation": equation,
            })

        g_final = g.copy()
        g_final["yhat_final"] = predict_piecewise(g_final["A"].values, final_models, final_breaks)

    elif best_model_name == "B1_simple_linear_regression":
        final_b1 = LinearRegression()
        final_b1.fit(g[["A"]], g["y"])
        final_model_object = final_b1

        alpha = float(final_b1.intercept_)
        beta = float(final_b1.coef_[0])

        print("\n=== FINAL REFIT MODEL ON FULL DATASET ===")
        print("Selected model: B1_simple_linear_regression")
        print(f"Final equation: y_hat = {alpha:.4f} + {beta:.6f} * A_fp")

        final_params.append({
            "selected_model": best_model_name,
            "target_type": target_type,
            "regime": 1,
            "lower_bound_A_fp": -np.inf,
            "upper_bound_A_fp": np.inf,
            "alpha_intercept": alpha,
            "beta_slope": beta,
            "equation": f"y_hat = {alpha:.4f} + {beta:.6f} * A_fp",
        })

        g_final = g.copy()
        g_final["yhat_final"] = final_b1.predict(g_final[["A"]])

    elif best_model_name == "B0_constant_training_mean":
        final_b0_value = float(g["y"].mean())
        final_model_object = final_b0_value

        print("\n=== FINAL REFIT MODEL ON FULL DATASET ===")
        print("Selected model: B0_constant_mean")
        print(f"Final constant estimate: y_hat = {final_b0_value:.4f} m²")

        final_params.append({
            "selected_model": best_model_name,
            "target_type": target_type,
            "regime": 1,
            "lower_bound_A_fp": -np.inf,
            "upper_bound_A_fp": np.inf,
            "alpha_intercept": final_b0_value,
            "beta_slope": 0.0,
            "equation": f"y_hat = {final_b0_value:.4f}",
        })

        g_final = g.copy()
        g_final["yhat_final"] = final_b0_value

    else:
        raise ValueError(f"Unknown selected model: {best_model_name}")

    g_final["residual_final"] = g_final["yhat_final"] - g_final["y"]

    final_params_df = pd.DataFrame(final_params)
    final_params_path = output_dir / "final_refit_model_parameters.csv"
    final_predictions_path = output_dir / "final_refit_predictions_full_dataset.csv"

    final_params_df.to_csv(final_params_path, index=False)
    g_final.to_csv(final_predictions_path, index=False)

    print(f"\nFinal model parameters saved to: {final_params_path}")
    print(f"Final fitted predictions saved to: {final_predictions_path}")

    # ============================================================
    # SAVE SUMMARY FILE
    # ============================================================
    summary_path = output_dir / "selected_model_summary.txt"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("MSD dwelling-area estimation model selection\n")
        f.write("===========================================\n\n")
        f.write(f"Input CSV: {input_csv}\n")
        f.write(f"Buildings after cleaning: {len(g)}\n")
        f.write(f"Training buildings: {len(train)}\n")
        f.write(f"Test buildings: {len(test)}\n")
        f.write(f"Target type: {target_type}\n")
        f.write(f"Test size: {test_size}\n")
        f.write(f"Random state: {random_state}\n\n")
        f.write("Selected model:\n")
        f.write(f"{best_model_name}\n\n")
        f.write("Model comparison results:\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\nFinal refit model parameters:\n")
        f.write(final_params_df.to_string(index=False))

    print(f"Summary saved to: {summary_path}")

    # ============================================================
    # PLOTS
    # ============================================================
    plot_model_comparison(results_df, output_dir)
    plot_final_fit(g, best_model_name, final_model_object, final_breaks, output_dir)
    plot_predicted_vs_observed(test_best, output_dir)
    plot_residual_vs_footprint(test_best, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()