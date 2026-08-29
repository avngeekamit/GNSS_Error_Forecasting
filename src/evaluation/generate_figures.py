"""
Generate Figures

Uses: data/GEO_day6_actual.csv
and prediction files from:
    results/hybrid_xgboost/
    results/gru/
    results/lstm/

Generates:
    1. Actual vs predicted forecast plots
    2. Horizon-wise absolute error plots
    3. Model comparison plots for MAE, RMSE and R2

All figures are generated from the same Day-6 ground-truth data.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# PATHS

BASE_DIR = Path(__file__).resolve().parents[2]

ACTUAL_PATH = (
    BASE_DIR
    / "data"
    / "GEO_day6_actual.csv"
)

PREDICTION_FILES = {

    "Hybrid XGBoost":
        BASE_DIR
        / "results"
        / "hybrid_xgboost"
        / "b12_day6_hybrid_predictions.csv",

    "GRU":
        BASE_DIR
        / "results"
        / "gru"
        / "GRU_DAY6_predictions.csv",

    "LSTM":
        BASE_DIR
        / "results"
        / "lstm"
        / "LSTM_DAY6_predictions.csv",
}


FIGURES_DIR = (
    BASE_DIR
    / "figures"
)

FORECAST_DIR = (
    FIGURES_DIR
    / "forecasts"
)

HORIZON_DIR = (
    FIGURES_DIR
    / "horizon_analysis"
)

COMPARISON_DIR = (
    FIGURES_DIR
    / "model_comparison"
)


# CONFIGURATION

TARGETS = [
    "x_error",
    "y_error",
    "z_error",
    "satclockerror",
]

TARGET_LABELS = {

    "x_error":
        "X-axis Ephemeris Error",

    "y_error":
        "Y-axis Ephemeris Error",

    "z_error":
        "Z-axis Ephemeris Error",

    "satclockerror":
        "Satellite Clock Error",
}


MODEL_FOLDER_NAMES = {

    "Hybrid XGBoost":
        "hybrid_xgboost",

    "GRU":
        "gru",

    "LSTM":
        "lstm",
}

# LOAD ACTUAL DATA


def load_actual_data():

    print("\nLoading actual Day-6 data...")

    actual_df = pd.read_csv(
        ACTUAL_PATH,
        index_col=0,
        parse_dates=True,
    )

    print(
        f"Actual observations: {len(actual_df)}"
    )

    return actual_df


# =============================================================================
# LOAD PREDICTIONS
# =============================================================================

def load_predictions():

    predictions = {}

    for model_name, path in PREDICTION_FILES.items():

        if not path.exists():

            raise FileNotFoundError(
                f"\nPrediction file not found for "
                f"{model_name}:\n{path}"
            )

        df = pd.read_csv(
            path
        )

        df["timestamp"] = pd.to_datetime(
            df["timestamp"]
        )

        predictions[model_name] = df

        print(
            f"{model_name}: "
            f"{len(df)} predictions loaded"
        )

    return predictions


# =============================================================================
# CREATE DIRECTORIES
# =============================================================================

def create_directories():

    FORECAST_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    HORIZON_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    COMPARISON_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# =============================================================================
# FORECAST FIGURES
# =============================================================================

def generate_forecast_figures(
    actual_df,
    predictions,
):

    print(
        "\nGenerating forecast figures..."
    )

    for model_name, prediction_df in predictions.items():

        model_folder = (
            FORECAST_DIR
            / MODEL_FOLDER_NAMES[model_name]
        )

        model_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        for target in TARGETS:

            plt.figure(
                figsize=(12, 6)
            )

            # -----------------------------------------------------------------
            # Actual
            # -----------------------------------------------------------------

            plt.plot(
                actual_df.index,
                actual_df[target],
                label="Actual",
                linewidth=2,
            )

            # -----------------------------------------------------------------
            # Prediction
            # -----------------------------------------------------------------

            plt.plot(
                prediction_df["timestamp"],
                prediction_df[target],
                label=model_name,
                linewidth=2,
                linestyle="--",
            )

            plt.title(
                f"{model_name}: "
                f"Day-6 {TARGET_LABELS[target]}"
            )

            plt.xlabel(
                "UTC Time"
            )

            plt.ylabel(
                "Error"
            )

            plt.legend()

            plt.grid(
                True,
                alpha=0.3,
            )

            plt.tight_layout()

            output_path = (
                model_folder
                / f"{target}.png"
            )

            plt.savefig(
                output_path,
                dpi=300,
                bbox_inches="tight",
            )

            plt.close()

            print(
                f"  Created: {output_path}"
            )


# =============================================================================
# HORIZON-WISE ERROR FIGURES
# =============================================================================

def generate_horizon_figures(
    actual_df,
    predictions,
):

    print(
        "\nGenerating horizon-wise error figures..."
    )

    for target in TARGETS:

        plt.figure(
            figsize=(12, 6)
        )

        for model_name, prediction_df in predictions.items():

            actual_values = (
                actual_df[target]
                .to_numpy()
            )

            predicted_values = (
                prediction_df[target]
                .to_numpy()
            )

            absolute_error = np.abs(
                predicted_values
                - actual_values
            )

            horizon_minutes = (
                prediction_df[
                    "forecast_horizon_minutes"
                ].to_numpy()
            )

            plt.plot(
                horizon_minutes,
                absolute_error,
                label=model_name,
                linewidth=1.8,
            )

        plt.title(
            f"Horizon-wise Absolute Error: "
            f"{TARGET_LABELS[target]}"
        )

        plt.xlabel(
            "Forecast Horizon (minutes)"
        )

        plt.ylabel(
            "Absolute Error"
        )

        plt.legend()

        plt.grid(
            True,
            alpha=0.3,
        )

        plt.tight_layout()

        output_path = (
            HORIZON_DIR
            / f"{target}_horizon_error.png"
        )

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        print(
            f"  Created: {output_path}"
        )


# =============================================================================
# METRIC CALCULATION
# =============================================================================

def calculate_metric(
    actual,
    predicted,
):

    error = (
        predicted
        - actual
    )

    absolute_error = np.abs(
        error
    )

    mae = np.mean(
        absolute_error
    )

    rmse = np.sqrt(
        np.mean(
            error ** 2
        )
    )

    # R2

    ss_res = np.sum(
        error ** 2
    )

    ss_tot = np.sum(
        (actual - np.mean(actual)) ** 2
    )

    if ss_tot == 0:

        r2 = np.nan

    else:

        r2 = 1 - (
            ss_res / ss_tot
        )

    return mae, rmse, r2


# =============================================================================
# MODEL COMPARISON FIGURES
# =============================================================================

def generate_model_comparison_figures(
    actual_df,
    predictions,
):

    print(
        "\nCalculating model comparison metrics..."
    )

    results = []

    for model_name, prediction_df in predictions.items():

        for target in TARGETS:

            actual = (
                actual_df[target]
                .to_numpy()
            )

            predicted = (
                prediction_df[target]
                .to_numpy()
            )

            mae, rmse, r2 = calculate_metric(
                actual,
                predicted,
            )

            results.append({

                "model":
                    model_name,

                "target":
                    target,

                "MAE":
                    mae,

                "RMSE":
                    rmse,

                "R2":
                    r2,
            })

    results_df = pd.DataFrame(
        results
    )

    # Save a clean metric table as well

    results_path = (
        COMPARISON_DIR
        / "figure_metric_values.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # MAE
    # -------------------------------------------------------------------------

    create_metric_comparison_plot(
        results_df,
        metric="MAE",
        title="Day-6 MAE Comparison",
        ylabel="MAE",
        filename="mae_comparison.png",
    )

    # -------------------------------------------------------------------------
    # RMSE
    # -------------------------------------------------------------------------

    create_metric_comparison_plot(
        results_df,
        metric="RMSE",
        title="Day-6 RMSE Comparison",
        ylabel="RMSE",
        filename="rmse_comparison.png",
    )

    # -------------------------------------------------------------------------
    # R2
    # -------------------------------------------------------------------------

    create_metric_comparison_plot(
        results_df,
        metric="R2",
        title="Day-6 R² Comparison",
        ylabel="R²",
        filename="r2_comparison.png",
    )


# =============================================================================
# METRIC PLOT FUNCTION
# =============================================================================

def create_metric_comparison_plot(
    results_df,
    metric,
    title,
    ylabel,
    filename,
):

    # Pivot:
    #
    # rows    = target
    # columns = model

    pivot_df = results_df.pivot(
        index="target",
        columns="model",
        values=metric,
    )

    # Keep consistent model order

    model_order = [
        "Hybrid XGBoost",
        "GRU",
        "LSTM",
    ]

    model_order = [
        model
        for model in model_order
        if model in pivot_df.columns
    ]

    pivot_df = pivot_df[
        model_order
    ]

    ax = pivot_df.plot(
        kind="bar",
        figsize=(12, 6),
    )

    ax.set_title(
        title
    )

    ax.set_xlabel(
        "GNSS Error Component"
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_xticklabels(
        [
            TARGET_LABELS[target]
            for target in pivot_df.index
        ],
        rotation=20,
        ha="right",
    )

    ax.legend(
        title="Model"
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    output_path = (
        COMPARISON_DIR
        / filename
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"  Created: {output_path}"
    )


# =============================================================================
# COMBINED FORECAST COMPARISON
# =============================================================================

def generate_combined_forecast_figures(
    actual_df,
    predictions,
):

    print(
        "\nGenerating combined model forecast figures..."
    )

    for target in TARGETS:

        plt.figure(
            figsize=(14, 7)
        )

        # Actual

        plt.plot(
            actual_df.index,
            actual_df[target],
            label="Actual",
            linewidth=2.5,
        )

        # Models

        for model_name, prediction_df in predictions.items():

            plt.plot(
                prediction_df["timestamp"],
                prediction_df[target],
                label=model_name,
                linewidth=1.8,
                linestyle="--",
            )

        plt.title(
            f"Day-6 Forecast Comparison: "
            f"{TARGET_LABELS[target]}"
        )

        plt.xlabel(
            "UTC Time"
        )

        plt.ylabel(
            "Error"
        )

        plt.legend()

        plt.grid(
            True,
            alpha=0.3,
        )

        plt.tight_layout()

        output_path = (
            FORECAST_DIR
            / f"{target}_all_models.png"
        )

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        print(
            f"  Created: {output_path}"
        )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print(
        "=" * 90
    )

    print(
        "GNSS ERROR FORECASTING - FIGURE GENERATION"
    )

    print(
        "=" * 90
    )

    create_directories()

    actual_df = load_actual_data()

    predictions = load_predictions()

    # -------------------------------------------------------------------------
    # Generate figures
    # -------------------------------------------------------------------------

    generate_forecast_figures(
        actual_df,
        predictions,
    )

    generate_combined_forecast_figures(
        actual_df,
        predictions,
    )

    generate_horizon_figures(
        actual_df,
        predictions,
    )

    generate_model_comparison_figures(
        actual_df,
        predictions,
    )

    print(
        "\n" + "=" * 90
    )

    print(
        "FIGURE GENERATION COMPLETE"
    )

    print(
        "=" * 90
    )

    print(
        "\nFigures saved under:"
    )

    print(
        FIGURES_DIR
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    main()