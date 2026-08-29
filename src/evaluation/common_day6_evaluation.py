"""
Common Day-6 Evaluation
GNSS Error Forecasting Research

Models compared:
    1. Hybrid XGBoost
    2. Direct GRU
    3. Direct LSTM

Ground truth:
    GEO_day6_actual.csv

All three models are evaluated using the same Day-6
ground-truth data and the same evaluation metrics.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)



# PATH CONFIGURATION

# parents[2] = repository root

BASE_DIR = Path(__file__).resolve().parents[2]

ACTUAL_PATH = (
    BASE_DIR
    / "data"
    / "GEO_day6_actual.csv"
)

PREDICTION_FILES = {

    "Hybrid_XGBoost":
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

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "model_comparison"
)



# DATA CONFIGURATION


TARGETS = [
    "x_error",
    "y_error",
    "z_error",
    "satclockerror",
]

PRED_LEN = 95

INTERVAL_MINUTES = 15

# LOAD ACTUAL DAY-6 DATA

def load_actual_data():

    if not ACTUAL_PATH.exists():

        raise FileNotFoundError(
            f"\nActual Day-6 file not found:\n"
            f"{ACTUAL_PATH}"
        )

    actual_df = pd.read_csv(
        ACTUAL_PATH,
        index_col=0,
        parse_dates=True,
    )

    # Check target columns

    missing_targets = [
        target
        for target in TARGETS
        if target not in actual_df.columns
    ]

    if missing_targets:

        raise ValueError(
            f"\nMissing target columns in actual data: "
            f"{missing_targets}"
        )

    # Check number of observations

    if len(actual_df) != PRED_LEN:

        raise ValueError(
            f"\nExpected {PRED_LEN} Day-6 observations, "
            f"but found {len(actual_df)}."
        )

    return actual_df


# LOAD MODEL PREDICTIONS

def load_prediction_file(
    model_name,
    prediction_path,
):

    if not prediction_path.exists():

        raise FileNotFoundError(
            f"\nPrediction file for {model_name} not found:\n"
            f"{prediction_path}"
        )

    prediction_df = pd.read_csv(
        prediction_path
    )

    # Check target columns

    missing_targets = [
        target
        for target in TARGETS
        if target not in prediction_df.columns
    ]

    if missing_targets:

        raise ValueError(
            f"\n{model_name}: missing target columns: "
            f"{missing_targets}"
        )

    # Check prediction length

    if len(prediction_df) != PRED_LEN:

        raise ValueError(
            f"\n{model_name}: expected {PRED_LEN} predictions, "
            f"but found {len(prediction_df)}."
        )

    return prediction_df



# TIMESTAMP VALIDATION


def validate_timestamps(
    model_name,
    prediction_df,
    actual_df,
):


    if "timestamp" not in prediction_df.columns:

        print(
            f"\nWarning: {model_name} prediction file "
            f"does not contain a timestamp column."
        )

        return

    prediction_timestamps = pd.to_datetime(
        prediction_df["timestamp"]
    ).to_numpy()

    actual_timestamps = pd.to_datetime(
        actual_df.index
    ).to_numpy()

   

    if not np.array_equal(
        prediction_timestamps,
        actual_timestamps
    ):

        raise ValueError(
            f"\n{model_name}: prediction timestamps "
            f"do not match GEO_day6_actual.csv."
        )



# METRIC CALCULATION


def calculate_metrics(
    actual,
    predicted,
):

    # Prediction error
    #
    # Error = Prediction - Actual

    error = predicted - actual

    absolute_error = np.abs(
        error
    )

   
    # MAE
    

    mae = mean_absolute_error(
        actual,
        predicted,
    )

    # RMSE

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted,
        )
    )

    
    # R²

    r2 = r2_score(
        actual,
        predicted,
    )

    
    # Maximum Absolute Error

    max_ae = np.max(
        absolute_error
    )

    
    # 95th Percentile Absolute Error
    p95_ae = np.percentile(
        absolute_error,
        95,
    )

    
    # Bias
    

    bias = np.mean(
        error
    )

    
    # Correlation
    

    if (
        np.std(actual) == 0
        or np.std(predicted) == 0
    ):

        correlation = np.nan

    else:

        correlation = np.corrcoef(
            actual,
            predicted,
        )[0, 1]

    return {

        "MAE": mae,

        "RMSE": rmse,

        "R2": r2,

        "MaxAE": max_ae,

        "P95_AE": p95_ae,

        "Bias": bias,

        "Correlation": correlation,
    }



# OVERALL DAY-6 METRICS


def calculate_overall_metrics(
    actual_df,
):

    rows = []

    for model_name, prediction_path in PREDICTION_FILES.items():

        prediction_df = load_prediction_file(
            model_name,
            prediction_path,
        )

        validate_timestamps(
            model_name,
            prediction_df,
            actual_df,
        )

        for target in TARGETS:

            actual = (
                actual_df[target]
                .to_numpy()
            )

            predicted = (
                prediction_df[target]
                .to_numpy()
            )

            metrics = calculate_metrics(
                actual,
                predicted,
            )

            row = {

                "model":
                    model_name,

                "target":
                    target,

                "forecast_points":
                    PRED_LEN,

                **metrics,
            }

            rows.append(row)

    return pd.DataFrame(
        rows
    )


# =============================================================================
# HORIZON-WISE ABSOLUTE ERROR
# =============================================================================

def calculate_horizon_metrics(
    actual_df,
):

    rows = []

    horizon_minutes = (
        np.arange(
            1,
            PRED_LEN + 1,
        )
        * INTERVAL_MINUTES
    )

    for model_name, prediction_path in PREDICTION_FILES.items():

        prediction_df = load_prediction_file(
            model_name,
            prediction_path,
        )

        validate_timestamps(
            model_name,
            prediction_df,
            actual_df,
        )

        for i in range(PRED_LEN):

            row = {

                "model":
                    model_name,

                "horizon_minutes":
                    int(
                        horizon_minutes[i]
                    ),
            }

            for target in TARGETS:

                actual_value = (
                    actual_df[target]
                    .iloc[i]
                )

                predicted_value = (
                    prediction_df[target]
                    .iloc[i]
                )

                absolute_error = abs(
                    predicted_value
                    - actual_value
                )

                row[
                    f"{target}_AE"
                ] = absolute_error

            rows.append(row)

    horizon_df = pd.DataFrame(
        rows
    )

    # -------------------------------------------------------------------------
    # Cumulative MAE
    # -------------------------------------------------------------------------

    for target in TARGETS:

        horizon_df[
            f"{target}_cumulative_MAE"
        ] = (

            horizon_df
            .groupby("model")
            [f"{target}_AE"]
            .transform(
                lambda x:
                x.expanding().mean()
            )
        )

    return horizon_df


# =============================================================================
# MODEL COMPARISON SUMMARY
# =============================================================================

def create_summary_table(
    metrics_df,
):

    metric_columns = [

        "MAE",

        "RMSE",

        "R2",

        "MaxAE",

        "P95_AE",

        "Bias",

        "Correlation",
    ]

    summary_df = metrics_df.pivot(
        index="target",
        columns="model",
        values=metric_columns,
    )

    return summary_df


# =============================================================================
# MAIN
# =============================================================================

def main():

    print(
        "=" * 90
    )

    print(
        "COMMON DAY-6 MODEL EVALUATION"
    )

    print(
        "GNSS ERROR FORECASTING RESEARCH"
    )

    print(
        "=" * 90
    )

    # -------------------------------------------------------------------------
    # Create output directory
    # -------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Load actual Day-6 data
    # -------------------------------------------------------------------------

    actual_df = load_actual_data()

    print(
        "\nActual Day-6 data:"
    )

    print(
        f"  Observations : {len(actual_df)}"
    )

    print(
        f"  Start        : {actual_df.index.min()}"
    )

    print(
        f"  End          : {actual_df.index.max()}"
    )

    # -------------------------------------------------------------------------
    # Check prediction files
    # -------------------------------------------------------------------------

    print(
        "\nPrediction files:"
    )

    for model_name, prediction_path in PREDICTION_FILES.items():

        print(
            f"  {model_name}:"
        )

        print(
            f"    {prediction_path}"
        )

    # -------------------------------------------------------------------------
    # Overall metrics
    # -------------------------------------------------------------------------

    metrics_df = calculate_overall_metrics(
        actual_df
    )

    metrics_path = (
        OUTPUT_DIR
        / "DAY6_model_comparison_metrics.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    print(
        "\n" + "=" * 90
    )

    print(
        "OVERALL DAY-6 METRICS"
    )

    print(
        "=" * 90
    )

    print(
        metrics_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    # -------------------------------------------------------------------------
    # Horizon-wise metrics
    # -------------------------------------------------------------------------

    horizon_df = calculate_horizon_metrics(
        actual_df
    )

    horizon_path = (
        OUTPUT_DIR
        / "DAY6_model_comparison_horizon_metrics.csv"
    )

    horizon_df.to_csv(
        horizon_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------------

    summary_df = create_summary_table(
        metrics_df
    )

    summary_path = (
        OUTPUT_DIR
        / "DAY6_model_comparison_summary.csv"
    )

    summary_df.to_csv(
        summary_path
    )

    # -------------------------------------------------------------------------
    # Final output
    # -------------------------------------------------------------------------

    print(
        "\n" + "=" * 90
    )

    print(
        "FILES GENERATED"
    )

    print(
        "=" * 90
    )

    print(
        f"\nOverall metrics:"
    )

    print(
        f"  {metrics_path}"
    )

    print(
        f"\nHorizon-wise metrics:"
    )

    print(
        f"  {horizon_path}"
    )

    print(
        f"\nComparison summary:"
    )

    print(
        f"  {summary_path}"
    )

    print(
        "\n" + "=" * 90
    )

    print(
        "DAY-6 EVALUATION COMPLETE"
    )

    print(
        "=" * 90
    )




if __name__ == "__main__":

    main()