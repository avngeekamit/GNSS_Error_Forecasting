import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

warnings.filterwarnings("ignore")

# HYBRID XGBOOST DAY-6
# Direct XGBoost:  X, Y
# Recursive XGBoost: Z, Clock
# Day-6 forecasting and evaluation

print("=" * 100)
print("B12 — HYBRID XGBOOST DAY-6 EVALUATION")
print("Direct XGBoost (X/Y) + Recursive XGBoost (Z/Clock)")
print("Actual Day-6 ground-truth evaluation")
print("=" * 100)



# PATHS


from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"
RESULT_DIR = BASE_DIR / "results" / "hybrid_xgboost"
PLOTS_DIR = BASE_DIR / "figures" / "forecasts" / "hybrid_xgboost"

TRAIN_PATH = DATA_DIR / "GEO_day6_train.csv"
ACTUAL_PATH = DATA_DIR / "GEO_day6_actual.csv"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# TARGETS


TARGETS = [
    "x_error",
    "y_error",
    "z_error",
    "satclockerror"
]



# FORECAST CONFIGURATION


INTERVAL_MINUTES = 15

FORECAST_STEPS = 95



# FEATURE CONFIGURATION
# SAME AS DIRECT + RECURSIVE B12

LAGS = list(range(1, 9))

ROLLING_WINDOWS = [
    4,
    8
]



# XGBOOST PARAMETERS
# SAME FOR BOTH STRATEGIES

XGB_PARAMS = {

    "n_estimators": 400,

    "max_depth": 5,

    "learning_rate": 0.03,

    "subsample": 0.8,

    "colsample_bytree": 0.8,

    "objective": "reg:squarederror",

    "random_state": 42,

    "n_jobs": -1
}


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading training data...")

train_df = pd.read_csv(
    TRAIN_PATH,
    index_col=0,
    parse_dates=True
).sort_index()

print(
    "Training:",
    train_df.shape
)

print(
    train_df.index.min(),
    "→",
    train_df.index.max()
)


print("\nLoading actual Day-6...")

actual_df = pd.read_csv(
    ACTUAL_PATH,
    index_col=0,
    parse_dates=True
).sort_index()

print(
    "Actual:",
    actual_df.shape
)

print(
    actual_df.index.min(),
    "→",
    actual_df.index.max()
)


# ============================================================
# VALIDATION
# ============================================================

if len(train_df) != 552:

    raise ValueError(
        f"Expected 552 training rows, "
        f"found {len(train_df)}"
    )


if len(actual_df) != 95:

    raise ValueError(
        f"Expected 95 actual rows, "
        f"found {len(actual_df)}"
    )


for target in TARGETS:

    if target not in train_df.columns:

        raise ValueError(
            f"{target} missing from training data"
        )

    if target not in actual_df.columns:

        raise ValueError(
            f"{target} missing from actual data"
        )


# ============================================================
# TEMPORAL ALIGNMENT
# ============================================================

expected_first = (
    train_df.index[-1]
    + pd.Timedelta(minutes=15)
)

expected_last = (
    train_df.index[-1]
    + pd.Timedelta(minutes=15 * 95)
)


print("\n" + "=" * 100)
print("DAY-6 ALIGNMENT")
print("=" * 100)

print(
    "Training cutoff:",
    train_df.index[-1]
)

print(
    "Expected first:",
    expected_first
)

print(
    "Actual first:",
    actual_df.index[0]
)

print(
    "Expected last:",
    expected_last
)

print(
    "Actual last:",
    actual_df.index[-1]
)


if actual_df.index[0] != expected_first:

    raise ValueError(
        "First actual timestamp is not "
        "15 minutes after training cutoff."
    )


if actual_df.index[-1] != expected_last:

    raise ValueError(
        "Last actual timestamp is not "
        "1425 minutes after training cutoff."
    )


future_timestamps = actual_df.index.copy()


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(
    df,
    target
):

    temp = df.copy()

    # --------------------------------------------------------
    # Target history
    # --------------------------------------------------------

    for lag in LAGS:

        temp[
            f"lag_{lag}"
        ] = (
            temp[target]
            .shift(lag)
        )

    # --------------------------------------------------------
    # Differences
    # --------------------------------------------------------

    temp["diff_1"] = (
        temp[target]
        .diff(1)
    )

    temp["diff_4"] = (
        temp[target]
        .diff(4)
    )

    # --------------------------------------------------------
    # Rolling statistics
    # --------------------------------------------------------

    for window in ROLLING_WINDOWS:

        temp[
            f"rolling_mean_{window}"
        ] = (
            temp[target]
            .rolling(window)
            .mean()
        )

        temp[
            f"rolling_std_{window}"
        ] = (
            temp[target]
            .rolling(window)
            .std()
        )

    # --------------------------------------------------------
    # Time features
    # --------------------------------------------------------

    temp["hour"] = (
        temp.index.hour
    )

    temp["minute"] = (
        temp.index.minute
    )

    temp["dayofweek"] = (
        temp.index.dayofweek
    )

    temp["hour_sin"] = np.sin(
        2 * np.pi *
        temp.index.hour / 24
    )

    temp["hour_cos"] = np.cos(
        2 * np.pi *
        temp.index.hour / 24
    )

    temp["minute_sin"] = np.sin(
        2 * np.pi *
        temp.index.minute / 60
    )

    temp["minute_cos"] = np.cos(
        2 * np.pi *
        temp.index.minute / 60
    )

    return temp


# ============================================================
# FEATURES
# ============================================================

ERROR_FEATURES = [

    f"lag_{lag}"

    for lag in LAGS
]


ERROR_FEATURES += [
    "diff_1",
    "diff_4"
]


for window in ROLLING_WINDOWS:

    ERROR_FEATURES += [

        f"rolling_mean_{window}",

        f"rolling_std_{window}"
    ]


TIME_FEATURES = [

    "hour",
    "minute",
    "dayofweek",
    "hour_sin",
    "hour_cos",
    "minute_sin",
    "minute_cos"
]


FEATURE_COLUMNS = (
    ERROR_FEATURES
    + TIME_FEATURES
)


print("\n" + "=" * 100)
print("FEATURE CONFIGURATION")
print("=" * 100)

print(
    "History features:",
    len(ERROR_FEATURES)
)

print(
    "Time features:",
    len(TIME_FEATURES)
)

print(
    "Total:",
    len(FEATURE_COLUMNS)
)

assert len(FEATURE_COLUMNS) == 21


# ============================================================
# MODEL STRATEGIES
# ============================================================

DIRECT_TARGETS = [
    "x_error",
    "y_error"
]

RECURSIVE_TARGETS = [
    "z_error",
    "satclockerror"
]


print("\n" + "=" * 100)
print("HYBRID STRATEGY")
print("=" * 100)

print(
    "Direct XGBoost:",
    ", ".join(DIRECT_TARGETS)
)

print(
    "Recursive XGBoost:",
    ", ".join(RECURSIVE_TARGETS)
)


# ============================================================
# DIRECT XGBOOST
# X + Y
# ============================================================

direct_predictions = {}


for target in DIRECT_TARGETS:

    print("\n" + "=" * 90)
    print(
        "DIRECT XGBOOST:",
        target
    )
    print("=" * 90)

    historical_features = create_features(
        train_df,
        target
    )

    predictions = []

    # One independent model per horizon
    for horizon_step in range(
        1,
        FORECAST_STEPS + 1
    ):

        horizon_index = (
            horizon_step - 1
        )

        minutes = (
            horizon_step
            * INTERVAL_MINUTES
        )

        print(
            f"[{target}] "
            f"+{minutes:4d} min "
            f"({horizon_index + 1:02d}/95)"
        )

        supervised = (
            historical_features.copy()
        )

        supervised[
            "future_target"
        ] = (
            supervised[target]
            .shift(-horizon_step)
        )

        supervised = (
            supervised
            .dropna()
        )

        X_train = (
            supervised[
                FEATURE_COLUMNS
            ]
        )

        y_train = (
            supervised[
                "future_target"
            ]
        )

        model = XGBRegressor(
            **XGB_PARAMS
        )

        model.fit(
            X_train,
            y_train,
            verbose=False
        )

        # ----------------------------------------------------
        # Final historical feature vector
        # ----------------------------------------------------

        last_features = (
            historical_features
            .iloc[-1]
            .copy()
        )

        future_timestamp = (
            future_timestamps[
                horizon_index
            ]
        )

        # Future time features
        last_features["hour"] = (
            future_timestamp.hour
        )

        last_features["minute"] = (
            future_timestamp.minute
        )

        last_features["dayofweek"] = (
            future_timestamp.dayofweek
        )

        last_features["hour_sin"] = np.sin(
            2 * np.pi *
            future_timestamp.hour / 24
        )

        last_features["hour_cos"] = np.cos(
            2 * np.pi *
            future_timestamp.hour / 24
        )

        last_features["minute_sin"] = np.sin(
            2 * np.pi *
            future_timestamp.minute / 60
        )

        last_features["minute_cos"] = np.cos(
            2 * np.pi *
            future_timestamp.minute / 60
        )

        X_future = pd.DataFrame(
            [last_features[
                FEATURE_COLUMNS
            ].values],
            columns=FEATURE_COLUMNS
        )

        prediction = (
            model
            .predict(X_future)[0]
        )

        predictions.append(
            prediction
        )

    direct_predictions[target] = (
        np.array(predictions)
    )


# ============================================================
# RECURSIVE XGBOOST
# Z + CLOCK
# ============================================================

recursive_models = {}

for target in RECURSIVE_TARGETS:

    print("\n" + "=" * 90)
    print(
        "TRAINING RECURSIVE MODEL:",
        target
    )
    print("=" * 90)

    feature_data = create_features(
        train_df,
        target
    ).dropna()

    X_train = (
        feature_data[
            FEATURE_COLUMNS
        ]
    )

    y_train = (
        feature_data[target]
    )

    print(
        "Training samples:",
        len(X_train)
    )

    model = XGBRegressor(
        **XGB_PARAMS
    )

    model.fit(
        X_train,
        y_train,
        verbose=False
    )

    recursive_models[target] = model

    print("✅ Model trained")


# ============================================================
# RECURSIVE FORECAST
# ============================================================

recursive_predictions = {

    target: []

    for target in RECURSIVE_TARGETS
}


# Separate history so direct training data
# is never modified.

recursive_history = (
    train_df.copy()
)


print("\n" + "=" * 100)
print("RECURSIVE FORECAST")
print("=" * 100)


for step in range(
    FORECAST_STEPS
):

    future_timestamp = (
        recursive_history.index[-1]
        + pd.Timedelta(minutes=15)
    )

    print(
        f"Step {step + 1:02d}/95"
        f" → {future_timestamp}"
    )

    new_row = {}

    for target in RECURSIVE_TARGETS:

        feature_data = create_features(
            recursive_history,
            target
        )

        last_features = (
            feature_data
            .dropna()
            .iloc[-1]
            .copy()
        )

        # Future time features
        last_features["hour"] = (
            future_timestamp.hour
        )

        last_features["minute"] = (
            future_timestamp.minute
        )

        last_features["dayofweek"] = (
            future_timestamp.dayofweek
        )

        last_features["hour_sin"] = np.sin(
            2 * np.pi *
            future_timestamp.hour / 24
        )

        last_features["hour_cos"] = np.cos(
            2 * np.pi *
            future_timestamp.hour / 24
        )

        last_features["minute_sin"] = np.sin(
            2 * np.pi *
            future_timestamp.minute / 60
        )

        last_features["minute_cos"] = np.cos(
            2 * np.pi *
            future_timestamp.minute / 60
        )

        X_future = pd.DataFrame(
            [last_features[
                FEATURE_COLUMNS
            ].values],
            columns=FEATURE_COLUMNS
        )

        prediction = (
            recursive_models[target]
            .predict(X_future)[0]
        )

        new_row[target] = prediction

        recursive_predictions[
            target
        ].append(
            prediction
        )

    # --------------------------------------------------------
    # IMPORTANT:
    # Feed predictions back into history.
    # --------------------------------------------------------

    recursive_history.loc[
        future_timestamp
    ] = new_row


# ============================================================
# COMBINE HYBRID PREDICTIONS
# ============================================================

hybrid_predictions = {

    "x_error":
        direct_predictions[
            "x_error"
        ],

    "y_error":
        direct_predictions[
            "y_error"
        ],

    "z_error":
        recursive_predictions[
            "z_error"
        ],

    "satclockerror":
        recursive_predictions[
            "satclockerror"
        ]
}


hybrid_df = pd.DataFrame(
    hybrid_predictions,
    index=future_timestamps
)

hybrid_df.index.name = "timestamp"

hybrid_df.insert(
    0,
    "forecast_horizon_minutes",
    np.arange(
        1,
        FORECAST_STEPS + 1
    ) * INTERVAL_MINUTES
)


# ============================================================
# DAY-6 ALIGNMENT
# ============================================================

assert len(hybrid_df) == 95
assert len(actual_df) == 95

print("\n" + "=" * 100)
print("FINAL DAY-6 ALIGNMENT")
print("=" * 100)
print("Hybrid:", hybrid_df.index[0], "→", hybrid_df.index[-1])
print("Actual:", actual_df.index[0], "→", actual_df.index[-1])

assert hybrid_df.index[0] == actual_df.index[0]
assert hybrid_df.index[-1] == actual_df.index[-1]


# ============================================================
# METRICS — HYBRID XGBOOST VS ACTUAL DAY-6
# ============================================================

results = []

for target in TARGETS:

    actual = actual_df[target].values
    hybrid_pred = hybrid_df[target].values

    hybrid_mae = mean_absolute_error(
        actual,
        hybrid_pred
    )

    hybrid_rmse = np.sqrt(
        mean_squared_error(
            actual,
            hybrid_pred
        )
    )

    hybrid_r2 = r2_score(
        actual,
        hybrid_pred
    )

    results.append({

        "target":
            target,

        "Hybrid_XGBoost_MAE":
            hybrid_mae,

        "Hybrid_XGBoost_RMSE":
            hybrid_rmse,

        "Hybrid_XGBoost_R2":
            hybrid_r2
    })


metrics_df = pd.DataFrame(results)


# ============================================================
# SAVE
# ============================================================

prediction_path = (
    RESULT_DIR
    / "b12_day6_hybrid_predictions.csv"
)

metrics_path = (
    RESULT_DIR
    / "b12_day6_hybrid_metrics.csv"
)

hybrid_df.to_csv(
    prediction_path
)

metrics_df.to_csv(
    metrics_path,
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 100)
print("HYBRID XGBOOST — DAY-6 RESULTS")
print("=" * 100)

print(
    metrics_df.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.6f}"
    )
)


# ============================================================
# TRAJECTORY PLOTS — HYBRID VS ACTUAL
# ============================================================

for target in TARGETS:

    plt.figure(
        figsize=(15, 6)
    )

    plt.plot(
        actual_df.index,
        actual_df[target],
        label="Actual",
        linewidth=2
    )

    plt.plot(
        hybrid_df.index,
        hybrid_df[target],
        label="Hybrid XGBoost"
    )

    plt.title(
        f"Day-6 — {target}"
    )

    plt.xlabel(
        "Time"
    )

    plt.ylabel(
        "Error"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.xticks(
        rotation=30
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR / f"{target}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 100)
print("B12 HYBRID DAY-6 EVALUATION COMPLETE")
print("=" * 100)

print(
    "\nPredictions:",
    prediction_path
)

print(
    "Metrics:",
    metrics_path
)

print(
    "Plots:",
    PLOTS_DIR
)

print("\n" + "=" * 100)