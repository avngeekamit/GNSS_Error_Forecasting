"""
DIRECT LSTM EVALUATION
GNSS Error Forecasting


Training:
    GEO_day6_train.csv

Evaluation:
    GEO_day6_actual.csv

Targets:
    x_error
    y_error
    z_error
    satclockerror

Architecture:
    Past 24 hours (96 points)
            ↓
          LSTM
            ↓
       Dense layers
            ↓
    Direct 95-step forecast
            ↓
       95 × 4 outputs

Same evaluation protocol as: Hybrid XGBoost,Direct GRU
"""

import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch.utils.data import (
    DataLoader,
    TensorDataset
)

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)



# CONFIGURATION


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

TRAIN_PATH = os.path.join(
    BASE_DIR,
    "data",
    "GEO_day6_train.csv"
)

ACTUAL_PATH = os.path.join(
    BASE_DIR,
    "data",
    "GEO_day6_actual.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "results"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "lstm_day6_model.pth"
)

PREDICTION_PATH = os.path.join(
    RESULT_DIR,
    "LSTM_DAY6_predictions.csv"
)

METRICS_PATH = os.path.join(
    RESULT_DIR,
    "LSTM_DAY6_metrics.csv"
)

HORIZON_METRICS_PATH = os.path.join(
    RESULT_DIR,
    "LSTM_DAY6_horizon_metrics.csv"
)

FORECAST_PLOT_PATH = os.path.join(
    RESULT_DIR,
    "LSTM_DAY6_forecast.png"
)

TRAINING_PLOT_PATH = os.path.join(
    RESULT_DIR,
    "LSTM_DAY6_training_curve.png"
)



# MODEL PARAMETERS


TARGETS = [
    "x_error",
    "y_error",
    "z_error",
    "satclockerror"
]

INPUT_SIZE = 4

SEQ_LEN = 96
PRED_LEN = 95

HIDDEN_SIZE = 64
NUM_LAYERS = 2

DROPOUT = 0.20

BATCH_SIZE = 16

LEARNING_RATE = 0.001

MAX_EPOCHS = 500
PATIENCE = 40

VALIDATION_RATIO = 0.20

SEED = 42



# REPRODUCIBILITY

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)



# DEVICE

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 90)
print("DAY-6 DIRECT LSTM EVALUATION")
print("GNSS ERROR FORECASTING")
print("=" * 90)

print(
    f"\nDevice: {DEVICE}"
)


# DIRECTORIES


os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)



# LOAD TRAINING DATA


print("\nLoading GEO_day6_train.csv...")

train_df = pd.read_csv(
    TRAIN_PATH,
    index_col=0,
    parse_dates=True
)

print(
    "\nTraining dataset:"
)

print(
    "Shape:",
    train_df.shape
)

print(
    "Start:",
    train_df.index.min()
)

print(
    "End:",
    train_df.index.max()
)


# =============================================================================
# LOAD ACTUAL DAY-6
# =============================================================================

print(
    "\nLoading GEO_day6_actual.csv..."
)

actual_df = pd.read_csv(
    ACTUAL_PATH,
    index_col=0,
    parse_dates=True
)

print(
    "\nActual Day-6:"
)

print(
    "Shape:",
    actual_df.shape
)

print(
    "Start:",
    actual_df.index.min()
)

print(
    "End:",
    actual_df.index.max()
)


# =============================================================================
# VALIDATION
# =============================================================================

for target in TARGETS:

    if target not in train_df.columns:

        raise ValueError(
            f"Missing {target} in training data."
        )

    if target not in actual_df.columns:

        raise ValueError(
            f"Missing {target} in actual data."
        )


if len(actual_df) != PRED_LEN:

    raise ValueError(
        f"Expected {PRED_LEN} actual observations, "
        f"found {len(actual_df)}."
    )


# =============================================================================
# CHECK TIMESTAMP INTERVAL
# =============================================================================

train_diffs = (
    train_df.index.to_series()
    .diff()
    .dropna()
)

actual_diffs = (
    actual_df.index.to_series()
    .diff()
    .dropna()
)

if not all(
    train_diffs == pd.Timedelta(minutes=15)
):

    print(
        "\nWARNING: Training timestamps are not "
        "all exactly 15 minutes apart."
    )

else:

    print(
        "\nTraining timestamps: OK — 15 minute interval."
    )


if not all(
    actual_diffs == pd.Timedelta(minutes=15)
):

    print(
        "WARNING: Actual timestamps are not "
        "all exactly 15 minutes apart."
    )

else:

    print(
        "Actual timestamps: OK — 15 minute interval."
    )


# =============================================================================
# RAW VALUES
# =============================================================================

train_values = (
    train_df[TARGETS]
    .values
    .astype(np.float32)
)

actual_values = (
    actual_df[TARGETS]
    .values
    .astype(np.float32)
)


# =============================================================================
# STANDARDIZATION
# =============================================================================
#
# IMPORTANT:
# Scaler is fitted ONLY on training data.
#
# Actual Day-6 data is never used to fit the scaler.
# =============================================================================

print(
    "\nFitting StandardScaler on training data only..."
)

scaler = StandardScaler()

train_scaled = scaler.fit_transform(
    train_values
).astype(np.float32)


print("Scaler fitted.")


# =============================================================================
# CREATE DIRECT FORECASTING WINDOWS
# =============================================================================
#
# Input:
#       96 observations = 24 hours
#
# Target:
#       next 95 observations = 23.75 hours
#
# We use overlapping windows inside the training period.
# =============================================================================

X = []
Y = []

max_start = (
    len(train_scaled)
    - SEQ_LEN
    - PRED_LEN
    + 1
)

if max_start <= 0:

    raise ValueError(
        "Training dataset is too small for "
        "the selected sequence and prediction length."
    )


for start in range(max_start):

    input_end = (
        start + SEQ_LEN
    )

    target_end = (
        input_end + PRED_LEN
    )

    X.append(
        train_scaled[
            start:input_end
        ]
    )

    Y.append(
        train_scaled[
            input_end:target_end
        ]
    )


X = np.asarray(
    X,
    dtype=np.float32
)

Y = np.asarray(
    Y,
    dtype=np.float32
)


print("\n" + "=" * 90)
print("DIRECT LSTM TRAINING WINDOWS")
print("=" * 90)

print(
    "Input shape :",
    X.shape
)

print(
    "Target shape:",
    Y.shape
)

print(
    f"Input sequence : {SEQ_LEN} points "
    f"= {SEQ_LEN * 15 / 60:.1f} hours"
)

print(
    f"Output sequence: {PRED_LEN} points "
    f"= {PRED_LEN * 15 / 60:.2f} hours"
)


# =============================================================================
# CHRONOLOGICAL TRAIN / VALIDATION SPLIT
# =============================================================================
#
# IMPORTANT:
# We do NOT randomly split time-series windows.
#
# Earlier windows → training
# Later windows  → validation
# =============================================================================

n_samples = len(X)

val_size = max(
    1,
    int(
        n_samples
        * VALIDATION_RATIO
    )
)

train_size = (
    n_samples
    - val_size
)

X_train = X[
    :train_size
]

Y_train = Y[
    :train_size
]

X_val = X[
    train_size:
]

Y_val = Y[
    train_size:
]


print(
    "\nTraining samples:",
    len(X_train)
)

print(
    "Validation samples:",
    len(X_val)
)


# =============================================================================
# TORCH DATA
# =============================================================================

X_train_tensor = torch.tensor(
    X_train,
    dtype=torch.float32
)

Y_train_tensor = torch.tensor(
    Y_train,
    dtype=torch.float32
)

X_val_tensor = torch.tensor(
    X_val,
    dtype=torch.float32
)

Y_val_tensor = torch.tensor(
    Y_val,
    dtype=torch.float32
)


train_dataset = TensorDataset(
    X_train_tensor,
    Y_train_tensor
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# =============================================================================
# LSTM MODEL
# =============================================================================

class DirectLSTM(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        pred_len,
        output_size,
        dropout
    ):

        super().__init__()

        self.pred_len = pred_len

        self.output_size = output_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=(
                dropout
                if num_layers > 1
                else 0.0
            )
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.fc = nn.Sequential(

            nn.Linear(
                hidden_size,
                hidden_size
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_size,
                pred_len * output_size
            )
        )


    def forward(self, x):

        output, (hidden, cell) = (
            self.lstm(x)
        )

        # Last temporal representation
        last_output = (
            output[:, -1, :]
        )

        last_output = (
            self.dropout(
                last_output
            )
        )

        prediction = self.fc(
            last_output
        )

        prediction = prediction.view(
            -1,
            self.pred_len,
            self.output_size
        )

        return prediction


# =============================================================================
# INITIALIZE MODEL
# =============================================================================

model = DirectLSTM(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    pred_len=PRED_LEN,
    output_size=INPUT_SIZE,
    dropout=DROPOUT
).to(DEVICE)


print("\n" + "=" * 90)
print("LSTM MODEL")
print("=" * 90)

print(model)

parameter_count = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(
    "\nTrainable parameters:",
    parameter_count
)


# =============================================================================
# LOSS / OPTIMIZER
# =============================================================================

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-5
)

scheduler = (
    torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10
    )
)


# =============================================================================
# TRAINING
# =============================================================================

print("\n" + "=" * 90)
print("TRAINING LSTM")
print("=" * 90)

best_val_loss = float("inf")

best_state = None

patience_counter = 0

train_losses = []

val_losses = []


for epoch in range(
    1,
    MAX_EPOCHS + 1
):

    # -------------------------------------------------------------------------
    # TRAIN
    # -------------------------------------------------------------------------

    model.train()

    running_loss = 0.0

    for batch_x, batch_y in train_loader:

        batch_x = batch_x.to(
            DEVICE
        )

        batch_y = batch_y.to(
            DEVICE
        )

        optimizer.zero_grad()

        prediction = model(
            batch_x
        )

        loss = criterion(
            prediction,
            batch_y
        )

        loss.backward()

        # Prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        running_loss += (
            loss.item()
            * len(batch_x)
        )


    train_loss = (
        running_loss
        / len(train_dataset)
    )


    # -------------------------------------------------------------------------
    # VALIDATION
    # -------------------------------------------------------------------------

    model.eval()

    with torch.no_grad():

        val_prediction = model(
            X_val_tensor.to(
                DEVICE
            )
        )

        val_loss = criterion(
            val_prediction,
            Y_val_tensor.to(
                DEVICE
            )
        ).item()


    scheduler.step(
        val_loss
    )


    train_losses.append(
        train_loss
    )

    val_losses.append(
        val_loss
    )


    # -------------------------------------------------------------------------
    # SAVE BEST MODEL
    # -------------------------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        best_state = {
            key: value.detach()
            .cpu()
            .clone()

            for key, value
            in model.state_dict().items()
        }

        patience_counter = 0

    else:

        patience_counter += 1


    # -------------------------------------------------------------------------
    # LOG
    # -------------------------------------------------------------------------

    if (
        epoch == 1
        or epoch % 10 == 0
    ):

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )


    # -------------------------------------------------------------------------
    # EARLY STOPPING
    # -------------------------------------------------------------------------

    if (
        patience_counter
        >= PATIENCE
    ):

        print(
            f"\nEarly stopping at "
            f"epoch {epoch}."
        )

        break


# =============================================================================
# RESTORE BEST MODEL
# =============================================================================

if best_state is None:

    raise RuntimeError(
        "No best model state was saved."
    )


model.load_state_dict(
    best_state
)


torch.save(
    model.state_dict(),
    MODEL_PATH
)


print(
    "\nBest LSTM model saved:"
)

print(
    MODEL_PATH
)


# =============================================================================
# TRAINING CURVE
# =============================================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    train_losses,
    label="Training Loss"
)

plt.plot(
    val_losses,
    label="Validation Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "MSE Loss"
)

plt.title(
    "Direct LSTM Training / Validation Loss"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    TRAINING_PLOT_PATH,
    dpi=200
)

plt.close()


# =============================================================================
# DAY-6 FORECAST
# =============================================================================

print("\n" + "=" * 90)
print("GENERATING DAY-6 FORECAST")
print("=" * 90)


model.eval()


# Last 96 observations available before Day-6
input_sequence = (
    train_scaled[-SEQ_LEN:]
)


input_tensor = torch.tensor(
    input_sequence,
    dtype=torch.float32
).unsqueeze(0).to(
    DEVICE
)


with torch.no_grad():

    forecast_scaled = (
        model(
            input_tensor
        )
        .cpu()
        .numpy()[0]
    )


print(
    "Scaled forecast shape:",
    forecast_scaled.shape
)


# =============================================================================
# INVERSE SCALE
# =============================================================================

forecast = scaler.inverse_transform(
    forecast_scaled
)


print(
    "Original-scale forecast shape:",
    forecast.shape
)


# =============================================================================
# CHECK FORECAST / ACTUAL LENGTH
# =============================================================================

if len(forecast) != len(
    actual_values
):

    raise ValueError(
        "Forecast and actual lengths "
        "do not match."
    )


# =============================================================================
# FORECAST HORIZON
# =============================================================================

horizon_minutes = (
    np.arange(
        1,
        PRED_LEN + 1
    )
    * 15
)


# =============================================================================
# SAVE PREDICTIONS
# =============================================================================

prediction_df = pd.DataFrame(
    forecast,
    columns=TARGETS
)

prediction_df.insert(
    0,
    "forecast_horizon_minutes",
    horizon_minutes
)

prediction_df.insert(
    1,
    "timestamp",
    actual_df.index
)

prediction_df.to_csv(
    PREDICTION_PATH,
    index=False
)


print(
    "\nPredictions saved:"
)

print(
    PREDICTION_PATH
)


# =============================================================================
# METRIC FUNCTION
# =============================================================================

def calculate_metrics(
    actual,
    predicted
):

    error = (
        predicted
        - actual
    )

    absolute_error = np.abs(
        error
    )

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    r2 = r2_score(
        actual,
        predicted
    )

    max_ae = np.max(
        absolute_error
    )

    p95_ae = np.percentile(
        absolute_error,
        95
    )

    bias = np.mean(
        error
    )

    if (
        np.std(actual) == 0
        or np.std(predicted) == 0
    ):

        correlation = np.nan

    else:

        correlation = np.corrcoef(
            actual,
            predicted
        )[0, 1]


    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MaxAE": max_ae,
        "P95_AE": p95_ae,
        "Bias": bias,
        "Correlation": correlation
    }


# =============================================================================
# OVERALL METRICS
# =============================================================================

metrics_rows = []


for i, target in enumerate(
    TARGETS
):

    metrics = calculate_metrics(
        actual_values[:, i],
        forecast[:, i]
    )

    metrics_rows.append({

        "target": target,

        "forecast_points": PRED_LEN,

        **metrics
    })


metrics_df = pd.DataFrame(
    metrics_rows
)


print("\n" + "=" * 90)
print("LSTM DAY-6 RESULTS")
print("=" * 90)

print(
    metrics_df.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.6f}"
    )
)


metrics_df.to_csv(
    METRICS_PATH,
    index=False
)


print(
    "\nMetrics saved:"
)

print(
    METRICS_PATH
)


# =============================================================================
# HORIZON-WISE ERROR
# =============================================================================

horizon_rows = []


for i in range(
    PRED_LEN
):

    row = {

        "horizon_minutes":
            int(
                horizon_minutes[i]
            )
    }


    for j, target in enumerate(
        TARGETS
    ):

        absolute_error = abs(
            forecast[i, j]
            - actual_values[i, j]
        )

        row[
            f"{target}_AE"
        ] = absolute_error


    horizon_rows.append(
        row
    )


horizon_df = pd.DataFrame(
    horizon_rows
)


# =============================================================================
# CUMULATIVE HORIZON MAE
# =============================================================================

for target in TARGETS:

    horizon_df[
        f"{target}_cumulative_MAE"
    ] = (
        horizon_df[
            f"{target}_AE"
        ]
        .expanding()
        .mean()
    )


horizon_df.to_csv(
    HORIZON_METRICS_PATH,
    index=False
)


print(
    "\nHorizon metrics saved:"
)

print(
    HORIZON_METRICS_PATH
)


# =============================================================================
# FORECAST VISUALIZATION
# =============================================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(16, 10)
)

axes = axes.flatten()


time_hours = (
    horizon_minutes / 60
)


for i, target in enumerate(
    TARGETS
):

    ax = axes[i]

    ax.plot(
        time_hours,
        actual_values[:, i],
        label="Actual"
    )

    ax.plot(
        time_hours,
        forecast[:, i],
        label="LSTM Forecast"
    )

    ax.set_title(
        target
    )

    ax.set_xlabel(
        "Forecast Horizon (hours)"
    )

    ax.set_ylabel(
        "GNSS Error"
    )

    ax.grid(True)

    ax.legend()


fig.suptitle(
    "Direct LSTM — Day-6 GNSS Error Forecast",
    fontsize=16
)


plt.tight_layout(
    rect=[
        0,
        0,
        1,
        0.96
    ]
)


plt.savefig(
    FORECAST_PLOT_PATH,
    dpi=200
)

plt.close()


print(
    "\nForecast plot saved:"
)

print(
    FORECAST_PLOT_PATH
)


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 90)
print("LSTM DAY-6 EVALUATION COMPLETE")
print("=" * 90)

print(
    "\nTraining:"
)

print(
    f"{train_df.index.min()} → "
    f"{train_df.index.max()}"
)

print(
    "\nActual Day-6:"
)

print(
    f"{actual_df.index.min()} → "
    f"{actual_df.index.max()}"
)

print(
    f"\nInput sequence: "
    f"{SEQ_LEN} points "
    f"({SEQ_LEN * 15 / 60:.1f} hours)"
)

print(
    f"Direct forecast: "
    f"{PRED_LEN} points "
    f"({PRED_LEN * 15 / 60:.2f} hours)"
)

print(
    "\nFiles generated:"
)

print(
    "Model:",
    MODEL_PATH
)

print(
    "Predictions:",
    PREDICTION_PATH
)

print(
    "Metrics:",
    METRICS_PATH
)

print(
    "Horizon metrics:",
    HORIZON_METRICS_PATH
)

print(
    "Forecast plot:",
    FORECAST_PLOT_PATH
)

print(
    "Training curve:",
    TRAINING_PLOT_PATH
)

print(
    "\nDone."
)