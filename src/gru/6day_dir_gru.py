"""
================================================================================
DAY-6 DIRECT GRU EVALUATION
GNSS Error Forecasting
================================================================================

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
          GRU
            ↓
       Dense layers
            ↓
    Direct 95-step forecast
            ↓
       95 × 4 outputs

IMPORTANT:
- Scaler is fitted ONLY on training data.
- Day-6 actual observations are NEVER used for training.
# Uses the same 552 → 95 train/Day-6 evaluation split
# as the other forecasting models.
================================================================================
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results")

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "gru_day6_model.pth"
)

PREDICTION_PATH = os.path.join(
    RESULT_DIR,
    "GRU_DAY6_predictions.csv"
)

METRICS_PATH = os.path.join(
    RESULT_DIR,
    "GRU_DAY6_metrics.csv"
)

HORIZON_METRICS_PATH = os.path.join(
    RESULT_DIR,
    "GRU_DAY6_horizon_metrics.csv"
)

PLOT_PATH = os.path.join(
    RESULT_DIR,
    "GRU_DAY6_forecast.png"
)


# =============================================================================
# HYPERPARAMETERS
# =============================================================================

TARGETS = [
    "x_error",
    "y_error",
    "z_error",
    "satclockerror"
]

SEQ_LEN = 96          # 24 hours at 15-minute intervals
PRED_LEN = 95         # Actual Day-6 has 95 points

INPUT_SIZE = 4
HIDDEN_SIZE = 64
NUM_LAYERS = 2

DROPOUT = 0.20

BATCH_SIZE = 16
LEARNING_RATE = 0.001

MAX_EPOCHS = 500
PATIENCE = 40

VALIDATION_RATIO = 0.20

SEED = 42


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 90)
print("DAY-6 DIRECT GRU EVALUATION")
print("GNSS ERROR FORECASTING")
print("=" * 90)

print(f"\nDevice: {DEVICE}")


# =============================================================================
# CREATE DIRECTORIES
# =============================================================================

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# =============================================================================
# LOAD DATA
# =============================================================================

print("\nLoading training data...")
train_df = pd.read_csv(
    TRAIN_PATH,
    index_col=0,
    parse_dates=True
)

print("Training shape:", train_df.shape)
print("Training start:", train_df.index.min())
print("Training end:", train_df.index.max())

print("\nLoading Day-6 actual data...")
actual_df = pd.read_csv(
    ACTUAL_PATH,
    index_col=0,
    parse_dates=True
)

print("Actual shape:", actual_df.shape)
print("Actual start:", actual_df.index.min())
print("Actual end:", actual_df.index.max())


# =============================================================================
# VALIDATE DATA
# =============================================================================

for col in TARGETS:

    if col not in train_df.columns:
        raise ValueError(
            f"Missing target '{col}' in training dataset."
        )

    if col not in actual_df.columns:
        raise ValueError(
            f"Missing target '{col}' in actual dataset."
        )


print("\n" + "=" * 90)
print("DAY-6 SPLIT")
print("=" * 90)

print(
    f"Training observations : {len(train_df)}"
)

print(
    f"Actual observations   : {len(actual_df)}"
)

if len(actual_df) != PRED_LEN:

    raise ValueError(
        f"Expected {PRED_LEN} actual observations, "
        f"but found {len(actual_df)}."
    )


# =============================================================================
# PREPARE RAW VALUES
# =============================================================================

train_values = train_df[TARGETS].values.astype(np.float32)

actual_values = actual_df[TARGETS].values.astype(np.float32)


# =============================================================================
# SCALE DATA
# =============================================================================

print("\nFitting StandardScaler on TRAINING DATA ONLY...")

scaler = StandardScaler()

train_scaled = scaler.fit_transform(
    train_values
).astype(np.float32)

print("Scaler fitted.")


# =============================================================================
# BUILD DIRECT FORECAST TRAINING WINDOWS
# =============================================================================
#
# Because we only have 552 observations and want a 95-step forecast,
# the training samples use:
#
#   96 historical points
#       ↓
#   next 95 points
#
# This gives multiple overlapping supervised samples.
#
# =============================================================================

X = []
Y = []

max_start = len(train_scaled) - SEQ_LEN - PRED_LEN + 1

if max_start <= 0:

    raise ValueError(
        "Not enough training data for "
        f"SEQ_LEN={SEQ_LEN}, PRED_LEN={PRED_LEN}."
    )


for start in range(max_start):

    input_end = start + SEQ_LEN

    target_end = input_end + PRED_LEN

    X.append(
        train_scaled[start:input_end]
    )

    Y.append(
        train_scaled[input_end:target_end]
    )


X = np.asarray(X, dtype=np.float32)
Y = np.asarray(Y, dtype=np.float32)


print("\n" + "=" * 90)
print("DIRECT SUPERVISED DATA")
print("=" * 90)

print("Input shape :", X.shape)
print("Target shape:", Y.shape)

print(
    f"Sequence length : {SEQ_LEN} "
    f"points = {SEQ_LEN * 15 / 60:.1f} hours"
)

print(
    f"Prediction length: {PRED_LEN} "
    f"points = {PRED_LEN * 15 / 60:.2f} hours"
)


# =============================================================================
# TRAIN / VALIDATION SPLIT
# =============================================================================
#
# Chronological split.
# No random shuffling across time.
#
# =============================================================================

n_samples = len(X)

val_size = max(
    1,
    int(n_samples * VALIDATION_RATIO)
)

train_size = n_samples - val_size

X_train = X[:train_size]
Y_train = Y[:train_size]

X_val = X[train_size:]
Y_val = Y[train_size:]


print("\nTraining samples  :", len(X_train))
print("Validation samples:", len(X_val))


# =============================================================================
# TORCH DATASETS
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
# MODEL
# =============================================================================

class DirectGRU(nn.Module):

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

        self.gru = nn.GRU(
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

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Sequential(

            nn.Linear(
                hidden_size,
                hidden_size
            ),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(
                hidden_size,
                pred_len * output_size
            )
        )

    def forward(self, x):

        output, hidden = self.gru(x)

        # Last temporal representation
        last_hidden = output[:, -1, :]

        last_hidden = self.dropout(
            last_hidden
        )

        prediction = self.fc(
            last_hidden
        )

        prediction = prediction.view(
            -1,
            self.pred_len,
            self.output_size
        )

        return prediction


model = DirectGRU(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    pred_len=PRED_LEN,
    output_size=INPUT_SIZE,
    dropout=DROPOUT
).to(DEVICE)


print("\n" + "=" * 90)
print("GRU MODEL")
print("=" * 90)

print(model)

print(
    "\nTrainable parameters:",
    sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
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

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=10
)


# =============================================================================
# TRAINING
# =============================================================================

print("\n" + "=" * 90)
print("TRAINING")
print("=" * 90)

best_val_loss = float("inf")

best_state = None

patience_counter = 0

train_losses = []
val_losses = []


for epoch in range(1, MAX_EPOCHS + 1):

    # -------------------------------------------------------------------------
    # TRAIN
    # -------------------------------------------------------------------------

    model.train()

    running_loss = 0.0

    for batch_x, batch_y in train_loader:

        batch_x = batch_x.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        optimizer.zero_grad()

        predictions = model(batch_x)

        loss = criterion(
            predictions,
            batch_y
        )

        loss.backward()

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

        val_predictions = model(
            X_val_tensor.to(DEVICE)
        )

        val_loss = criterion(
            val_predictions,
            Y_val_tensor.to(DEVICE)
        ).item()

    scheduler.step(val_loss)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    # -------------------------------------------------------------------------
    # EARLY STOPPING
    # -------------------------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        best_state = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        }

        patience_counter = 0

    else:

        patience_counter += 1

    # -------------------------------------------------------------------------
    # LOGGING
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

    if patience_counter >= PATIENCE:

        print(
            f"\nEarly stopping at epoch {epoch}."
        )

        break


# =============================================================================
# RESTORE BEST MODEL
# =============================================================================

if best_state is None:

    raise RuntimeError(
        "No valid model state was saved."
    )


model.load_state_dict(best_state)

torch.save(
    model.state_dict(),
    MODEL_PATH
)

print(
    f"\nBest model saved to:\n{MODEL_PATH}"
)


# =============================================================================
# TRAINING CURVE
# =============================================================================

plt.figure(figsize=(10, 5))

plt.plot(
    train_losses,
    label="Training Loss"
)

plt.plot(
    val_losses,
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")

plt.title(
    "Direct GRU Training / Validation Loss"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

loss_plot_path = os.path.join(
    RESULT_DIR,
    "GRU_DAY6_training_curve.png"
)

plt.savefig(
    loss_plot_path,
    dpi=200
)

plt.close()


# =============================================================================
# DAY-6 FORECAST
# =============================================================================

print("\n" + "=" * 90)
print("DAY-6 FORECAST")
print("=" * 90)


model.eval()


# Last 96 observations from the training split
input_sequence = train_scaled[-SEQ_LEN:]

input_tensor = torch.tensor(
    input_sequence,
    dtype=torch.float32
).unsqueeze(0).to(DEVICE)


with torch.no_grad():

    forecast_scaled = model(
        input_tensor
    ).cpu().numpy()[0]


# =============================================================================
# INVERSE TRANSFORM
# =============================================================================

forecast = scaler.inverse_transform(
    forecast_scaled
)


# =============================================================================
# ALIGNMENT
# =============================================================================

print("\nForecast shape:", forecast.shape)

print(
    "Forecast points:",
    len(forecast)
)

print(
    "Actual points:",
    len(actual_values)
)


if len(forecast) != len(actual_values):

    raise ValueError(
        "Forecast and actual lengths do not match."
    )


# =============================================================================
# SAVE PREDICTIONS
# =============================================================================

horizon_minutes = (
    np.arange(1, PRED_LEN + 1) * 15
)

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
    f"\nPredictions saved to:\n"
    f"{PREDICTION_PATH}"
)


# =============================================================================
# METRIC FUNCTIONS
# =============================================================================

def calculate_metrics(
    actual,
    predicted
):

    error = predicted - actual

    abs_error = np.abs(error)

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
        abs_error
    )

    p95_ae = np.percentile(
        abs_error,
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


for i, target in enumerate(TARGETS):

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
print("GRU DAY-6 RESULTS")
print("=" * 90)

print(
    metrics_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


metrics_df.to_csv(
    METRICS_PATH,
    index=False
)


print(
    f"\nMetrics saved to:\n"
    f"{METRICS_PATH}"
)


# =============================================================================
# HORIZON-WISE MAE
# =============================================================================
#
# Calculate cumulative/segment error to understand how performance changes
# throughout the 24-hour horizon.
#
# =============================================================================

horizon_rows = []


for i in range(PRED_LEN):

    row = {

        "horizon_minutes":
            int(horizon_minutes[i])
    }

    for j, target in enumerate(TARGETS):

        absolute_error = abs(
            forecast[i, j]
            - actual_values[i, j]
        )

        row[
            f"{target}_AE"
        ] = absolute_error

    horizon_rows.append(row)


horizon_df = pd.DataFrame(
    horizon_rows
)


# Add cumulative MAE
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
    f"\nHorizon metrics saved to:\n"
    f"{HORIZON_METRICS_PATH}"
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


for i, target in enumerate(TARGETS):

    ax = axes[i]

    ax.plot(
        time_hours,
        actual_values[:, i],
        label="Actual"
    )

    ax.plot(
        time_hours,
        forecast[:, i],
        label="GRU Forecast"
    )

    ax.set_title(
        target
    )

    ax.set_xlabel(
        "Forecast Horizon (hours)"
    )

    ax.set_ylabel(
        "Error"
    )

    ax.grid(True)

    ax.legend()


fig.suptitle(
    "Direct GRU — Day-6 GNSS Error Forecast",
    fontsize=16
)

plt.tight_layout(
    rect=[0, 0, 1, 0.96]
)

plt.savefig(
    PLOT_PATH,
    dpi=200
)

plt.close()


print(
    f"\nForecast plot saved to:\n"
    f"{PLOT_PATH}"
)


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 90)
print("GRU DAY-6 EVALUATION COMPLETE")
print("=" * 90)

print(
    "\nTraining data:"
)

print(
    f"  {train_df.index.min()} → "
    f"{train_df.index.max()}"
)

print(
    "\nEvaluation data:"
)

print(
    f"  {actual_df.index.min()} → "
    f"{actual_df.index.max()}"
)

print(
    f"\nInput sequence: {SEQ_LEN} points "
    f"({SEQ_LEN * 15 / 60:.1f} hours)"
)

print(
    f"Direct forecast: {PRED_LEN} points "
    f"({PRED_LEN * 15 / 60:.2f} hours)"
)

print(
    "\nOutputs:"
)

print(
    f"  Model:       {MODEL_PATH}"
)

print(
    f"  Predictions: {PREDICTION_PATH}"
)

print(
    f"  Metrics:     {METRICS_PATH}"
)

print(
    f"  Horizon:     {HORIZON_METRICS_PATH}"
)

print(
    f"  Plot:        {PLOT_PATH}"
)

print(
    "\nDone."
)