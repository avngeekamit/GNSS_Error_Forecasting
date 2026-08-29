# GNSS Error Forecasting Research

A machine learning based framework for forecasting GNSS satellite
ephemeris and satellite clock errors using past GNSS observations.

The repository contains a comparative evaluation of three forecasting
approaches:

- Hybrid XGBoost
- Direct GRU
- Direct LSTM

The models are evaluated on a held-out sixth-day GNSS observation set
using the actual Day-6 observations as ground truth.


# Research Objective

GNSS positioning accuracy can be affected by errors in satellite
ephemeris and satellite clock information.

This project investigates whether machine learning models can forecast
future GNSS error behaviour from previously observed GNSS data.

The main objective is to forecast:

- X-axis ephemeris error
- Y-axis ephemeris error
- Z-axis ephemeris error
- Satellite clock error

The study focuses on a long-horizon Day-6 forecasting experiment.

---

## Forecasting Setup

The models use past GNSS observations to forecast future error values.

The final evaluation uses:

- 95 forecast points
- 15-minute sampling interval
- Forecast duration: 23 hours 45 minutes
- Actual Day-6 observations as ground truth

The forecast horizons are:

```text
15, 30, 45, ... , 1425 minutes
```

---

## Models

### 1. Hybrid XGBoost

The proposed Hybrid XGBoost approach combines:

- Direct XGBoost forecasting
- Recursive XGBoost forecasting

The current implementation uses:

- Direct forecasting for X and Y ephemeris errors
- Recursive forecasting for Z ephemeris error and satellite clock error

The model uses temporal and historical features including:

- Lag features
- Difference features
- Rolling mean features
- Rolling standard deviation features
- Time-based features

The current Hybrid XGBoost implementation contains 21 input features.

---

### 2. Direct GRU

A multivariate GRU model is used as a deep-learning baseline.

The model uses historical GNSS error observations as input and directly
produces the complete Day-6 forecast.

The implementation is developed using PyTorch.

---

### 3. Direct LSTM

A multivariate LSTM model is used as another deep-learning baseline.

Similar to the GRU model, it uses past GNSS observations and directly
generates the Day-6 forecast.

The implementation is developed using PyTorch.

---

## Dataset

The repository contains two Day-6 related datasets:

```text
data/
├── GEO_day6_train.csv
├── GEO_day6_actual.csv
└── README.md
```

### Training Data

`GEO_day6_train.csv`

Contains the GNSS observations used by the forecasting models during
model development and training.

### Actual Day-6 Data

`GEO_day6_actual.csv`

Contains the actual GNSS observations for the sixth day.

These observations are used as the ground truth for evaluating the
forecasted values.

---

## Evaluation Methodology

All three models are evaluated using the same actual Day-6 dataset.

```text
Past GNSS observations
          |
          v
+---------------------------+
|     Forecasting Models    |
|                           |
| Hybrid XGBoost            |
| Direct GRU                |
| Direct LSTM               |
+-------------+-------------+
              |
              v
       95-step forecast
              |
              v
+---------------------------+
| Actual Day-6 observations |
+-------------+-------------+
              |
              v
       Common evaluation
```

The common evaluation script is:

```text
src/evaluation/common_day6_evaluation.py
```

---

## Evaluation Metrics

The following metrics are calculated:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score
- Maximum Absolute Error (MaxAE)
- 95th Percentile Absolute Error (P95-AE)
- Bias
- Correlation

Horizon-wise absolute errors are also calculated to study how model
accuracy changes with increasing forecast horizon.

---

## Results

The repository contains the common Day-6 comparison results:

```text
results/
└── model_comparison/
    ├── DAY6_model_comparison_metrics.csv
    ├── DAY6_model_comparison_horizon_metrics.csv
    └── DAY6_model_comparison_summary.csv
```

The current Day-6 experiment shows that Hybrid XGBoost achieves lower
MAE and RMSE than the GRU and LSTM baselines across the four evaluated
GNSS error components.

However, the models produce negative R² values at this long forecasting
horizon. Therefore, the results should be interpreted as a comparison
of forecasting error rather than evidence of universally strong
long-horizon predictive performance.

---

## Figures

Research figures are generated using:

```text
src/evaluation/generate_figures.py
```

The generated figures are organized as:

```
figures/
├── forecasts/
│   ├── hybrid_xgboost/
│   ├── gru/
│   ├── lstm/
│   └── *_all_models.png
│
├── horizon_analysis/
│
└── model_comparison/
```

The figures include:

- Actual vs predicted forecasts
- Model-to-model forecast comparison
- Horizon-wise absolute error
- MAE comparison
- RMSE comparison
- R² comparison

---

## Repository Structure

```text
GNSS_Error_Forecasting_Research/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── GEO_day6_train.csv
│   ├── GEO_day6_actual.csv
│   └── README.md
│
├── src/
│   ├── evaluation/
│   │   ├── common_day6_evaluation.py
│   │   └── generate_figures.py
│   │
│   ├── gru/
│   │   └── 6day_dir_gru.py
│   │
│   ├── hybrid_xgboost/
│   │   └── hybrid_day6_xgboost.py
│   │
│   └── lstm/
│       └── 6day_direct_lstm.py
│
├── results/
│   ├── gru/
│   ├── hybrid_xgboost/
│   ├── lstm/
│   └── model_comparison/
│
├── figures/
│   ├── forecasts/
│   ├── horizon_analysis/
│   └── model_comparison/
│
└── references/
    └── README.md
```

---

## Installation

Create and activate a Python virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Running the Evaluation

From the repository root:

```bash
python src/evaluation/common_day6_evaluation.py
```

This generates the common Day-6 evaluation results under:

```text
results/model_comparison/
```

---

## Generating Figures

Run:

```bash
python src/evaluation/generate_figures.py
```

The figures will be generated under:

```text
figures/
```

---

## Research Scope and Future Work

The current repository focuses on temporal forecasting using past GNSS
observations.

Potential future improvements include:

- Cross-variable feature learning between ephemeris and clock errors
- More extensive training datasets
- Additional satellites and GNSS constellations
- Longer evaluation periods
- GRU/LSTM architecture optimization
- More advanced temporal deep-learning models
- Improved long-horizon forecasting accuracy

In particular, cross-variable relationships between GNSS error
components can be investigated in future versions of the framework.

---

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- PyTorch
- Matplotlib

---

## Disclaimer

This repository represents an experimental research implementation.
The reported results are based on the available Day-6 evaluation
dataset and should not be interpreted as operational GNSS correction
or navigation-grade performance.
