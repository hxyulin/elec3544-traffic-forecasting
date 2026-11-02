# Spatiotemporal traffic speed forecasting

Group 30, ELEC3544. This repository forecasts METR-LA traffic speeds 15, 30 and 60 minutes ahead. It covers weeks 1–3 of the proposal: data pipeline, exploratory analysis, baselines and classical models.

## Run

Requires [uv](https://docs.astral.sh/uv/). The first run downloads the dataset (13 MB) to `data/raw/` and checks its SHA-256.

```sh
uv sync
uv run pytest            # data alignment, split and metric checks
uv run scripts/eda.py    # figures and summary in outputs/eda/
uv run scripts/evaluate.py   # metrics in outputs/results/metrics.csv
```

`evaluate.py` takes model names to rerun only some models, for example `uv run scripts/evaluate.py ridge`. The full run takes under a minute on a laptop.

## Layout

- `src/traffic/data.py` loads the data, builds the sensor graph, and defines splits and windows.
- `src/traffic/metrics.py` holds MAE, RMSE and MAPE with missing targets masked out.
- `src/traffic/models.py` holds the baselines and the ridge and gradient-boosting models.
- `scripts/` holds the runnable analysis. `outputs/predictions/` (not committed) holds test predictions for error analysis.

## Data decisions

These follow DCRNN (Li et al., 2018), so our numbers compare directly with published results.

- **Source.** `metr_la.h5`: 34,272 five-minute steps from 1 March to 27 June 2012, 207 sensors. It comes from the Torch Spatiotemporal mirror of the DCRNN files.
- **Windows.** Each window has 12 input steps (1 hour) and 12 target steps. We report steps 3, 6 and 12, which are 15, 30 and 60 minutes.
- **Splits.** Windows are split 70/10/20 in time order: 23,974 train, 3,425 validation and 6,850 test windows. These match DCRNN exactly.
- **Missing values.** A speed of 0 means the sensor did not report. This covers 8.1% of readings overall and 12.1% in the test period.
  - Targets: zeros are excluded from training and from every metric.
  - Inputs: we carry the last observed value forward. This uses only past data. A gap at the very start of the series takes the sensor's training mean instead.
- **Graph.** The weight between two sensors is `exp(-(d/σ)^2)`, where `d` is road distance and `σ` is the standard deviation of all distances. Weights below 0.1 are set to 0. The resulting graph is directed and has 1,515 edges besides self-loops.
- **Normalisation.** `MetrLA.normalizer()` returns the mean and standard deviation of observed training speeds, for z-scoring inputs to the neural models. The classical models standardise their own features.
- **Leakage.** The weekly profile, fill values and normaliser use only the training period. `tests/test_data.py` checks this for the profile.

## Models so far

| Model | Inputs |
|---|---|
| Persistence | last observed speed |
| Historical average | mean training speed for that sensor and 5-minute slot of the week |
| Ridge, lags only | 12 lagged speeds, share of the window that was missing |
| Ridge | lags, calendar (time of day, day of week, weekend) at the target time, weekly-profile speed |
| Gradient boosting | ridge inputs plus sensor ID as a category; 1M sampled training rows; early stopping on validation |

The ridge and gradient-boosting models train one model per horizon, pooled across all sensors.

## Results (test set)

| Model | MAE 15 min | MAE 30 min | MAE 60 min | RMSE 15 min | RMSE 30 min | RMSE 60 min | MAPE 15 min | MAPE 30 min | MAPE 60 min |
|---|---|---|---|---|---|---|---|---|---|
| Persistence | 3.50 | 4.21 | 5.38 | 6.42 | 8.05 | 10.34 | 8.6% | 10.9% | 14.5% |
| Historical average | 4.19 | 4.19 | 4.19 | 7.85 | 7.85 | 7.85 | 13.0% | 13.0% | 13.0% |
| Ridge, lags only | 3.38 | 4.21 | 5.56 | 6.10 | 7.60 | 9.52 | 9.0% | 12.0% | 16.6% |
| Ridge | 3.36 | 3.76 | 4.08 | 5.81 | 6.69 | 7.39 | 9.1% | 10.7% | 12.1% |
| Gradient boosting | 3.08 | 3.50 | 3.86 | 5.67 | 6.56 | 7.28 | 8.4% | 10.0% | 11.5% |

MAE and RMSE are in mph. As a check, DCRNN reports historical-average errors of 4.16 MAE, 7.80 RMSE and 13.0% MAPE. For reference, DCRNN itself reaches 2.77, 3.15 and 3.60 MAE at the three horizons.
