# Spatiotemporal traffic speed forecasting

Group 30, ELEC3544. We forecast METR-LA traffic speeds 15, 30 and 60 minutes ahead and test whether road-network information improves forecasts. The project compares persistence and historical averages, ridge regression, gradient boosting, a temporal GRU, and a GRU with learned graph filtering.

## Results

Test-set MAE and RMSE in mph. Lower is better. Settings and neural checkpoints were selected on validation MAE; test scores describe the selected models. Values below come from [`metrics.csv`](outputs/results/metrics.csv).

| Model | MAE 15 min | MAE 30 min | MAE 60 min | RMSE 15 min | RMSE 30 min | RMSE 60 min |
|---|---:|---:|---:|---:|---:|---:|
| Persistence | 3.498 | 4.205 | 5.378 | 6.416 | 8.049 | 10.340 |
| Historical average | 4.186 | 4.186 | 4.186 | 7.851 | 7.851 | 7.851 |
| Ridge (lags only) | 3.380 | 4.206 | 5.556 | 6.102 | 7.597 | 9.524 |
| Ridge | 3.212 | 3.679 | 4.081 | 5.693 | 6.611 | 7.356 |
| Ridge + neighbours | 3.228 | 3.693 | 4.088 | 5.626 | 6.548 | 7.316 |
| Gradient boosting | 2.958 | 3.400 | 3.818 | 5.402 | 6.323 | 7.125 |
| Gradient boosting + neighbours | **2.887** | **3.332** | **3.771** | **5.164** | **6.111** | **6.986** |
| Temporal GRU | 3.006 | 3.596 | 4.424 | 5.891 | 7.201 | 8.858 |
| Graph GRU | 3.000 | 3.584 | 4.408 | 5.848 | 7.141 | 8.815 |

The separately trained disabled-graph control reproduces every temporal-GRU test prediction exactly, so its duplicate row is omitted. It remains in the saved metrics and [control verification](outputs/neural/graph_gru/control_verification.json).

**Gradient boosting + neighbours is the strongest measured model.** It reduces persistence MAE by 17.5%, 20.8% and 29.9% at the three horizons. Adding neighbour features improves boosting MAE at every horizon. For ridge, neighbour features slightly worsen MAE while improving RMSE. See the [feature ablations](docs/classical_models.md).

The graph GRU improves temporal-GRU MAE by only 0.005, 0.012 and 0.015 mph. At 60 minutes it helps 118 of 207 sensors and worsens 89. This one-seed experiment shows a small measured spatial gain; it does not establish statistical significance or the best graph architecture. Classical boosting also receives sensor identity and weekly-profile features that the neural models do not receive.

![Graph and temporal models compared with persistence and gradient boosting + neighbours](outputs/neural/graph_gru/comparison.png)

## Congestion and missing inputs

Overall error hides larger errors on observed targets below 40 mph. This low-speed definition was chosen using training-only EDA; it is not ground-truth congestion or a congestion detector.

| Model | MAE 15 min | MAE 30 min | MAE 60 min |
|---|---:|---:|---:|
| Persistence | 7.336 | 9.673 | 13.421 |
| Gradient boosting + neighbours | 6.849 | 8.784 | 10.776 |
| Temporal GRU | 7.472 | 10.101 | 14.597 |
| Graph GRU | 7.421 | 10.080 | 14.518 |

Both neural models beat persistence overall but have higher MAE on low-speed targets at every horizon. At 60 minutes, fully observed input histories account for 92.63% of observed targets. Graph filtering improves temporal-GRU MAE by 0.031 mph in that group, but worsens it by 0.081, 0.266 and 0.210 mph when 1–3, 4–11 or all twelve input readings are missing. These groups differ in sensor and traffic composition; the comparison is descriptive.

The [error-analysis report](docs/error_analysis.md) includes errors by sensor and target hour, missing inputs, speed changes, representative traces and selected failure cases. All 30 model/horizon aggregates and 150 subgroup partitions are checked against pooled metrics. Counts are observed forecast-target pairs, not independent samples.

## Reproduce the results

Requires Python 3.12 or later and [uv](https://docs.astral.sh/uv/). The first data-loading command downloads the 13 MB dataset and checks its archive SHA-256. The `neural` extra adds PyTorch; the classical models and analysis also run without it.

### Set up and validate

```sh
uv sync --extra neural
uv run ruff check src scripts tests
uv run ruff format --check src scripts tests
uv run --extra neural pytest
```

Neural tests skip when PyTorch is absent. Ruff settings and development dependencies are defined in `pyproject.toml` and locked in `uv.lock`.

### Restore the recorded models and analyse predictions

The selected neural checkpoints are committed. Classical models refit using the committed validation-selected settings. Predictions are generated locally and remain ignored by git.

```sh
uv run scripts/evaluate.py
uv run --extra neural scripts/train_gru.py --device cpu --evaluate-only
uv run --extra neural scripts/train_gru.py --model graph_gru --device cpu --evaluate-only
uv run --extra neural scripts/train_gru.py --model graph_gru_no_graph --device cpu --evaluate-only
uv run scripts/error_analysis.py
uv run scripts/detailed_error_analysis.py
```

Checkpoint restoration verifies the speed-data hash, sensor order, horizons and normalizer. Graph checkpoints also verify the adjacency hash. CPU restoration reproduced every saved neural prediction exactly in the recorded environment; other devices can introduce numerical differences.

### Retrain or regenerate reports

```sh
# Repeat validation selection and feature ablations.
uv run scripts/evaluate.py --tune
uv run scripts/evaluate.py
uv run scripts/evaluate.py --ablation

# Retrain the temporal, graph and disabled-graph variants.
uv run --extra neural scripts/train_gru.py --device cpu
uv run --extra neural scripts/train_gru.py --model graph_gru --device cpu
uv run --extra neural scripts/train_gru.py --model graph_gru_no_graph --device cpu

# Regenerate EDA, analysis, report sections and figures.
uv run scripts/eda.py
uv run scripts/error_analysis.py
uv run scripts/detailed_error_analysis.py
uv run scripts/summarize_results.py
uv run scripts/summarize_neural.py
uv run scripts/summarize_graph.py
```

`evaluate.py` accepts a subset of model names, for example `uv run scripts/evaluate.py ridge`. Tuning and ablation repeat training. Neural variants share seed 42, a fixed sample of one million training window/sensor rows, batch size 2048, learning rates 0.001 and 0.003, and a 25-epoch cap with patience four. Histories, settings, package versions and checkpoints are saved under `outputs/`.

## Data and evaluation decisions

- METR-LA contains 34,272 five-minute steps from 207 sensors, from 1 March to 27 June 2012. Each window uses one hour of history and twelve future steps; we report steps 3, 6 and 12.
- The benchmark splits contain 23,974 training, 3,425 validation and 6,850 test windows in time order. Adjacent target periods overlap by eleven timestamps, or 55 minutes. These are not fully disjoint target periods.
- Zero speeds denote missing readings and are excluded from training targets and every score. Inputs carry the last observation forward. Initial gaps use sensor means from the training period. Normalization uses observed training speeds.
- Training profile features exclude the target's whole calendar week from both origin/target estimates and fallback means. Validation and test use the full training profile.
- The road graph has 1,515 directed non-self edges. Classical neighbour features symmetrize weights; the graph GRU keeps incoming and outgoing directions separate, removes self-loops from neighbour aggregation, and row-normalizes weights. Its local path remains separate.
- The graph model uses a one-hop learned input filter with local recurrent states. It does not reproduce the full DCRNN architecture. One neural seed, sampled training rows and overlapping windows limit the conclusions. Repeated seeds and purged-split sensitivity studies remain extensions.

## Repository guide

| Path | Contents |
|---|---|
| `src/traffic/` | Data pipeline, metrics, classical/neural models, analysis checks and reporting helpers |
| `scripts/` | Evaluation, training, EDA and report-generation commands |
| `tests/` | Alignment, missing-target, profile-exclusion and graph-control checks |
| `outputs/results/` | Metrics, validation selection, feature ablations and previous baseline scores |
| `outputs/neural/` | Selected checkpoints, training histories, settings and comparison figures |
| `outputs/error_analysis/` | Condition and subgroup scores, trace values and case-selection records |
| `outputs/eda/` | Training-data summaries, outage audits, congestion rules and sensor relationships |
| `docs/` | Methods, results, interpretation, provenance and remaining submission work |

Read the [EDA](docs/eda.md), [classical-model](docs/classical_models.md), [temporal-model](docs/temporal_model.md), [graph-model](docs/graph_model.md) and [error-analysis](docs/error_analysis.md) sections for the full experimental account. The model sequence and planned error analysis are complete; the [remaining work](docs/tasks.md) is final report/presentation assembly and member contribution statements.

The classical-model extension originated in Saif's [task2-models commit](https://github.com/saifho/elec3544-traffic-forecast/commit/ce18f4c1d603e3a2b413ce7233eb59aefbfa7baf). Source attribution and foundation-model assistance are recorded in [experiment provenance](docs/experiment_notes.md) and git history.
