# Multi-hop and geometry replication

AI-assisted exploratory experiments. Inspect the implementation and evidence before drawing conclusions.

## Protocol and audit

The replication extends the pilot from base `0272ea47390ef3ac7ee89252d33da923fe79e690`. Four input ablations share 82,713 trainable parameters: local, incoming/outgoing one-hop histories, one-hop plus strict two-hop histories, and those histories plus geometry. Masks are buffers; initialization checks hash named trainable parameters rather than variant-specific masks. Seeds are 42, 43 and 44. Each run allows 50 epochs, draws one million distinct training rows freshly each epoch, and uses batch size 2048. The paired runs use the same initialization and sample sequence within a seed.

The shared policy remains AdamW (learning rate .001, weight decay .0001), masked MAE, gradient clipping at 1, and ReduceLROnPlateau (factor .5, patience 3, threshold .0001). Early stopping allows eight epochs without a new minimum macro validation MAE. Selection uses the unweighted mean at 15, 30 and 60 minutes. No original test evaluation is performed.

The feature audit found no material bug requiring a change to the pilot feature definition. Graph supports remove self connections. Strict two-hop channels square each directed support, remove self and direct neighbours, and renormalize nonempty rows. Incoming road lengths use the transposed distance matrix. Empty supports stay zero. Reliability and gap ages use only observations at or before each history timestamp. Window inputs end before their forecast targets. Target calendar identities are known in advance. Geometry uses static coordinates and listed road distances only.

Training target windows are purged at the first validation target. Normalization and the fallback for gaps before any observation use only that purged training period. The fallback is a fitted training statistic, rather than a strictly online initial estimate within the training period. Validation histories may include earlier training-period readings. Missing raw targets stay zero and are excluded from loss and metrics. The existing benchmark validation period has been reused in earlier model development.

The frozen `protocol.json` records source, raw-file and derived-array hashes, split boundaries, initialization hashes and policy before any fits. Each run recomputes and checks the contract. Existing run directories reject replacement. The pilot directory and results remain separate. The runner records sample hashes every epoch, saves the selected predictions remotely, restores the selected checkpoint and requires exact prediction equality and exact macro MAE equality. The summary checks every observed sample hash against the seeded sampling sequence, including epochs beyond another variant's early stop.

## Execution and reproduction

Use the project uv environment:

```sh
uv sync --python 3.12 --extra neural --group dev
uv run ruff check src/traffic/multihop.py src/traffic/training.py scripts/try_multihop*.py tests/test_multihop.py
uv run --extra neural pytest -q
uv run --extra neural scripts/try_multihop_replication.py --prepare
# Run each combination of seed 42/43/44 and variant local/one/two/geometry:
CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run --extra neural scripts/try_multihop_replication.py --seed 42 --variant local
uv run --extra neural scripts/summarize_multihop_replication.py
```

The isolated Linux workstation has an RTX 5090 with 32,607 MiB device memory, 123 GiB system RAM and 32 CPU threads. The study uses four CPU threads per fit, Python 3.12.14, PyTorch 2.15.0+cu132, CUDA 13.2 and NVIDIA driver 615.78.08. Package versions are recorded per run with editable installation paths omitted. The persistent session runs fits sequentially, with a 900-second per-run timeout and 10,800-second overall hard bound. Failures stop the sequence and record exit status. Session and private access details are kept in the conversation handoff.

GPU synchronization bounds epoch and fit timings. `runtime.json` separates feature/model setup from fitting, checkpoint restoration and final validation. Dataset loading, environment setup and initial protocol preparation are outside these fit timings. Epoch times exclude checkpoint and history-file writes; fit times include them. Exact selected predictions and checkpoints remain remote; only hashes, metrics and logs are delivered.

Validation actually run: Ruff formatting and lint passed on the new feature module, both fit runners, shared training change and feature tests. The full remote pytest suite passed with 50 tests in 5.72 seconds. Tests cover horizon alignment, causal input invariance with frozen normalization, missing masks, orientation, strict two-hop exclusions, geometry, empty supports, disabled gradients, equal initialization and purged boundaries. The summary script also passed Ruff format and lint.

## Limits for interpretation

This is a validation-only repeated-seed comparison. It cannot establish out-of-period generalization or statistical significance. Three seeds describe training variability on this reused validation period. They do not supply independent data replications. Historical model scores are not matched controls and are not compared here.

Geometry includes coordinates and road distance, with no lane limits or junction inventory. Spatial diffusion uses static support and timestamp-aligned history aggregation. It does not model full temporal propagation. Any convergence statement must account for runs that reach the 50-epoch cap.

## Observed results

All 12 fits completed successfully. All ran 50 epochs; none stopped early. The selected epochs range from 46 to 50, and two selections fall exactly on the cap. Convergence beyond this budget remains untested. The wall-clock session ran from 15:32:34 to 15:41:12 UTC on 2026-11-12 (518 seconds), including process startup and dataset loading.

Macro MAE and RMSE are unweighted means of the three horizon metrics, in mph. SD below is the sample SD across three seeds.

| Seed | Variant | Macro MAE | Macro RMSE | Best epoch | Epochs |
|---|---|---:|---:|---:|---:|
| 42 | local | 3.003454 | 6.226325 | 48 | 50 |
| 42 | one | 2.919702 | 5.991053 | 48 | 50 |
| 42 | two | 2.907729 | 5.957794 | 48 | 50 |
| 42 | geometry | 2.919594 | 5.955881 | 48 | 50 |
| 43 | local | 3.010091 | 6.226112 | 50 | 50 |
| 43 | one | 2.932822 | 6.002451 | 49 | 50 |
| 43 | two | 2.913691 | 5.951187 | 46 | 50 |
| 43 | geometry | 2.920002 | 5.956530 | 46 | 50 |
| 44 | local | 3.005413 | 6.250196 | 49 | 50 |
| 44 | one | 2.924433 | 5.995009 | 49 | 50 |
| 44 | two | 2.910883 | 5.941279 | 50 | 50 |
| 44 | geometry | 2.903549 | 5.945353 | 49 | 50 |

Across-seed horizon metrics:

| Variant | Horizon (minutes) | MAE mean ± SD | RMSE mean ± SD |
|---|---|---:|---:|
| local | 15 | 2.622837 ± 0.003204 | 5.249822 ± 0.006093 |
| local | 30 | 2.999892 ± 0.003726 | 6.257488 ± 0.008707 |
| local | 60 | 3.396229 ± 0.005248 | 7.195322 ± 0.028183 |
| local | Macro | 3.006320 ± 0.003410 | 6.234211 ± 0.013844 |
| one | 15 | 2.553828 ± 0.004280 | 5.014998 ± 0.005865 |
| one | 30 | 2.909727 ± 0.005654 | 5.999445 ± 0.004263 |
| one | 60 | 3.313402 ± 0.010214 | 6.974070 ± 0.007826 |
| one | Macro | 2.925652 ± 0.006645 | 5.996171 ± 0.005787 |
| two | 15 | 2.547552 ± 0.004893 | 4.998212 ± 0.008876 |
| two | 30 | 2.893955 ± 0.004003 | 5.951277 ± 0.012007 |
| two | 60 | 3.290796 ± 0.003305 | 6.900771 ± 0.013893 |
| two | Macro | 2.910768 ± 0.002983 | 5.950087 ± 0.008312 |
| geometry | 15 | 2.546364 ± 0.005793 | 4.994019 ± 0.005191 |
| geometry | 30 | 2.897116 ± 0.009634 | 5.952379 ± 0.009707 |
| geometry | 60 | 3.299665 ± 0.012733 | 6.911365 ± 0.009259 |
| geometry | Macro | 2.914382 ± 0.009384 | 5.952588 ± 0.006274 |

Paired deltas subtract the stated baseline within each seed. Negative values improve the metric. Improvement counts are out of three seeds.

| Comparison | Horizon (minutes) | MAE delta mean ± SD | MAE improvements | RMSE delta mean ± SD | RMSE improvements |
|---|---|---:|---:|---:|---:|
| two-one | 15 | -0.006276 ± 0.000614 | 3/3 | -0.016786 ± 0.005238 | 3/3 |
| two-one | 30 | -0.015772 ± 0.001882 | 3/3 | -0.048168 ± 0.011337 | 3/3 |
| two-one | 60 | -0.022606 ± 0.010136 | 3/3 | -0.073299 ± 0.021621 | 3/3 |
| two-one | Macro | -0.014885 ± 0.003761 | 3/3 | -0.046084 ± 0.011175 | 3/3 |
| geometry-two | 15 | -0.001188 ± 0.007062 | 2/3 | -0.004193 ± 0.004706 | 2/3 |
| geometry-two | 30 | 0.003161 ± 0.007993 | 1/3 | 0.001102 ± 0.005095 | 1/3 |
| geometry-two | 60 | 0.008869 ± 0.015958 | 1/3 | 0.010595 ± 0.007830 | 0/3 |
| geometry-two | Macro | 0.003614 ± 0.009880 | 1/3 | 0.002501 ± 0.003875 | 1/3 |

Strict two-hop improves macro MAE over one-hop in all three seeds. Geometry improves macro MAE over two-hop in one seed and increases it in two. These observations support student review of the two-hop feature at this budget; they do not establish a generalization gain or significance.

Every observed epoch sample hash matches the seeded policy and agrees across variants within each seed. All selected restored checkpoints reproduce their stored predictions exactly. Recomputed macro MAE also equals the stored selection score exactly. Every frozen executable source hash matches the delivered local source.

Artifacts are in `outputs/experiments/multihop_geometry_replication`: `per_seed.csv` contains every seed/horizon MAE and RMSE; `paired_deltas.csv` contains each paired difference; `aggregate.csv` and `paired_aggregate.csv` contain means, sample SDs and sign counts. `curves.png` shows validation curves and selected epochs. Per-run `history.csv`, `selection.json`, `runtime.json` and `environment.json` preserve detailed records. `protocol.json`, `lifecycle.json`, `verification.json`, `tests.log`, `prepare.log`, `replication.log` and `summary.log` preserve the contract and observed execution.

Total recorded fit/restoration/final-validation time was 486.952 seconds. Total recorded feature/model setup time was 7.863 seconds. The session wall time includes the additional dataset-loading and process overhead.
