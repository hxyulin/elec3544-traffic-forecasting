# DCRNN-style reference

This is AI-assisted exploratory work for student inspection and decisions. It is not independent student work or final report prose.

The reference follows Li, Yu, Shahabi and Liu, [DCRNN, ICLR 2018](https://arxiv.org/html/1707.01926v3), and the [original TensorFlow implementation](https://github.com/liyaguang/DCRNN). Upstream source commit and file hashes are in `third_party/dcrnn/source.json`; its MIT notice is preserved there. The current upstream cell file is named `dcrnn_cell.py`.

## Mechanism and differences

The model keeps a joint batch × sensor × hidden state. Each reset gate, update gate and candidate sees diffusion of both its input and recurrent state. Thus a sensor's past state can affect another sensor's next recurrent update. The earlier GraphSensorGRU only filtered spatial inputs before a conventional GRU.

For supplied adjacency A, A[i,j] means a listed edge from i to j. This does not independently establish physical traffic flow. The operators are transpose(row_normalize(A)) and transpose(row_normalize(A.T)), matching upstream orientation. Each direction uses X, PX, and 2P²X−X, with self X included once, making five slots. Unlike strict two-hop neighbor inputs, these polynomials include return paths and self contributions. They are not exactly the monomial random-walk expansion written in the paper. We reset the recurrence independently for each direction; upstream carries its modified x0 between directions. Disabled diffusion explicitly zeros all four diffusion slots before the same bias-free projections. It preserves the self slot, parameter count and initialized parameters. A zero adjacency alone is not equivalent: its second polynomial is −X.

The encoder processes twelve historical full-network steps. A separate decoder starts from encoder state and a zero GO value. Each projected prediction becomes the next decoder input, for twelve steps. A direct-horizon MLP predicts its reported horizons from one input representation instead. Autoregressive errors can accumulate through the decoder state and feedback.

This is not an exact DCRNN reproduction. It uses one layer with 32 hidden units rather than the reference configuration's two layers of 64. Historical inputs are train-normalized causally filled speed and observed flags. There are no calendar, sensor identity or geometry inputs. Decoder inputs contain only predicted speed, or training-only scheduled targets. Missing teacher targets fall back to the current prediction. Loss is normalized observed-only MAE over all twelve targets; reported physical-unit MAE and RMSE use steps 3, 6 and 12. Checkpoint selection minimizes their mean validation MAE. The original reference's optimizer, clipping, epoch schedule and preprocessing are replaced by the frozen project policy. Read `protocol.json` for exact source, data, split and initialization hashes.

Scheduled sampling uses a dedicated CPU generator seeded with seed+1000. One scalar draw per decoder transition chooses teacher forcing across the batch and sensors, with missing targets still falling back to predictions. The probability is 2000/(2000+exp(update/2000)). Evaluation ignores both target values and target masks and always feeds predictions back. Teacher forcing in evaluation would give the model future observations unavailable during forecasting and invalidate the comparison.

## Study protocol

The purged capacity-budget train split and reused benchmark validation starts are retained. Training targets end before the earliest validation target. The normalizer and initial missing-value fallback use training data only. No test evaluation or test checkpoint selection is implemented in the new runner.

Each epoch samples floor(1,000,000/N) distinct full-network starts, bounded by train size. With 207 sensors this is 4,830 windows and 999,810 window-sensor equivalents. Training covers all twelve targets, with raw missing values excluded. Actual observed-target counts and window and scheduled-sampling RNG hashes are logged per epoch. Seeds 42, 43 and 44 share initial parameters, sampled windows and scheduled-sampling draws across enabled and disabled fits. Each batch contains at most sixteen full-network windows.

The engineering pilot uses only seed 42 and a fixed tiny training subset. It checks finite output, learning, memory and elapsed time; it does not compare architectures using validation scores. Measured cost determines the epoch cap before fits. The maximum is fifty epochs with patience eight, AdamW lr .001, weight decay .0001, gradient norm clipping one, and ReduceLROnPlateau factor .5/patience three. Each run has a thirty-minute process limit and the sequence has a three-hour limit. The runner also stops at an epoch boundary after 1,700 seconds. Checkpoint predictions stay remote, and exact equality is checked after restoration.

## Interpretation limits

Enabled-minus-disabled paired differences estimate the contribution of these supplied diffusion operators within this fixed recurrent architecture and budget. Three seeds and a reused validation period limit inference. The graph's physical orientation remains unverified.

Capacity-budget current/wide one/two-hop MLP results provide context on the same evaluation targets. They cannot isolate recurrence: those models use different features, three-target training, row sampling, batch sizes and exposure. This model uses joint full-network windows and twelve-target training. Equal nominal window-sensor counts do not imply equal optimization or computational budgets. Low-speed targets below 40 mph and sixty-minute drops of at least 10 mph from an observed origin are descriptive diagnostics only, never selection criteria.

The paper reports sampling decay 3,000, while the pinned repository configuration uses 2,000; this study freezes the latter. With 302 updates per epoch and fifty epochs, teacher-forcing probability remains about 0.51 at the end of training. Evaluation still uses zero teacher forcing. This capped training schedule may leave a distribution mismatch. No annealing alternative was selected after observing validation results. Dense matrix operators replace upstream sparse TensorFlow operators, and PyTorch parameter initialization is matched across this study's variants rather than mapped to TensorFlow checkpoint values.

## Completed results

All six fits completed fifty epochs. The selected epochs were enabled 49/50/49 and disabled 50/50/48 for seeds 42/43/44. Every selected checkpoint reproduced its saved prediction array exactly after disk restoration. The audit recomputed all metrics, verified every sampled-window hash and checked identical paired scheduled-sampling RNG states. All 129 prior study artifacts retained their original hashes. No study jobs remained active at collection.

Values below are means ± sample standard deviations over three seeds, in mph. Macro values average the three reported horizon metrics within each seed.

| Horizon | Enabled MAE | Disabled MAE | Paired MAE difference | Enabled RMSE | Disabled RMSE |
|---|---:|---:|---:|---:|---:|
| 15 min | 2.654 ± 0.020 | 2.818 ± 0.002 | −0.165 ± 0.019 | 5.219 ± 0.045 | 5.729 ± 0.014 |
| 30 min | 3.145 ± 0.037 | 3.424 ± 0.003 | −0.279 ± 0.037 | 6.500 ± 0.084 | 7.236 ± 0.024 |
| 60 min | 3.899 ± 0.076 | 4.384 ± 0.007 | −0.484 ± 0.080 | 8.164 ± 0.175 | 9.204 ± 0.037 |
| Macro | 3.233 ± 0.044 | 3.542 ± 0.004 | −0.309 ± 0.045 | 6.628 ± 0.101 | 7.390 ± 0.024 |

Enabled errors were lower for all three seeds at every reported horizon, for both MAE and RMSE. Macro MAE differences were −0.257/−0.328/−0.342 mph for seeds 42/43/44. These are descriptive paired results within this model and budget.

| Seed | Enabled macro MAE / RMSE | Disabled macro MAE / RMSE |
|---|---:|---:|
| 42 | 3.284 / 6.743 | 3.541 / 7.379 |
| 43 | 3.210 / 6.583 | 3.539 / 7.373 |
| 44 | 3.204 / 6.557 | 3.546 / 7.418 |

For low-speed observed targets below 40 mph, enabled versus disabled mean MAE was 6.671 versus 7.359 mph at fifteen minutes, 8.996 versus 9.883 at thirty minutes, and 12.480 versus 13.800 at sixty minutes. The respective target counts were 70,133, 70,031 and 69,632 per seed. On the 35,479 observed sixty-minute targets with a drop of at least 10 mph from an observed origin, MAE was 21.624 versus 24.640 mph. These larger errors show that the aggregate results do not imply good incident forecasting. Per-seed MAE and RMSE for these diagnostics are in `diagnostics.csv`.

The earlier fifty-epoch MLP macro MAEs on the same evaluation targets were current/one 2.926 ± 0.007, current/two 2.911 ± 0.003, wide/one 2.910 ± 0.001, and wide/two 2.892 ± 0.003 mph. Thus this compact graph-recurrent reference did not beat those MLPs under their respective protocols. This does not establish that recurrence is inferior: the feature sets, loss horizons, sampling, initialization and compute differ. `same_evaluation_targets.json` records the speed, graph and split hash match; `contextual_comparison.csv` preserves per-seed comparisons with the fifty-epoch MLP checkpoints.

## Validation and accounting

The complete project test suite passed: 58 tests in 3.50 seconds. Ruff format and checks passed for the new Python files. Tests cover directed operators against direct matrix arithmetic, disabled self-path equivalence and sensor isolation, parameter matching, window alignment, finite gradients and empty masks, training-only scheduled sampling, missing-target fallback, evaluation independence from future truths and masks, state restoration, purging and train-only normalization/fill bounds. The completed-run audit additionally tests disk checkpoint restoration, exact predictions, metric recomputation and paired RNG records.

The pilot's forty fixed-subset updates reduced normalized loss from 0.497 to 0.264. Its synchronized batch cost was 0.0222 seconds and full validation cost 0.720 seconds, forecasting 7.44 seconds per epoch. It justified retaining fifty epochs, not selecting an architecture by validation score. Actual total fit and restoration time was 2,435 seconds, about 40.6 minutes. Enabled fits took approximately 457–458 seconds each and disabled fits approximately 352–356 seconds each. Runtime includes data construction and full validation; pilot boundaries and completed-run timing synchronize CUDA.

Both variants have 32,385 trainable parameters. Each fit used 241,500 full-network windows, 49,990,500 window-sensor equivalents and 599,886,000 possible target entries over fifty epochs. Across six fits this is 1,449,000 windows, 299,943,000 window-sensor equivalents and 3,342,309,528 observed target entries. The final batch partition is 302 batches per epoch, each at most sixteen windows. Enabled and disabled observed-target exposure matches exactly within each seed. Validation exposure is separate and is not used for teacher forcing or gradient updates.

The verified runtime used Python 3.12.14, PyTorch 2.15.0+cu132, CUDA runtime 13.2, and an RTX 5090. The installed NVIDIA driver was 615.78.08 and reported CUDA compatibility 13.4. Peak allocated PyTorch GPU memory was 195,863,552 bytes per full fit; this excludes driver and desktop allocations. Dependencies were installed into a fresh project-local uv environment. Existing data was copied into the isolated study directory; prior directories and checkpoints were not overwritten.

Compact artifacts are under `outputs/experiments/dcrnn_reference`: protocol and data/source/initialization hashes, per-seed and aggregate tables, paired differences and sign counts, diagnostics, curves, histories with exposure and RNG hashes, environment, test and lifecycle logs, checkpoint/prediction hashes and audit results. Checkpoints and prediction arrays remain remote and ignored. Private operational paths are supplied only in the handoff. The existing training runner and prior frozen source lineage were not changed.
