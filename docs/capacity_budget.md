# Capacity and training-budget comparison

AI-assisted exploratory experiments. Inspect the implementation and evidence before drawing conclusions.

The frozen factorial protocol compares current and wider residual MLPs, one-hop and one-plus-strict-two-hop histories, and seeds 42, 43 and 44. All twelve are fresh fits on the same purged training period and reused benchmark validation period. No original test scores or geometry ablations enter this batch. Each run has a 100-epoch cap, eight-epoch early stopping, one million freshly sampled training rows per epoch without replacement, batch size 2048, and the existing AdamW, scheduler, masked MAE and clipping policy. The run limit is 900 seconds and the batch limit is 10,800 seconds.

Current dimensions are local 64, spatial 32 and residual width 128, totaling 82,713 trainable parameters. Wider dimensions are local 128, spatial 64 and residual width 224, totaling 229,241. Both retain the same 32 embedding channels, two residual blocks, dropout 0.1 and all 24 bias-free spatial encoder slots. Input masks activate eight slots for one hop and sixteen for two hops. Geometry slots remain zero. Only width changes.

The constructor takes the original initialization path unchanged for the default model. A test loads the prior committed class, compares every state value, assigns a shared nonzero head and checks exact outputs. Initialization hashes in the protocol include parameter names and values. Hashes match across hop variants within each size and seed; different sizes do not support identical initialization claims.

The shared trainer saves a second checkpoint and prediction whenever validation improves within the first 50 epochs. The full run uses the unchanged scheduler continuously through at most 100 epochs. This gives a nested budget comparison without restarting or retuning. Early-stopped runs can have identical prefix and final selections. Both retained checkpoints are restored and predictions must match exactly. Checkpoint hashes and horizon scores are saved separately.

The data path is unchanged from the prior study. Targets remain raw and are masked when nonpositive. Filling uses only observations available by each history timestamp, with training-only fallback and normalization. Training windows whose targets intersect the validation history boundary are purged. Graph aggregation uses contemporaneous history values; strict two-hop supports exclude self and direct neighbours. Known target calendar identities do not contain future traffic measurements. Existing alignment, causality, raw-mask and control tests remain in the full suite.

The machine is a Linux GPU workstation with an NVIDIA GeForce RTX 5090, 32,607 MiB VRAM. Fits run sequentially in an isolated directory with uv dependencies and a persistent session. Earlier studies are preserved. Setup time is separate from synchronized fit-and-verification time. Raw data, checkpoints and prediction arrays remain remote and are excluded from the tracked evidence bundle.

`outputs/experiments/capacity_budget/protocol.json` records source, data, split, initialization and policy hashes before fitting. Its inherited `confounds` and `geometry` text describes the earlier implementation lineage; `variants`, `sizes`, `epochs` and this document define this batch. No geometry variant runs here.

Three seeds are optimization replications on reused data. Their standard deviations are descriptive, not significance tests or temporal generalization estimates. Diagnostic subgroups use observed future targets retrospectively: target speed below 40 mph at each horizon, and an observed-origin-to-target drop of at least 10 mph at 60 minutes. They do not select models or assess a detector.

The initial-gap fallback is a training-period estimate rather than a strictly online estimate for the earliest training windows. It can use observations later within training. It uses no validation targets. The same fallback is retained across all compared fits.

## Observed results

All twelve fits completed with eight-epoch early stopping; none reached the cap. All observed epoch sample hashes matched the seeded sampling policy. All 24 checkpoint restorations reproduced saved predictions exactly. Fit plus restoration time totaled 641.72 seconds; feature setup totaled 7.90 seconds. The CPU was an AMD Ryzen 9 9950X3D. The environment used Python 3.12.14, PyTorch 2.15.0+cu132, CUDA 13.2 and NumPy 2.5.4. Complete package versions are saved per run.

| Size | Hops | Budget | Macro MAE mean ± SD | Macro RMSE mean ± SD |
|---|---|---|---|---|
| current | one | 50 | 2.925652 ± 0.006645 | 5.996171 ± 0.005787 |
| current | one | 100 | 2.919088 ± 0.003373 | 5.984783 ± 0.005206 |
| current | two | 50 | 2.910768 ± 0.002983 | 5.950087 ± 0.008312 |
| current | two | 100 | 2.902472 ± 0.004778 | 5.936644 ± 0.018600 |
| wide | one | 50 | 2.910171 ± 0.001302 | 5.978328 ± 0.009184 |
| wide | one | 100 | 2.906683 ± 0.005919 | 5.975163 ± 0.014324 |
| wide | two | 50 | 2.891817 ± 0.002884 | 5.925071 ± 0.017710 |
| wide | two | 100 | 2.891817 ± 0.002884 | 5.925071 ± 0.017710 |

At the full budget, widening reduced macro MAE by 0.012405 mph with one hop and 0.010654 mph with two hops, averaged over seeds. Two hops reduced MAE by 0.016616 mph at current size and 0.014866 mph at wider size. All four full-budget paired MAE comparisons favored the candidate in every seed. These improvements are about 0.4–0.6% of the baseline scores. Their practical value depends on the application; this study does not establish temporal generalization or statistical significance.

Extra budget reduced average MAE by 0.006564 mph for current/one, 0.008296 for current/two, 0.003488 for wide/one, and zero for wide/two. All wider two-hop selections occurred by epoch 50. This is a nested checkpoint comparison under one continuous scheduler policy, not an independent 50-epoch retraining experiment.

| Seed | Size | Hops | Epochs observed | Best ≤50 epoch | Final best epoch |
|---|---|---|---|---|---|
| 42 | current | one | 71 | 48 | 63 |
| 42 | current | two | 56 | 48 | 48 |
| 42 | wide | one | 69 | 46 | 61 |
| 42 | wide | two | 54 | 46 | 46 |
| 43 | current | one | 75 | 49 | 67 |
| 43 | current | two | 79 | 46 | 71 |
| 43 | wide | one | 57 | 49 | 49 |
| 43 | wide | two | 57 | 49 | 49 |
| 44 | current | one | 84 | 49 | 76 |
| 44 | current | two | 67 | 50 | 59 |
| 44 | wide | one | 65 | 49 | 57 |
| 44 | wide | two | 48 | 40 | 40 |

### Horizon scores at the full budget

| Size | Hops | Horizon minutes | MAE mean ± SD | RMSE mean ± SD |
|---|---|---|---|---|
| current | one | 60 | 3.305072 ± 0.006090 | 6.959535 ± 0.006372 |
| current | one | 15 | 2.549079 ± 0.001800 | 5.007016 ± 0.002908 |
| current | one | 30 | 2.903112 ± 0.003358 | 5.987798 ± 0.006895 |
| current | two | 60 | 3.280294 ± 0.008140 | 6.881276 ± 0.031567 |
| current | two | 15 | 2.540708 ± 0.002233 | 4.988546 ± 0.007701 |
| current | two | 30 | 2.886412 ± 0.004775 | 5.940111 ± 0.017525 |
| wide | one | 60 | 3.289332 ± 0.009846 | 6.955726 ± 0.020654 |
| wide | one | 15 | 2.537707 ± 0.003405 | 4.990558 ± 0.009850 |
| wide | one | 30 | 2.893010 ± 0.005227 | 5.979205 ± 0.012899 |
| wide | two | 60 | 3.267186 ± 0.006546 | 6.878767 ± 0.023328 |
| wide | two | 15 | 2.531666 ± 0.000541 | 4.970528 ± 0.012737 |
| wide | two | 30 | 2.876601 ± 0.002020 | 5.925917 ± 0.020605 |

### Diagnostic errors at the full budget

| Size | Hops | Horizon minutes | Subgroup | Count per seed | Mean MAE | Mean RMSE |
|---|---|---|---|---|---|---|
| current | one | 15 | below_40_mph | 70133 | 6.561585 | 10.667182 |
| current | one | 30 | below_40_mph | 70031 | 8.502665 | 13.569532 |
| current | one | 60 | below_40_mph | 69632 | 10.747393 | 16.489853 |
| current | one | 60 | drop_at_least_10_mph | 35479 | 16.702879 | 21.266700 |
| current | two | 15 | below_40_mph | 70133 | 6.465126 | 10.530185 |
| current | two | 30 | below_40_mph | 70031 | 8.353846 | 13.364526 |
| current | two | 60 | below_40_mph | 69632 | 10.557604 | 16.215048 |
| current | two | 60 | drop_at_least_10_mph | 35479 | 16.434065 | 20.948835 |
| wide | one | 15 | below_40_mph | 70133 | 6.482500 | 10.555967 |
| wide | one | 30 | below_40_mph | 70031 | 8.428868 | 13.484508 |
| wide | one | 60 | below_40_mph | 69632 | 10.638969 | 16.408017 |
| wide | one | 60 | drop_at_least_10_mph | 35479 | 16.418974 | 21.048062 |
| wide | two | 15 | below_40_mph | 70133 | 6.405030 | 10.460652 |
| wide | two | 30 | below_40_mph | 70031 | 8.302068 | 13.317678 |
| wide | two | 60 | below_40_mph | 69632 | 10.453876 | 16.144924 |
| wide | two | 60 | drop_at_least_10_mph | 35479 | 16.252007 | 20.844880 |

CSV files retain per-seed and per-horizon results for both budgets. `paired_deltas.csv` uses candidate minus baseline, so negative values favor the candidate. `budget_deltas.csv` uses full minus prefix. `curves.png` shows every validation curve. Each run has history, environment, convergence, runtime and both selection records. Raw data and large retained arrays are remote only.

A final source audit matched every executable Python source and the training runner to the frozen protocol. The archive also contains macOS AppleDouble metadata entries named `._*.py`; the remote glob included their hashes in the protocol. They are packaging metadata, not imported Python modules. `source_verification.json` distinguishes these entries. The frozen protocol is preserved unchanged.
