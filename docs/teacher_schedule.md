# Predefined teacher-forcing schedule experiment

AI-assisted exploratory experiments. Inspect the implementation and evidence before drawing conclusions.

The hypothesis is that exposing this compact encoder-decoder to zero teacher forcing before epoch 50 may improve free-running validation forecasts. The prior inverse-sigmoid policy still forces roughly 51% of decoder transitions near the cap. That observation does not establish the cause of forecast error.

The frozen factorial crosses two schedules, graph enabled or disabled, and seeds 42, 43, 44. Schedule A is `2000/(2000+exp(u/2000))`. Schedule B is `max(0,1-u/(30*updates_per_epoch))`. The update index is zero based before each optimizer update. Each fit runs exactly 50 epochs. Schedule B reaches zero at the beginning of epoch 31 and receives 20 full epochs at zero. No schedule is selected or tuned from these results.

The new runner imports the unchanged model. It preserves the one-layer hidden-32 architecture with 32,385 parameters, independent direction polynomial supports, speed and observed flag encoder input, speed-only decoder input, and all-12-target masked loss. It preserves AdamW learning rate .001, weight decay .0001, gradient clip 1, and validation-driven ReduceLROnPlateau factor .5, patience 3, including its existing defaults. The comparison therefore includes optimizer trajectory adaptation through that validation scheduler.

Each epoch samples 4,830 distinct full-network training windows. NumPy partitions them into 302 batches of at most 16 using the prior array-split policy. The train-only normalizer, causal fallback, purged training windows, reused benchmark validation set, and missing-target prediction fallback remain unchanged. No original test set is evaluated. Evaluation calls the model without targets and uses no target masks for decoder input.

Training uses a dedicated CPU generator seeded with seed+1000. The unchanged forward draws one scalar for each of 11 decoder transitions, even when probability is zero. A separate audit generator measures threshold decisions without changing the model's draw stream. Histories record sample and generator hashes, probability boundaries, eligible and forced scalar transitions, and forced observed target entries. A scalar forcing decision applies to the whole batch; missing entries still use predictions.

All fits select the minimum mean validation MAE over horizons 3, 6, and 12 across all 50 epochs. Additional checkpoints retain the best epochs at or below 30 and above 30. Every retained checkpoint is restored and its prediction array checked exactly. Phase summaries do not change selection. The diagnostic subgroups are observed speed below 40 mph at each scored horizon and an observed drop of at least 10 mph at horizon 12, as in the prior study.

Each fit has a 30-minute external timeout. The sequential tmux sequence has a three-hour external timeout and stops on failure. Runtime recorded inside the runner starts after data/model/optimizer setup and GPU synchronization, and ends after retained checkpoint restoration and array checks. The sequence log records UTC launch and exit boundaries, including process setup time.

Teacher forcing changes the training decoder input distribution. At positive probability, some decoder inputs come from observed targets; at zero, all come from model predictions. Validation is always free running. This experiment tests two predefined policies under one compact architecture and one reused validation set. It does not establish pristine holdout performance, significance, general convergence, or a conclusion about DCRNN architectures in general. Prior MLP results are context only because inputs, loss horizons, batching, initialization, sampling, and compute differ.

## Results

All 12 fresh fits completed 50 epochs. The sequence ran from 01:28:13 to 02:49:56 UTC on 2026-11-13, including process setup and checkpoint checks: 4,903 seconds. All 36 retained checkpoint restorations produced exactly matching prediction arrays. Scores were recomputed from the arrays. All six legacy trajectories match their previous runs over every shared epoch for loss, validation MAE, learning rate, sampled windows, observed exposure, and generator state. Legacy seed 42 graph-enabled also exactly reproduces the selected epoch, metrics, and prediction-array hash. Source, data, and initial parameter hashes match the prior study.

Both local and GPU-workstation test runs passed 59 tests. Lint and format checks passed. The first remote test attempt lacked historical Git objects; importing the lineage bundle resolved that engineering issue before the final passing checks. The old source/output integrity snapshot was captured during the first new fit, rather than before its launch. Every file in that snapshot passed the final hash check. The old study was never used as a write destination.

| Schedule | Graph | Seed | Macro MAE | Macro RMSE | Best epoch |
|---|---|---:|---:|---:|---:|
| A | enabled | 42 | 3.283879 | 6.743248 | 49 |
| A | disabled | 42 | 3.541362 | 7.378608 | 50 |
| A | enabled | 43 | 3.210277 | 6.582795 | 50 |
| A | disabled | 43 | 3.538751 | 7.373135 | 50 |
| A | enabled | 44 | 3.203758 | 6.556665 | 49 |
| A | disabled | 44 | 3.545737 | 7.417988 | 48 |
| B | enabled | 42 | 3.057000 | 6.108115 | 49 |
| B | disabled | 42 | 3.530874 | 7.321521 | 50 |
| B | enabled | 43 | 3.053290 | 6.148076 | 47 |
| B | disabled | 43 | 3.529632 | 7.307397 | 49 |
| B | enabled | 44 | 3.087725 | 6.223086 | 48 |
| B | disabled | 44 | 3.531576 | 7.327955 | 46 |

Schedule differences below are B minus A. Negative values favor B. SD is the sample standard deviation across three paired seeds. The sign count reports negative/positive differences.

| Graph | Horizon | MAE mean ± SD | RMSE mean ± SD | MAE signs | RMSE signs |
|---|---|---:|---:|---:|---:|
| disabled | 12 | -0.027709 ± 0.004932 | -0.151030 ± 0.035563 | 3/0 | 3/0 |
| disabled | 3 | -0.000234 ± 0.001570 | -0.013941 ± 0.005428 | 2/1 | 3/0 |
| disabled | 6 | -0.005825 ± 0.002748 | -0.047887 ± 0.012746 | 3/0 | 3/0 |
| disabled | macro | -0.011256 ± 0.002607 | -0.070953 ± 0.017081 | 3/0 | 3/0 |
| enabled | 12 | -0.346404 ± 0.093062 | -0.865316 ± 0.261797 | 3/0 | 3/0 |
| enabled | 3 | -0.027711 ± 0.029965 | -0.149217 ± 0.077197 | 3/0 | 3/0 |
| enabled | 6 | -0.125784 ± 0.045654 | -0.388899 ± 0.122279 | 3/0 | 3/0 |
| enabled | macro | -0.166633 ± 0.056049 | -0.467811 ± 0.153476 | 3/0 | 3/0 |

Graph contrasts are enabled minus disabled. Negative values favor graph use.

| Schedule | Horizon | MAE mean ± SD | RMSE mean ± SD | MAE signs | RMSE signs |
|---|---|---:|---:|---:|---:|
| A | 12 | -0.484463 ± 0.079908 | -1.040226 ± 0.199920 | 3/0 | 3/0 |
| A | 3 | -0.164621 ± 0.019439 | -0.510054 ± 0.048834 | 3/0 | 3/0 |
| A | 6 | -0.278851 ± 0.036932 | -0.736742 ± 0.098267 | 3/0 | 3/0 |
| A | macro | -0.309312 ± 0.045391 | -0.762341 ± 0.115555 | 3/0 | 3/0 |
| B | 12 | -0.803158 ± 0.026536 | -1.754513 ± 0.096589 | 3/0 | 3/0 |
| B | 3 | -0.192098 ± 0.011555 | -0.645330 ± 0.032319 | 3/0 | 3/0 |
| B | 6 | -0.398810 ± 0.017570 | -1.077753 ± 0.035892 | 3/0 | 3/0 |
| B | macro | -0.464689 ± 0.018088 | -1.159199 ± 0.054268 | 3/0 | 3/0 |

Every whole-run selected checkpoint comes from epochs 46–50. The graph-enabled B best epochs are 49, 47, and 48. All selected checkpoints lie after epoch 30, but both schedules also improve after epoch 30. The phase comparison therefore does not isolate the causal effect of switching to zero from additional training or learning-rate adaptation. The B-versus-A result tests the entire predefined schedule.

The per-seed horizon scores, paired differences, phase-best metrics, actual exposure, diagnostic subgroups, histories, protocol, restoration records, and hashes are in `outputs/experiments/teacher_schedule`. The A and B subdirectories contain validation curves with the epoch-30 boundary marked. No checkpoint or diagnostic subgroup was used to tune the schedules.


## Exposure and phase detail

Each fit used 241,500 sampled training windows and 166,100 eligible scalar decoder transitions. The final A probability was 0.512848; B began at 1 and ended at 0. A forced 151,417, 151,204, and 151,360 transitions for seeds 42–44. B forced 50,077, 49,834, and 49,809. Counts match between graph controls. Every B post-30 counter is zero. Window hashes, observed-target exposure, and generator states match across all four variants within each seed.

| Schedule | Graph | Best ≤30 mean MAE | Best >30 mean MAE |
|---|---|---:|---:|
| A | disabled | 3.553057 | 3.541950 |
| A | enabled | 3.314749 | 3.232638 |
| B | disabled | 3.540932 | 3.530694 |
| B | enabled | 3.155394 | 3.066005 |

Diagnostic subgroup scores below are means across seeds, without selection on these scores. The horizon-12 drop subgroup uses a drop of at least 10 mph from the last observed input speed. The graph-disabled linear policy improves overall macro MAE but worsens these difficult subgroup averages.

| Schedule | Graph | Subgroup | Horizon | MAE | RMSE |
|---|---|---|---:|---:|---:|
| A | disabled | below_40_mph | 3 | 7.359302 | 11.928653 |
| A | enabled | below_40_mph | 3 | 6.670557 | 10.847885 |
| A | disabled | below_40_mph | 6 | 9.882514 | 15.435669 |
| A | enabled | below_40_mph | 6 | 8.995571 | 14.224646 |
| A | disabled | below_40_mph | 12 | 13.799694 | 19.993243 |
| A | enabled | below_40_mph | 12 | 12.480282 | 18.483798 |
| A | disabled | drop_at_least_10_mph | 12 | 24.639570 | 27.805791 |
| A | enabled | drop_at_least_10_mph | 12 | 21.623910 | 25.282225 |
| B | disabled | below_40_mph | 3 | 7.417844 | 11.980441 |
| B | enabled | below_40_mph | 3 | 6.643413 | 10.662893 |
| B | disabled | below_40_mph | 6 | 10.161632 | 15.661270 |
| B | enabled | below_40_mph | 6 | 8.638991 | 13.644639 |
| B | disabled | below_40_mph | 12 | 14.662194 | 20.586402 |
| B | enabled | below_40_mph | 12 | 11.392205 | 17.087696 |
| B | disabled | drop_at_least_10_mph | 12 | 25.200535 | 28.259747 |
| B | enabled | drop_at_least_10_mph | 12 | 18.038286 | 22.205417 |
