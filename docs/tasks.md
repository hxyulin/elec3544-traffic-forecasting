# Remaining project work

Status assessed on 12 November 2026 against the local project proposal and assessment brief. This list describes repository evidence, not work that may exist elsewhere.

## Integrated in commit 1

Saif's `task2-models` commit `ce18f4c1d603e3a2b413ce7233eb59aefbfa7baf` adds matching-slot profile self-exclusion, validation tuning, neighbour features, ablation code, and seven synthetic model tests. Source: https://github.com/saifho/elec3544-traffic-forecast/tree/task2-models. His `main` at `6cca256` contains the original baseline plus the earlier task brief. Our later error analysis is retained.

The imported implementation changes three files: `src/traffic/models.py`, `scripts/evaluate.py`, and `tests/test_models.py`. It supplies no tuning results, ablation results, updated metrics, or classical-model report section. Existing committed results still describe the old models.

## Finish the classical experiments

- Check profile exclusion before final runs. The implementation removes the matching slot's own reading, rather than all readings from that calendar week. Its sensor-mean fallback still includes the reading when no other week supplies that slot. Add a sparse-slot regression check and exclude the held-out week from the fallback. Decide whether the origin profile should also exclude the target's whole week to meet the original task literally.
- Document the existing split boundary overlap: adjacent train/validation windows share target timestamps. Decide whether to preserve the benchmark split with an explicit limitation or use a purged split for a sensitivity experiment. Keep all model comparisons on the same split.
- Run `uv run scripts/evaluate.py --tune`, then `uv run scripts/evaluate.py`, then `uv run scripts/evaluate.py --ablation`. Retain `tuning.csv`, `best_params.json`, updated `metrics.csv`, and `ablation.csv` as experimental evidence. Test results must not drive model selection.
- Record that boosting uses settings selected for the profile feature set across all ablations. Tune each feature set separately only if the comparison is intended to measure each set's best validation-selected performance.
- Check that persistence and historical-average scores remain unchanged. Compare corrected and leaky profile rows and quantify the neighbour contribution at every horizon.
- Extend `scripts/error_analysis.py` beyond its hard-coded five models, regenerate its outputs, and replace the old profile-leak limitation text when those outputs come from the corrected models. Do not label old prediction files as corrected results.
- Write the classical-model report section with the settings, feature definitions, ablations, measured changes, and limitations.

## Complete exploratory analysis

- Compare the provisional 40 mph rule with a per-sensor relative threshold, such as 60% of the sensor's training-only 85th-percentile speed. Report flagged readings and sensors, map them, and save the chosen rule in `outputs/eda/summary.json`.
- Explain simultaneous network outages, long sensor outages, and the higher test missingness. Distinguish observed patterns from unverified explanations.
- Measure lagged correlations of speed deviations for neighbouring sensors. Check road direction before interpreting any lead as traffic propagation.
- Write the EDA report section with the selected figures and reasons for the data decisions.

## Implement the central model comparison

1. Add a temporal neural baseline, such as a shared per-sensor GRU. Use the existing windows, training normalizer, and masked targets. Select settings and early stopping on validation data. Save seeds, settings, checkpoints, training curves, metrics, and predictions in the existing alignment format.
2. Add graph-based temporal modelling using the same data and forecast horizons. Document adjacency direction, self-loops, and normalization.
3. Compare temporal-only and temporal-plus-graph models under comparable training settings. Include a no-graph ablation to isolate the spatial contribution. Report MAE and RMSE at 15, 30, and 60 minutes, alongside the strongest classical models.

There is currently no neural-model module, training script, neural dependency, checkpoint, or neural result in this repository. These models are central to the proposal's research question.

## Complete evaluation and submission

- Extend the existing congestion analysis to all final models and the chosen threshold. Add error by time of day and sensor, plus representative prediction-versus-observation traces and failure cases. Account for missing targets and dependence between overlapping windows.
- Produce final comparison tables and figures from saved results. State whether spatial information helps, where it helps, and where it fails. Keep published reference scores separate from our own measurements.
- Write the final report connecting the question, data, cleaning decisions, model iterations, results, and limitations. Existing checkpoint PDFs are not evidence of a finished final report.
- Create presentation slides and rehearse the explanation of the model changes and evidence. No presentation source is present here.
- Record each member's contributions and foundation-model use. Preserve Saif's source attribution in the integration commit.
- Document reproducible commands, dependency versions, seeds, and expected output files. Run the final validation after implementation.

PEMS-BAY is optional in the proposal. Prioritise the METR-LA neural and graph comparison before adding another dataset. New live data is encouraged by the assessment brief but is not required.

## Sources

- `../Group project/Group_30_Traffic_Forecasting_Project_Proposal.docx`: research question, model sequence, horizons, error-analysis dimensions, foundation-model use, and six-week plan.
- `../Group project/group-project-assessment-standard-2.pdf`: workload evidence, reasoning about model iterations, report, presentation, and optional new-data evaluation.
- Saif's `docs/tasks.md` at upstream `6cca256`: detailed EDA and classical-model task requirements. The source files above live beside the repository and are not included in a clone.
