# Remaining project work

Status updated on 12 November 2026 against the local project proposal and assessment brief. This list describes repository evidence, not work that may exist elsewhere.

## Integrated in commit 1

Saif's `task2-models` commit `ce18f4c1d603e3a2b413ce7233eb59aefbfa7baf` adds matching-slot profile self-exclusion, validation tuning, neighbour features, ablation code, and seven synthetic model tests. Source: https://github.com/saifho/elec3544-traffic-forecast/tree/task2-models. His `main` at `6cca256` contains the original baseline plus the earlier task brief. Our later error analysis is retained.

The imported implementation changes three files: `src/traffic/models.py`, `scripts/evaluate.py`, and `tests/test_models.py`. It supplies no tuning results, ablation results, updated metrics, or classical-model report section. Existing committed results still describe the old models.

## Classical experiments and EDA

The follow-up implementation excludes each target's whole calendar week from both training profile features and fallback means. Regression tests cover sparse slots, a sensor with no observations outside that week, insufficient data, and forecasts crossing week boundaries. The benchmark split is retained with eleven shared target timestamps per boundary; this is a disclosed limitation, not a claim of fully disjoint evaluation.

See [classical-model methods and results](classical_models.md) for validation selection, test metrics, feature ablations, profile comparisons, and baseline stability checks. See [exploratory analysis](eda.md) for threshold comparison, outage decomposition, spatial timing results, and figures. Experimental artifacts live in `outputs/results/` and `outputs/eda/`. The condition analysis covers all seven classical models. [Experiment provenance](experiment_notes.md) records source attribution and foundation-model assistance.

The selected low-speed rule is 40 mph. The training-only relative rule remains a sensitivity comparison. The lag search does not establish upstream/downstream propagation. Boosting uses settings selected for the profile feature set across ablations; separately tuning every ablation and a purged-split sensitivity study remain optional extensions.

## Implement the central model comparison

1. Add a temporal neural baseline, such as a shared per-sensor GRU. Use the existing windows, training normalizer, and masked targets. Select settings and early stopping on validation data. Save seeds, settings, checkpoints, training curves, metrics, and predictions in the existing alignment format.
2. Add graph-based temporal modelling using the same data and forecast horizons. Document adjacency direction, self-loops, and normalization.
3. Compare temporal-only and temporal-plus-graph models under comparable training settings. Include a no-graph ablation to isolate the spatial contribution. Report MAE and RMSE at 15, 30, and 60 minutes, alongside the strongest classical models.

There is currently no neural-model module, training script, neural dependency, checkpoint, or neural result in this repository. These models are central to the proposal's research question.

## Complete evaluation and submission

- Extend the seven-classical-model congestion analysis to the final neural models. Add error by time of day and sensor, plus representative prediction-versus-observation traces and failure cases. Account for missing targets and dependence between overlapping windows.
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
