# Remaining project work

Status updated on 12 November 2026 against the local project proposal and assessment brief. This list describes repository evidence, not work that may exist elsewhere.

## Integrated in commit 1

Saif's `task2-models` commit `ce18f4c1d603e3a2b413ce7233eb59aefbfa7baf` adds matching-slot profile self-exclusion, validation tuning, neighbour features, ablation code, and seven synthetic model tests. Source: https://github.com/saifho/elec3544-traffic-forecast/tree/task2-models. His `main` at `6cca256` contains the original baseline plus the earlier task brief. Our later error analysis is retained.

The imported implementation changes three files: `src/traffic/models.py`, `scripts/evaluate.py`, and `tests/test_models.py`. It supplies no tuning results, ablation results, updated metrics, or classical-model report section. At integration time, the saved results still described the old models; the follow-up experiments below replaced them.

## Classical experiments and EDA

The follow-up implementation excludes each target's whole calendar week from both training profile features and fallback means. Regression tests cover sparse slots, a sensor with no observations outside that week, insufficient data, and forecasts crossing week boundaries. The benchmark split is retained with eleven shared target timestamps per boundary; this is a disclosed limitation, not a claim of fully disjoint evaluation.

See [classical-model methods and results](classical_models.md) for validation selection, test metrics, feature ablations, profile comparisons, and baseline stability checks. See [exploratory analysis](eda.md) for threshold comparison, outage decomposition, spatial timing results, and figures. Experimental artifacts live in `outputs/results/` and `outputs/eda/`. The condition analysis includes all recorded classical and neural models. [Experiment provenance](experiment_notes.md) records source attribution and foundation-model assistance.

The selected low-speed rule is 40 mph. The training-only relative rule remains a sensitivity comparison. The lag search does not establish upstream/downstream propagation. Boosting uses settings selected for the profile feature set across ablations; separately tuning every ablation and a purged-split sensitivity study remain optional extensions.

## Central model comparison

1. The shared per-sensor GRU is implemented. See [temporal baseline](temporal_model.md) for validation selection, results, learning curves, and checkpoint reproduction. Additional seeds and a wider search remain optional follow-up experiments.
2. The GRU with learned graph filtering and its separately trained identity-graph control are implemented. See [graph methods and results](graph_model.md) for directed supports, self-loop handling, normalization, matched training, multi-horizon errors, and limits.
3. The temporal-versus-graph comparison includes MAE and RMSE at 15, 30, and 60 minutes, low-speed errors, and classical neighbour boosting. Checkpoints, histories, and control verification are saved under `outputs/neural/`.

The core planned model sequence is now implemented. Repeated seeds, a larger search, deeper graph propagation, and purged-split sensitivity experiments remain optional extensions. The planned error analysis is complete. Prioritize final submission materials before widening the model search.

## Further neural investigation

A [separate neural investigation](neural_investigation.md) tests sensor/time identities, causal gap age, a residual MLP, and resampled training rows. A matched sensor-identity ablation and second-seed validation replication are complete. The selected seed-42 model has an exploratory test comparison, separate from the recorded baseline results. Its MAE gain comes with higher RMSE and low-speed error than gradient boosting + neighbours. Decisions about adoption, a matched stronger graph model, and independent-period validation remain follow-up work.

## Complete evaluation and submission

- Detailed error analysis is complete. See [error analysis](error_analysis.md) for sensor and target-time scores, input availability, speed changes, training-rank-selected traces, and explicit post-hoc failure cases. All groups exclude missing targets and retain observed counts; dependence between windows and sensors is documented.
- Comparison tables, ablations, and error-analysis figures are saved. Select the final report/presentation figures and state where spatial inputs help and fail. Keep published reference scores separate from our own measurements.
- Write the final report connecting the question, data, cleaning decisions, model iterations, results, and limitations. Existing checkpoint PDFs are not evidence of a finished final report.
- Create presentation slides and rehearse the explanation of the model changes and evidence. No presentation source is present here.
- Record each member's contributions and foundation-model use. Preserve Saif's source attribution in the integration commit.
- Document reproducible commands, dependency versions, seeds, and expected output files. Run the final validation after implementation.

PEMS-BAY is optional in the proposal. Prioritise the METR-LA neural and graph comparison before adding another dataset. New live data is encouraged by the assessment brief but is not required.

## Sources

- `../Group project/Group_30_Traffic_Forecasting_Project_Proposal.docx`: research question, model sequence, horizons, error-analysis dimensions, foundation-model use, and six-week plan.
- `../Group project/group-project-assessment-standard-2.pdf`: workload evidence, reasoning about model iterations, report, presentation, and optional new-data evaluation.
- Saif's `docs/tasks.md` at upstream `6cca256`: detailed EDA and classical-model task requirements. The source files above live beside the repository and are not included in a clone.
