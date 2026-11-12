# Experiment provenance

The baseline pipeline and original traffic-condition analysis were present before integration. Saif's classical-model implementation is attributed in integration commit `e62c1aa`, which records source commit `ce18f4c1d603e3a2b413ce7233eb59aefbfa7baf`.

For the follow-up work, the user asked Codex to finish classical-model checks and experiments and complete exploratory analysis. Codex assisted with full-calendar-week profile exclusion, fallback and boundary regression tests, experiment execution, threshold and outage audits, observed-only neighbour lag analysis, result summaries, figures, and draft report sections. The retained code and measured outputs are in this repository. This note records tool assistance; it does not assign member contributions beyond the source attribution in git history. Members should review the methods and conclusions and add their own contribution statements before submission.

The fixed low-speed threshold was retained for a common absolute interpretation and continuity, using training-only comparison statistics. The benchmark split was retained for comparability despite eleven shared target timestamps per boundary. Boosting settings are selected on the profile feature set and reused for ablations. These choices are documented in the report sections rather than treated as evidence of optimality.
