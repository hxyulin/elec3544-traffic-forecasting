"""Summarize the isolated neural investigation without changing official results."""

from __future__ import annotations

import hashlib
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from traffic.data import ROOT
from traffic.reporting import MODEL_LABELS, markdown_table

OUT = ROOT / "outputs/experiments/identity_mlp"
LABELS = {
    **MODEL_LABELS,
    "mlp_shared": "MLP without sensor ID",
    "mlp_identity": "MLP with sensor ID",
}


def comparison_table(frame, metrics=("mae",)):
    table = frame.pivot(index="model", columns="minutes", values=list(metrics))
    table.columns = [f"{metric.upper()} {h} min" for metric, h in table.columns]
    table.index = table.index.map(LABELS)
    return markdown_table(table.map(lambda v: f"{v:.3f}").reset_index())


def main():
    seed = OUT / "seed_42"
    selections = pd.read_csv(seed / "selection.csv")
    validation = pd.read_csv(seed / "validation.csv")
    tested = pd.read_csv(seed / "test.csv")
    metadata = json.loads((seed / "test_evaluation.json").read_text())
    official_path = ROOT / "outputs/results/metrics.csv"
    official = pd.read_csv(official_path)
    refs = ["gru", "graph_gru", "gradient_boosting_neighbours"]
    vbase = official[(official.split == "val") & official.model.isin(refs)].copy()
    candidates = validation[validation.condition == "all_observed"].copy()
    candidates["model"] = "mlp_" + candidates.variant
    val = pd.concat([vbase, candidates], ignore_index=True)
    winner = "mlp_" + metadata["selected_variant"]
    tnew = tested[tested.condition == "all_observed"].copy()
    tnew["model"] = winner
    test = pd.concat(
        [official[(official.split == "test") & official.model.isin(refs)], tnew],
        ignore_index=True,
    )
    old_conditions = pd.read_csv(ROOT / "outputs/error_analysis/condition_metrics.csv")
    low_base = old_conditions[
        (old_conditions.condition == "below_40_mph") & old_conditions.model.isin(refs)
    ]
    low_new = tested[tested.condition == "below_40_mph"].copy()
    low_new["model"] = winner
    low = pd.concat([low_base, low_new], ignore_index=True)
    val[["model", "minutes", "mae", "rmse", "mape", "n"]].to_csv(
        OUT / "validation_comparison.csv", index=False
    )
    test[["model", "minutes", "mae", "rmse", "mape", "n"]].to_csv(
        OUT / "test_comparison.csv", index=False
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ["mae", "rmse"]):
        for name in refs + [winner]:
            g = test[test.model == name].sort_values("minutes")
            ax.plot(g.minutes, g[metric], marker="o", label=LABELS[name])
        ax.set(
            xlabel="Forecast horizon (minutes)",
            ylabel=f"Test {metric.upper()} (mph)",
            xticks=[15, 30, 60],
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "test_comparison.png", dpi=160)
    plt.close(fig)
    histories = {
        v: pd.read_csv(seed / v / "history.csv") for v in ["shared", "identity"]
    }
    common = min(len(h) for h in histories.values())
    pd.testing.assert_series_equal(
        histories["shared"].sample_sha256[:common],
        histories["identity"].sample_sha256[:common],
        check_names=False,
    )
    assert selections.parameters.nunique() == 1
    means = val.groupby("model").mae.mean()
    identity_gain = 100 * (1 - means.mlp_identity / means.mlp_shared)
    test_pivot = test.set_index(["model", "minutes"])
    mae_gain = [
        100
        * (
            1
            - test_pivot.loc[(winner, h), "mae"]
            / test_pivot.loc[("gradient_boosting_neighbours", h), "mae"]
        )
        for h in (15, 30, 60)
    ]
    replication = OUT / "seed_43/selection.csv"
    if not replication.exists():
        raise FileNotFoundError(
            "Finish the unchanged-settings validation replication before writing the final report"
        )
    repeated = pd.read_csv(replication)
    repeat_value = float(repeated[repeated.variant == "identity"].iloc[0].val_mae)
    verification = {
        "matched_training_sample_epochs": common,
        "parameters_per_variant": int(selections.parameters.iloc[0]),
        "official_metrics_sha256": hashlib.sha256(
            official_path.read_bytes()
        ).hexdigest(),
        "selection": "seed-42 winner chosen on mean validation MAE; seed 43 is a validation-only replication",
        "test_model": winner,
        "reference_values": "existing metrics.csv for original models; raw separate experiment tables for candidates",
        "test_policy": metadata["test_policy"],
    }
    (OUT / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    subset = (
        tested[tested.condition.isin(["complete_input", "missing_input"])]
        .pivot(index="condition", columns="minutes", values="mae")
        .round(3)
        .reset_index()
    )
    subset.columns = ["Input availability", "MAE 15 min", "MAE 30 min", "MAE 60 min"]
    report = f"""# Strengthening the neural baseline

A residual MLP with sensor and target-time embeddings improves on our original neural models in this investigation. It also has lower aggregate test MAE than gradient boosting + neighbours at all three horizons. Boosting retains lower RMSE and lower MAE on low-speed targets. This is a measured tradeoff, not a universal model ranking.

The published results remain unchanged. New experiments are stored under `outputs/experiments/identity_mlp/`. This investigation starts from the existing benchmark split, with the same twelve input steps, three horizons, training normalizer, and observed-target mask.

## Diagnosis of the original neural comparison

- The temporal GRU has 4,035 parameters shared across all 207 sensors, with no sensor identity or weekly-profile feature. Two sensors with identical input readings and calendar features receive identical forecasts. The classical boosting model receives sensor identity and a training weekly profile, so its feature set is richer.
- The graph variant adds only eight coefficients. It compresses incoming/outgoing speed and flag signals back into two channels before recurrence. This tests a small input filter, not a full graph neural forecasting architecture.
- The old GRUs reuse a fixed one-million-row sample, about one fifth of all training window/sensor rows, for at most 25 epochs. Some trials improve at the cap. This leaves training-data coverage and optimization budget as possible limits.
- An observation flag identifies a missing input but does not tell the model how long its filled value has been stale. The earlier analysis highlighted missing-history failures, motivating a causal gap-age feature. Its contribution is not isolated by the identity ablation below.

These are architectural and protocol observations. They do not establish that any one issue alone caused the old performance gap.

## Experiment and matched identity ablation

The candidate is inspired by [STID](https://arxiv.org/abs/2208.05233), which studies sensor/time identities in MLP forecasting. Our implementation is a custom horizon-conditioned residual MLP, not a reproduction of the published model or its reported scores.

The input is the twelve normalized speeds and observation flags plus `log(1 + consecutive missing steps)` at the forecast origin. Leading gaps count from the beginning of the series; gap age uses no future observations. A 64-unit history projection is concatenated with a 16-dimensional sensor embedding, target time-of-day embedding (8 dimensions), target day-of-week embedding (4), and horizon embedding (4). Target calendar values are known at forecast time. Two residual blocks operate on the resulting 96 channels, with ReLU and dropout 0.1. A shared linear head predicts each horizon's residual around the local persistence value. Its zero initialization reproduces persistence before training.

Both variants have {int(selections.parameters.iloc[0]):,} parameters. The control multiplies sensor embeddings by zero, while retaining calendar identities, gap age, capacity, optimizer, and training budget. Thus the comparison isolates access to sensor identity within this architecture. It does not isolate gap age, calendar embeddings, MLP architecture, or increased training budget relative to the old GRU.

Both seed-42 variants use AdamW (initial learning rate 0.001, weight decay 0.0001), gradient clipping at 1, batch size 2048, masked MAE divided by the training speed standard deviation, and up to 50 epochs. ReduceLROnPlateau halves the rate after validation plateaus; early stopping uses patience eight. Each epoch draws a fresh uniform sample of one million training window/sensor rows without replacement. The two variants have identical initialized parameter values and matched sample hashes for all {common} common epochs. Checkpoints minimize mean validation MAE across the horizons. Training budgets differ from the original GRU experiment and are disclosed rather than attributed solely to architecture.

Validation selections:

{markdown_table(selections[["variant", "seed", "best_epoch", "val_mae", "parameters"]].round(4))}

Validation MAE by horizon:

{comparison_table(val)}

Mean validation MAE falls from {means.mlp_shared:.4f} mph without sensor identity to {means.mlp_identity:.4f} with it, an {identity_gain:.2f}% reduction under matched settings. The original GRU is {means.gru:.4f}, the graph GRU {means.graph_gru:.4f}, and gradient boosting + neighbours {means.gradient_boosting_neighbours:.4f} mph. Reference metrics are stored to four decimals; candidate scores retain full precision in their experiment CSVs.

![Matched validation learning curves](../outputs/experiments/identity_mlp/seed_42/learning_curves.png)

A second seed (43) repeats the identity model with unchanged settings and validation-only scoring. Its best mean validation MAE is {repeat_value:.4f} mph, compared with seed 42's {means.mlp_identity:.4f}. It was not used to replace the seed-42 test candidate and has no test evaluation. Two seeds are a robustness check, not a confidence interval or a full seed study.

## Selected candidate on the existing test period

Only the seed-42 validation winner was evaluated on test. No test result was used to change its settings or select another checkpoint. Checkpoint restoration first reproduced the selected validation score. Predictions use the existing window-major `(window, horizon, sensor)` format and are saved separately from official model predictions.

{comparison_table(test, ("mae", "rmse"))}

Relative to gradient boosting + neighbours, candidate MAE is lower by {", ".join(f"{v:.2f}%" for v in mae_gain)} at 15, 30 and 60 minutes. Its RMSE is higher at every horizon. Lower mean absolute error therefore does not imply fewer large misses.

![Selected candidate compared on MAE and RMSE](../outputs/experiments/identity_mlp/test_comparison.png)

Low-speed MAE, grouping by observed targets below 40 mph:

{comparison_table(low)}

The candidate improves low-speed error over the previous neural models, but boosting remains better in this subset at every horizon. The training objective optimizes average masked MAE; it does not specifically optimize congestion errors or RMSE.

Candidate MAE by own-sensor input availability:

{markdown_table(subset)}

These subgroups are descriptive and have different sensor and traffic compositions. Gap age was supplied to both MLP variants; an age-feature ablation would be required to measure its independent contribution.

## What to do next

1. Use the identity-aware MLP as the stronger neural reference. Keep the old GRU experiment as evidence of development. Repeat further seeds and evaluate on an independently reserved period before making broad generalization claims.
2. Test spatial modelling on top of this stronger reference. Preserve local and neighbour information as separate hidden channels, add observation/gap-aware message weights, and compare against a matched identity-only control. The original two-channel graph filter may discard information before recurrence; that is a hypothesis for a controlled ablation.
3. If a fuller graph architecture is justified, [Graph WaveNet](https://arxiv.org/abs/1906.00121) combines adaptive dependencies with dilated temporal convolutions. It addresses learned spatial relationships and temporal context beyond our small graph filter. A faithful implementation and matched evaluation would be a separate experiment.
4. Investigate the RMSE/congestion tradeoff using validation data: compare objectives such as Huber or mixed MAE/MSE and inspect a validation Pareto frontier. Define the intended metric before selection. Do not use these test results to tune the candidate retrospectively.
5. Extend training-data coverage or the epoch budget based on validation learning curves. The no-identity control's best checkpoint hits epoch 50; convergence is not established. Resampling improves coverage, but each epoch still uses a sample rather than a full pass through all rows.

## Limits

The test period had already been examined in earlier project analysis, and that analysis helped motivate the missing-gap feature. The new test comparison is exploratory evidence, not a fresh unseen holdout. The benchmark also retains eleven overlapping target timestamps at each split boundary. A new independent period or rolling-origin evaluation is needed for stronger claims about generalization.

Sensor identity is learned for the existing METR-LA nodes; unseen-sensor transfer is not tested. The MLP uses no road graph. Improvements from identity do not prove spatial propagation or road-topology benefits. The two-seed validation check, limited optimizer search, changed data budget, and feature differences must remain part of the interpretation.

## Reproduction

```sh
uv run --extra neural scripts/investigate_neural.py
uv run --extra neural scripts/investigate_neural.py --evaluate-only
uv run --extra neural scripts/evaluate_identity.py --seed 42
uv run --extra neural scripts/investigate_neural.py --seed 43 --variants identity
uv run scripts/summarize_identity.py
```

The contract records data hash, sensor order, horizons, normalizer, seed, optimizer settings and software versions. A different contract in an existing seed folder is rejected. Trial histories record each epoch's training-sample hash. Checkpoints are saved on validation improvement. All official result CSVs and original neural checkpoints are retained.
"""
    (ROOT / "docs/neural_investigation.md").write_text(report)
    print(comparison_table(test, ("mae", "rmse")))
    print(f"Seed-43 validation replication: {repeat_value:.6f}")


if __name__ == "__main__":
    main()
