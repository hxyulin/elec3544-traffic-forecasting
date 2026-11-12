"""Write the validation diagnosis and controlled neural follow-up report."""

from __future__ import annotations

import hashlib
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from traffic.data import ROOT
from traffic.reporting import markdown_table
from traffic.spatial_identity import NeighbourIdentityMLP

OUT = ROOT / "outputs/experiments/neural_followup"
LABELS = {
    "identity": "Identity MLP (MAE)",
    "huber": "Identity MLP (Huber)",
    "mixed": "Identity MLP (mixed loss)",
    "spatial_control": "Spatial control (zero messages)",
    "spatial": "Neighbour-aware MLP",
    "boosting": "Boosting + neighbours",
}
ORDER = list(LABELS)


def main():
    aggregate = pd.read_csv(OUT / "diagnosis/aggregate.csv")
    partitions = pd.read_csv(OUT / "diagnosis/partitions.csv")
    tails = pd.read_csv(OUT / "diagnosis/tails.csv")
    selection = pd.read_csv(OUT / "seed_42/selection.csv")
    if set(selection.variant) != {"huber", "mixed", "spatial", "spatial_control"}:
        raise ValueError("Finish all four validation trials before summarizing")
    means = aggregate.groupby("model")[["mae", "rmse"]].mean().loc[ORDER]
    low = partitions[
        (partitions.partition == "speed") & (partitions.group == "below_40")
    ]
    means["low_speed_mae"] = low.groupby("model").mae.mean()
    means["bias_60"] = aggregate[aggregate.minutes == 60].set_index("model").bias
    # Mean horizon metrics, not RMSE pooled over horizons.
    means["pareto_mae_rmse"] = False
    for model in ORDER:
        means.loc[model, "pareto_mae_rmse"] = not any(
            means.loc[other, "mae"] <= means.loc[model, "mae"]
            and means.loc[other, "rmse"] <= means.loc[model, "rmse"]
            and (
                means.loc[other, "mae"] < means.loc[model, "mae"]
                or means.loc[other, "rmse"] < means.loc[model, "rmse"]
            )
            for other in ORDER
            if other != model
        )
    means.to_csv(OUT / "comparison.csv")
    panels = []
    for metric, title in [
        ("mae", "Mean validation MAE"),
        ("rmse", "Mean validation RMSE"),
    ]:
        ceiling = float(np.ceil(means[metric].max()))
        bars = []
        for index, model in enumerate(ORDER):
            value = means.loc[model, metric]
            bars.append(
                f'<div class="row"><div class="label"><span>{LABELS[model]}</span><b>{value:.3f}</b></div>'
                f'<div class="track"><div style="width:{value / ceiling * 100:.3f}%;background:var(--chart-{index + 1})"></div></div></div>'
            )
        panels.append(
            f'<section><h3>{title}</h3><p class="axis">mph · lower is better · scale 0–{ceiling:g}</p>{"".join(bars)}</section>'
        )
    low_rows = "".join(
        f"<tr><td>{LABELS[model]}</td><td>{means.loc[model, 'low_speed_mae']:.3f}</td></tr>"
        for model in ORDER
    )
    html = (
        """<!doctype html><html><head><meta charset="utf-8"><style>
:root { color-scheme: light dark; }
body { font-family:var(--font-sans, sans-serif); color:var(--foreground); margin:0; }
.charts { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:24px; }
h3 { font-size:16px; margin:16px 0 4px; font-weight:600; }
p { font-size:13px; line-height:1.5; }
.axis,.note { color:var(--muted-foreground); }
.axis { margin:0 0 16px; }
.row { margin:0 0 17px; }
.label { display:flex; justify-content:space-between; gap:8px; font-size:12px; margin-bottom:6px; }
.label b { font-family:var(--font-mono,monospace); font-weight:500; }
.track { height:9px; background:var(--secondary); border-radius:3px; overflow:hidden; }
.track div { height:9px; border-radius:3px; }
table { border-collapse:collapse; width:100%; font-size:13px; margin-top:20px; }
th,td { padding:7px 0; border-bottom:1px solid var(--border); text-align:left; }
th:last-child,td:last-child { text-align:right; }
@media(max-width:580px) { .charts { grid-template-columns:1fr; gap:0; } }
</style></head><body><p>Validation only. Each value averages the 15-, 30- and 60-minute metric. All neural checkpoints were selected on mean MAE.</p><div class="charts">"""
        + "".join(panels)
        + """</div><table><thead><tr><th>Model</th><th>Mean low-speed MAE (mph)</th></tr></thead><tbody>"""
        + low_rows
        + """</tbody></table><p class="note">Low-speed targets are observed speeds below 40 mph. One seed per new trial; no new test evaluation.</p></body></html>"""
    )
    (OUT / "comparison.html").write_text(html)
    table = means.reset_index().rename(
        columns={
            "model": "Model",
            "mae": "Mean MAE",
            "rmse": "Mean RMSE",
            "low_speed_mae": "Mean low-speed MAE",
            "bias_60": "60-min bias",
            "pareto_mae_rmse": "MAE/RMSE frontier",
        }
    )
    table["Model"] = table.Model.map(LABELS)
    table["MAE/RMSE frontier"] = table["MAE/RMSE frontier"].map(
        {True: "yes", False: "no"}
    )
    tables = markdown_table(table.round(4))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for model in ORDER:
        measured = aggregate[aggregate.model == model].sort_values("minutes")
        congestion = low[low.model == model].sort_values("minutes")
        for ax, frame, metric in zip(
            axes, [measured, measured, congestion], ["mae", "rmse", "mae"]
        ):
            ax.plot(frame.minutes, frame[metric], marker="o", label=LABELS[model])
    for ax, title in zip(axes, ["MAE", "RMSE", "Low-speed MAE (target <40 mph)"]):
        ax.set(
            xlabel="Forecast horizon (minutes)",
            ylabel="Validation error (mph)",
            title=title,
            xticks=[15, 30, 60],
        )
        ax.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(OUT / "validation_comparison.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    transition = partitions[
        (partitions.partition == "transition") & (partitions.minutes == 60)
    ]
    groups = ["origin_missing", "drop_ge10", "rise_ge10", "stable_low", "stable_other"]
    names = [
        "Origin\nmissing",
        "Drop\n≥10 mph",
        "Rise\n≥10 mph",
        "Stable\nlow speed",
        "Stable\nother",
    ]
    for j, model in enumerate(["identity", "boosting", "spatial"]):
        frame = transition[transition.model == model].set_index("group").loc[groups]
        axes[0].bar(
            np.arange(5) + (j - 1) * 0.25,
            frame.sse_share * 100,
            width=0.25,
            label=LABELS[model],
        )
    axes[0].set(
        xticks=np.arange(5),
        xticklabels=names,
        ylabel="Share of squared error (%)",
        title="60-minute validation error contributions",
    )
    axes[0].legend(fontsize=8)
    for model in ["identity", "huber", "mixed", "spatial_control", "spatial"]:
        path = (
            ROOT / "outputs/experiments/identity_mlp/seed_42/identity/history.csv"
            if model == "identity"
            else OUT / f"seed_42/{model}/history.csv"
        )
        history = pd.read_csv(path)
        axes[1].plot(history.epoch, history.val_mae, label=LABELS[model])
    axes[1].set(
        xlabel="Epoch",
        ylabel="Mean validation MAE (mph)",
        title="Checkpoint selection curves",
    )
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "diagnosis_and_learning.png", dpi=160)
    plt.close(fig)
    histories = {
        v: pd.read_csv(OUT / f"seed_42/{v}/history.csv") for v in selection.variant
    }
    common = min(len(h) for h in histories.values())
    baseline = pd.read_csv(
        ROOT / "outputs/experiments/identity_mlp/seed_42/identity/history.csv"
    )
    for history in histories.values():
        n = min(len(history), len(baseline))
        pd.testing.assert_series_equal(
            history.sample_sha256[:n], baseline.sample_sha256[:n], check_names=False
        )
    torch.manual_seed(42)
    active = NeighbourIdentityMLP(207)
    torch.manual_seed(42)
    control = NeighbourIdentityMLP(207, False)
    for a, b in zip(active.parameters(), control.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    s = selection.set_index("variant")
    assert s.loc["spatial", "parameters"] == s.loc["spatial_control", "parameters"]
    winner = min(
        ["identity", "huber", "mixed", "spatial_control", "spatial"],
        key=lambda m: means.loc[m, "mae"],
    )
    verification = dict(
        scope="validation only; no new test scores",
        neural_winner_by_mean_validation_mae=winner,
        shared_sample_epochs_all_four=common,
        spatial_initial_parameters="exactly identical with seed 42",
        spatial_parameters=int(s.loc["spatial", "parameters"]),
        partition_verification=json.loads(
            (OUT / "diagnosis/verification.json").read_text()
        ),
        checkpoints={
            v: hashlib.sha256((OUT / f"seed_42/{v}/model.pt").read_bytes()).hexdigest()
            for v in selection.variant
        },
        baseline_metrics_sha256=hashlib.sha256(
            (ROOT / "outputs/results/metrics.csv").read_bytes()
        ).hexdigest(),
    )
    (OUT / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    base_transition = transition[transition.model == "identity"].set_index("group")
    drop = base_transition.loc["drop_ge10"]
    excess = transition.pivot(index="group", columns="model", values="sse")
    excess["identity_minus_boosting"] = excess.identity - excess.boosting
    excess["share_of_net_excess"] = (
        excess.identity_minus_boosting / excess.identity_minus_boosting.sum()
    )
    excess.to_csv(OUT / "squared_error_difference_60.csv")
    drop_excess = excess.loc["drop_ge10", "share_of_net_excess"]
    base_tails = tails[
        (tails.model == "identity") & (tails.minutes == 60) & (tails.fraction == 0.05)
    ].iloc[0]
    ranges = pd.read_csv(OUT / "diagnosis/prediction_ranges.csv")
    outside = ranges[(ranges.model == "identity") & (ranges.minutes == 60)].iloc[0]
    complete = partitions[
        (partitions.model == "identity")
        & (partitions.minutes == 60)
        & (partitions.partition == "input_missing")
        & (partitions.group == "none")
    ].iloc[0]
    spatial_gain = 100 * (
        1 - means.loc["spatial", "mae"] / means.loc["spatial_control", "mae"]
    )
    spatial_direction = "reduces" if spatial_gain >= 0 else "increases"
    spatial_drop = (
        transition[transition.model == "spatial"]
        .set_index("group")
        .loc["drop_ge10", "mae"]
    )
    control_drop = (
        transition[transition.model == "spatial_control"]
        .set_index("group")
        .loc["drop_ge10", "mae"]
    )
    details = aggregate[aggregate.model.isin(ORDER)].pivot(
        index="model", columns="minutes", values=["mae", "rmse"]
    )
    details = details.loc[ORDER]
    details.columns = [f"{metric.upper()} {h} min" for metric, h in details.columns]
    details.index = details.index.map(LABELS)
    transition_table = (
        transition[transition.model.isin(["identity", "boosting", "spatial"])]
        .pivot(index="group", columns="model", values="mae")
        .loc[groups]
    )
    transition_table.columns = transition_table.columns.map(LABELS)
    report = f"""# Neural loss and neighbour experiments

This follow-up diagnoses validation residuals and tests four fixed candidates. The selected neural model by mean validation MAE is **{LABELS[winner]}**, with {means.loc[winner, "mae"]:.4f} mph. No new test predictions were generated or scored. Baseline tables and earlier checkpoints remain unchanged.

## What the validation errors show

At 60 minutes, observed drops of at least 10 mph represent {drop.count_share * 100:.2f}% of observed forecast-target pairs but {drop.sse_share * 100:.2f}% of the identity MLP's squared error. Its signed bias in that group is {drop.bias:+.2f} mph (prediction minus observation). It forecasts speeds that are too high during these drops. The largest 5% of absolute errors account for {base_tails.sse_share * 100:.2f}% of squared error.

Compared with boosting on exactly the same target pairs, the drop group contributes {drop_excess * 100:.2f}% of the identity MLP's net excess squared error. A contribution above 100% is possible because the MLP has less squared error than boosting in the stable-other group, offsetting part of its disadvantage elsewhere. `squared_error_difference_60.csv` records these paired contributions.

Fully observed local histories account for {complete.count_share * 100:.2f}% of target pairs and {complete.sse_share * 100:.2f}% of squared error. Missing-history errors matter per target, but missingness alone does not explain most aggregate squared error. Neighbour availability is separate from this local-history grouping.

Predictions outside the observed training-speed range account for only {outside.outside_fraction * 100:.3f}% of observed 60-minute targets and {outside.outside_sse_share * 100:.3f}% of squared error. Output overshoot outside this range is not the main source of the aggregate RMSE gap; `prediction_ranges.csv` records all horizons and models. The training range is an empirical reference, not a physical speed limit.

These groups describe observed origin-to-target changes, not detected incident onset or a causal explanation. Stable low-speed means an observed origin, a target below 40 mph, and a change strictly within 10 mph. Drops and rises take precedence; a missing raw origin forms a separate group. Future target values define these diagnostic groups and are never model inputs.

60-minute validation MAE by transition:

{markdown_table(transition_table.round(3).reset_index())}

## Fixed experiment protocol

All trials retain twelve input steps, sensor and target-calendar embeddings, local gap age, observed-target masking, the training normalizer, and the benchmark chronological windows. They use seed 42, CPU with four PyTorch threads, AdamW (learning rate 0.001, weight decay 0.0001), batches of 2048, gradient clipping at 1, fresh one-million-row samples each epoch, and a cap of 50 epochs. ReduceLROnPlateau uses factor 0.5, patience three, and threshold 0.0001; early stopping uses patience eight.

Checkpoint selection is minimum mean validation MAE over 15, 30 and 60 minutes for every objective. RMSE and low-speed MAE are secondary measurements, not alternate checkpoint selection rules. The formulas below were fixed before the runs. All losses divide by the training speed standard deviation `std` and exclude zero targets.

- MAE reference: `mean(abs(error)) / std`, using the recorded identity MLP checkpoint.
- Huber candidate: PyTorch smooth L1 with beta 5 mph, divided by `std` (Huber divided by its delta). It changes both the preferred predictor and gradients near zero error; it does not selectively upweight large errors relative to MAE.
- Mixed candidate: `(0.8 * mean(abs(error)) + 0.2 * mean(error**2) / std) / std`. This increases the relative contribution of large errors.
- Spatial candidate and its control: the original MAE objective, with the architecture described below.

{markdown_table(selection[["variant", "best_epoch", "val_mae", "parameters", "matched_baseline_sample_epochs"]].round(4))}

Histories record training-sample hashes. Every candidate matches the original seed-42 sampling plan for all common epochs; all four share {common} epochs. The spatial pair starts with exactly identical parameter values and has {int(s.loc["spatial", "parameters"]):,} parameters each. The identity loss variants retain 44,665 parameters. Differences between the larger control and the original baseline bundle capacity and initialization changes; only the spatial-versus-control contrast isolates access to neighbour messages within the new architecture.

## Neighbour model and matched control

Each road direction aggregates four channels at every input timestamp: normalized filled speed, observed-weight share, log gap age, and reliability mass. Road weights have no self-loops and are row-normalized separately for incoming and outgoing directions. Reliability is `1 / (1 + consecutive missing steps)`; speed and log age are averaged with road weights multiplied by reliability. Observed share uses the original normalized road weights. Reliability mass is supplied explicitly. An empty road row produces zeros. All messages use current or earlier input timestamps, never future target speeds.

The twelve-step incoming/outgoing histories feed a bias-free 96-to-32 linear projection and ReLU. These 32 hidden channels concatenate with the original 64 local history channels and 32 identity/calendar/horizon channels. Two 128-wide residual blocks predict a correction around local persistence. The zero-initialized head reproduces persistence before training.

The control multiplies all neighbour channels by zero before the bias-free spatial encoder. It keeps the same parameters, initialization, optimizer and sample plan. This is a pooled-neighbour history MLP, not Graph WaveNet, DCRNN, or learned graph message passing. Reliability weighting and road direction are bundled; their separate contributions are not isolated.

## Validation results

The table reports the arithmetic mean of the three horizon metrics. Mean RMSE is not a pooled RMSE. Low-speed groups use observed targets below 40 mph. Bias is prediction minus observation. The frontier marks models not dominated on both mean MAE and mean RMSE among these six recorded candidates; it does not measure uncertainty or include low-speed MAE in dominance.

{tables}

{markdown_table(details.round(4).reset_index(names="Model"))}

![Validation MAE, RMSE and low-speed MAE](../outputs/experiments/neural_followup/validation_comparison.png)

The neighbour-aware model {spatial_direction} mean validation MAE by {abs(spatial_gain):.2f}% relative to its matched control. In the 60-minute drop group, its MAE is {spatial_drop:.3f} mph versus {control_drop:.3f} for the control. Model rankings apply to this validation period and budget. Read MAE, RMSE and low-speed MAE together; the report does not promote a model solely because one secondary metric improves.

![Error contributions and validation learning curves](../outputs/experiments/neural_followup/diagnosis_and_learning.png)

## What this batch supports

The Huber candidate changes mean MAE by {means.loc["huber", "mae"] - means.loc["identity", "mae"]:+.4f} mph and mean RMSE by {means.loc["huber", "rmse"] - means.loc["identity", "rmse"]:+.4f} mph relative to the original identity MLP. Mixed loss changes them by {means.loc["mixed", "mae"] - means.loc["identity", "mae"]:+.4f} and {means.loc["mixed", "rmse"] - means.loc["identity", "rmse"]:+.4f} mph. These are fixed-objective comparisons using MAE-selected checkpoints; a different primary selection metric would define a different experiment.

The spatial pair measures the contribution of pooled road-neighbour histories within this architecture. It does not identify whether road direction, reliability weights, neighbour speeds or gap features caused that contribution. Repeating the matched pair with further seeds is the next check before treating any spatial gain as stable. A fuller temporal convolution or adaptive-graph model would then test a separate architectural hypothesis. No new test-period selection is needed to describe this validation evidence.

## Verification and limits

All saved validation arrays are checked for start indices, horizons, shape and finite predictions. Every checkpoint is restored and its selected score reproduced. `extend_neural.py --evaluate-only` also requires exact equality with an existing saved prediction array. Each transition, speed, missing-count and gap-age partition reproduces pooled count, MAE, RMSE, MAPE and squared-error totals. The boosting reference is refitted with its existing chosen settings solely to recover validation predictions; all nine validation MAE/RMSE/MAPE values must match the recorded four-decimal metrics within 0.000051 before saving.

This is a single-seed architecture/loss comparison. Validation has been reused for model development; the measured winner is not an unbiased estimate of future performance. Adjacent benchmark split targets overlap eleven timestamps, as in earlier experiments. Counts are dependent forecast-target pairs, not independent observations. A new independent period or rolling-origin evaluation and repeated matched seeds are required before broader claims. A best epoch at the cap does not establish convergence. No conclusion about traffic propagation or incident causes follows from subgroup errors.

## Reproduction

Generate the original identity validation predictions first if they are absent, using the earlier investigation's checkpoint restoration. Then:

```sh
uv run --extra neural scripts/extend_neural.py
uv run --extra neural scripts/extend_neural.py --evaluate-only
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 uv run --extra neural scripts/diagnose_neural.py --restore-boosting
uv run --extra neural scripts/diagnose_neural.py
uv run --extra neural scripts/summarize_neural_followup.py
```

Artifacts live in `outputs/experiments/neural_followup/`; large prediction arrays remain under ignored `outputs/predictions/`. The contract records data and graph hashes, sensor order, normalizer, objective formulas, selection rule, settings and software versions. No official results file is written by these scripts.
"""
    (ROOT / "docs/neural_followup.md").write_text(report)
    print(tables)
    print(
        f"Selected neural candidate: {winner}; spatial gain against control: {spatial_gain:.2f}%"
    )


if __name__ == "__main__":
    main()
