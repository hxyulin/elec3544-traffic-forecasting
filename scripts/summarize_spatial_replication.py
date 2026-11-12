"""Matched-seed spatial replication and the conditional spatial mixed-loss study.

--gate-only checks the predeclared spatial criterion before training mixed loss.
The final report uses validation predictions only and never selects a best seed.
"""

from __future__ import annotations

import argparse
import hashlib
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from traffic.analysis import (
    forecast_partitions,
    load_predictions,
    residual_summary,
    verify_partition,
)
from traffic.data import IN_STEPS, REPORT_HORIZONS, ROOT, load_metr_la
from traffic.identity import gap_age, predict_identity
from traffic.reporting import markdown_table
from traffic.spatial_identity import NeighbourIdentityData, NeighbourIdentityMLP

OUT = ROOT / "outputs/experiments/spatial_replication"
FOLLOWUP = ROOT / "outputs/experiments/neural_followup"
SEEDS = (42, 43, 44)
LABELS = {
    "spatial_control": "Zero-message control",
    "spatial": "Neighbour MLP (MAE)",
    "spatial_mixed": "Neighbour MLP (mixed loss)",
}


def check_gate():
    protocol = json.loads((OUT / "protocol.json").read_text())
    assert protocol["replication_seeds"] == [43, 44]
    rows = []
    for seed in SEEDS:
        selections = pd.read_csv(FOLLOWUP / f"seed_{seed}/selection.csv").set_index(
            "variant"
        )
        control, spatial = [
            float(selections.loc[v, "val_mae"]) for v in ("spatial_control", "spatial")
        ]
        rows.append(
            dict(
                seed=seed,
                control_mae=control,
                spatial_mae=spatial,
                spatial_gain_pct=100 * (1 - spatial / control),
                improved=spatial < control,
            )
        )
    passed = all(
        r["improved"] for r in rows if r["seed"] in protocol["replication_seeds"]
    )
    result = dict(
        passed=passed,
        criterion=protocol["spatial_gate"],
        rows=rows,
        scope="mean validation MAE; seed 42 is discovery, seeds 43 and 44 are replication",
    )
    (OUT / "replication_gate.json").write_text(json.dumps(result, indent=2) + "\n")
    pd.DataFrame(rows).to_csv(OUT / "paired_spatial_gain.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print(f"Predeclared spatial replication gate passed: {passed}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-only", action="store_true")
    args = parser.parse_args()
    gate = check_gate()
    if args.gate_only:
        return
    variants = ["spatial_control", "spatial"] + (
        ["spatial_mixed"] if gate["passed"] else []
    )
    torch.set_num_threads(4)
    ds = load_metr_la()
    restore_data = NeighbourIdentityData(ds, "cpu")
    starts, hs = ds.splits["val"], np.array(REPORT_HORIZONS)
    origins = starts + IN_STEPS - 1
    truth = ds.speed[origins[:, None] + hs[None]]
    missing_count = ds.missing[starts[:, None] + np.arange(IN_STEPS)].sum(1)
    age = gap_age(ds.speed)[origins]
    data_hash = hashlib.sha256(ds.speed.tobytes()).hexdigest()
    graph_hash = hashlib.sha256(ds.adjacency.tobytes()).hexdigest()
    rows, parts, selections, verified, sampling = [], [], [], 0, []
    prediction_hashes, checkpoint_hashes = {}, {}
    reference_contract = json.loads((FOLLOWUP / "seed_42/contract.json").read_text())
    for seed in SEEDS:
        histories = {}
        for variant in variants:
            root = OUT if variant == "spatial_mixed" else FOLLOWUP
            directory = root / f"seed_{seed}"
            contract = json.loads((directory / "contract.json").read_text())
            assert (
                contract["data_sha256"] == data_hash
                and contract["adjacency_sha256"] == graph_hash
            )
            assert contract["sensor_ids"] == ds.sensor_ids and contract[
                "horizons"
            ] == list(REPORT_HORIZONS)
            assert (
                contract["seed"] == seed
                and contract["epochs"] == 50
                and contract["threads"] == 4
            )
            for setting in ("optimizer", "sampling", "selection", "python", "torch"):
                assert contract[setting] == reference_contract[setting], setting
            np.testing.assert_allclose(
                contract["normalizer"], ds.normalizer(), rtol=0, atol=1e-7
            )
            selected = (
                pd.read_csv(directory / "selection.csv")
                .set_index("variant")
                .loc[variant]
            )
            model_path = directory / variant / "model.pt"
            digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
            assert digest == selected.checkpoint_sha256
            checkpoint_hashes[f"{variant}_seed{seed}"] = digest
            path = ROOT / f"outputs/predictions/followup_{variant}_seed{seed}_val.npz"
            pred = load_predictions(path, starts, hs, truth.shape)
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
            assert checkpoint["variant"] == variant and checkpoint["seed"] == seed
            assert checkpoint["objective"] == (
                "mixed" if variant == "spatial_mixed" else "mae"
            )
            np.testing.assert_allclose(
                [checkpoint["mean"], checkpoint["std"]],
                ds.normalizer(),
                rtol=0,
                atol=1e-7,
            )
            restored = NeighbourIdentityMLP(
                ds.n_sensors, use_neighbours=variant != "spatial_control"
            )
            restored.load_state_dict(checkpoint["state_dict"])
            np.testing.assert_array_equal(
                predict_identity(restored, restore_data, "val"), pred
            )
            print(
                f"{variant} seed {seed}: all {pred.size:,} validation predictions restore exactly.",
                flush=True,
            )
            prediction_hashes[f"{variant}_seed{seed}"] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            measured = []
            for i, h in enumerate(hs):
                p, y = pred[:, i], truth[:, i]
                aggregate = residual_summary(p, y)
                measured.append(aggregate["mae"])
                rows.append(
                    dict(seed=seed, variant=variant, minutes=int(h * 5), **aggregate)
                )
                for partition, (labels, names) in forecast_partitions(
                    ds.speed[origins], y, missing_count, age
                ).items():
                    group_rows = []
                    for j, name in enumerate(names):
                        mask = np.broadcast_to(labels, y.shape) == j
                        result = residual_summary(p[mask], y[mask])
                        group_rows.append(result)
                        parts.append(
                            dict(
                                seed=seed,
                                variant=variant,
                                minutes=int(h * 5),
                                partition=partition,
                                group=name,
                                sse_share=result["sse"] / aggregate["sse"],
                                **result,
                            )
                        )
                    verify_partition(group_rows, aggregate)
                    np.testing.assert_allclose(
                        sum(r["sse"] for r in group_rows), aggregate["sse"], rtol=1e-12
                    )
                    verified += 1
            np.testing.assert_allclose(
                np.mean(measured), selected.val_mae, rtol=0, atol=1e-6
            )
            selections.append(
                dict(
                    seed=seed,
                    variant=variant,
                    best_epoch=int(selected.best_epoch),
                    parameters=int(selected.parameters),
                    val_mae=float(selected.val_mae),
                )
            )
            histories[variant] = pd.read_csv(directory / variant / "history.csv")
        common = min(map(len, histories.values()))
        reference = histories["spatial_control"].sample_sha256[:common]
        for history in histories.values():
            pd.testing.assert_series_equal(
                history.sample_sha256[:common], reference, check_names=False
            )
        sampling.append(
            dict(
                seed=seed,
                all_variants_matched_epochs=common,
                epochs={v: len(h) for v, h in histories.items()},
            )
        )
    aggregate, partitioned = pd.DataFrame(rows), pd.DataFrame(parts)
    aggregate.to_csv(OUT / "validation.csv", index=False)
    partitioned.to_csv(OUT / "partitions.csv", index=False)
    selection = pd.DataFrame(selections)
    assert selection.parameters.nunique() == 1
    cap_hits = int((selection.best_epoch == 50).sum())
    selection.to_csv(OUT / "selection.csv", index=False)
    per_seed = aggregate.groupby(["seed", "variant"])[["mae", "rmse"]].mean()
    low = partitioned[
        (partitioned.partition == "speed") & (partitioned.group == "below_40")
    ]
    per_seed["low_speed_mae"] = low.groupby(["seed", "variant"]).mae.mean()
    drop = partitioned[
        (partitioned.partition == "transition")
        & (partitioned.group == "drop_ge10")
        & (partitioned.minutes == 60)
    ]
    per_seed["drop_mae_60"] = drop.set_index(["seed", "variant"]).mae
    per_seed.to_csv(OUT / "per_seed.csv")
    summary = per_seed.groupby("variant").agg(["mean", "std"]).loc[variants]
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.to_csv(OUT / "summary.csv")
    if "spatial_mixed" in variants:
        paired = per_seed.unstack("variant")
        changes = pd.DataFrame(index=SEEDS)
        changes.index.name = "seed"
        for metric in ("mae", "rmse", "low_speed_mae", "drop_mae_60"):
            changes[metric + "_mixed_minus_mae"] = (
                paired[metric].spatial_mixed - paired[metric].spatial
            )
        changes.to_csv(OUT / "mixed_loss_changes.csv")
    else:
        changes = None
    boost_path = ROOT / "outputs/predictions/followup_boosting_val.npz"
    boost = load_predictions(boost_path, starts, hs, truth.shape)
    recorded = pd.read_csv(ROOT / "outputs/results/metrics.csv")
    recorded = recorded[
        (recorded.model == "gradient_boosting_neighbours") & (recorded.split == "val")
    ].set_index("minutes")
    boost_rows, boost_low = [], []
    for i, horizon in enumerate(hs):
        result = residual_summary(boost[:, i], truth[:, i])
        for metric in ("mae", "rmse", "mape"):
            np.testing.assert_allclose(
                result[metric],
                recorded.loc[int(horizon * 5), metric],
                rtol=0,
                atol=0.000051,
            )
        boost_rows.append(result)
        mask = (truth[:, i] > 0) & (truth[:, i] < 40)
        boost_low.append(residual_summary(boost[:, i][mask], truth[:, i][mask])["mae"])
    drop_mask = (
        (ds.speed[origins] > 0)
        & (truth[:, 2] > 0)
        & (truth[:, 2] - ds.speed[origins] <= -10)
    )
    boost_reference = dict(
        mae=float(np.mean([r["mae"] for r in boost_rows])),
        rmse=float(np.mean([r["rmse"] for r in boost_rows])),
        low_speed_mae=float(np.mean(boost_low)),
        drop_mae_60=residual_summary(boost[:, 2][drop_mask], truth[:, 2][drop_mask])[
            "mae"
        ],
    )
    pd.DataFrame([boost_reference]).to_csv(OUT / "boosting_reference.csv", index=False)
    prediction_hashes["boosting_fixed_reference"] = hashlib.sha256(
        boost_path.read_bytes()
    ).hexdigest()
    # Display individual runs; connected points denote matched seeds, not time.
    fig, axes = plt.subplots(3, 1, figsize=(8, 9))
    colors = ["#2563eb", "#d97706", "#059669"]
    for j, seed in enumerate(SEEDS):
        frame = per_seed.loc[seed].loc[variants]
        for ax, metric, title in zip(
            axes,
            ["mae", "rmse", "low_speed_mae"],
            ["Mean validation MAE", "Mean validation RMSE", "Mean low-speed MAE"],
        ):
            ax.plot(
                np.arange(len(variants)),
                frame[metric],
                marker="o",
                color=colors[j],
                label=f"Seed {seed}" + (" (discovery)" if seed == 42 else ""),
                alpha=0.9,
            )
            ax.set(
                xticks=np.arange(len(variants)),
                xticklabels=[LABELS[v] for v in variants],
                ylabel="mph",
                title=title,
            )
            ax.grid(alpha=0.2)
    for ax, metric in zip(axes, ["mae", "rmse", "low_speed_mae"]):
        ax.axhline(
            boost_reference[metric],
            color="#6b7280",
            linestyle="--",
            label="Boosting (fixed reference)",
        )
    fig.legend(
        *axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=10
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "validation_replication.png", dpi=160)
    plt.close(fig)
    previous = json.loads((OUT / "prior_artifacts.json").read_text())
    for path, digest in previous.items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    verification = dict(
        scope="validation only; no new test evaluation",
        partition_checks=verified,
        matched_sampling=sampling,
        preserved_prior_artifacts=len(previous),
        prediction_sha256=prediction_hashes,
        checkpoint_sha256=checkpoint_hashes,
        exact_checkpoint_restorations=len(checkpoint_hashes),
        gate_passed=gate["passed"],
        metric_aggregation="mean over horizons, then mean and sample SD across seeds; RMSE is not pooled",
    )
    (OUT / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    display = summary[
        [
            "mae_mean",
            "mae_std",
            "rmse_mean",
            "rmse_std",
            "low_speed_mae_mean",
            "low_speed_mae_std",
        ]
    ].copy()
    display.index = display.index.map(LABELS)
    display.columns = [
        "Mean MAE",
        "MAE seed SD",
        "Mean RMSE",
        "RMSE seed SD",
        "Mean low-speed MAE",
        "Low-speed seed SD",
    ]
    per_seed_display = per_seed.reset_index()
    per_seed_display["variant"] = per_seed_display.variant.map(LABELS)
    per_seed_display = per_seed_display.rename(
        columns={
            "seed": "Seed",
            "variant": "Model",
            "mae": "Mean MAE",
            "rmse": "Mean RMSE",
            "low_speed_mae": "Mean low-speed MAE",
            "drop_mae_60": "Drop MAE 60 min",
        }
    )
    neural_winner = summary.mae_mean.idxmin()
    mixed_text = (
        "The mixed-loss stage was not run because the predeclared replication criterion failed."
        if changes is None
        else f"Mixed loss changes mean MAE by {changes.mae_mixed_minus_mae.mean():+.4f} mph, mean RMSE by {changes.rmse_mixed_minus_mae.mean():+.4f} mph, and mean low-speed MAE by {changes.low_speed_mae_mixed_minus_mae.mean():+.4f} mph relative to the spatial MAE model, averaged over the three paired seeds. Negative changes indicate lower error. Checkpoints still minimize MAE; there is no RMSE-based checkpoint reselection."
    )
    gate_table = pd.DataFrame(gate["rows"])
    report = f"""# Spatial replication and mixed-loss experiment

The spatial/control comparison was repeated with seeds 43 and 44, retaining seed 42 as discovery evidence. The predeclared replication criterion **{"passed" if gate["passed"] else "failed"}**: neighbour-aware MAE must be below its matched control separately for both new seeds. The selected architecture/objective by mean validation MAE across all three seeds is **{LABELS[neural_winner]}**. No best seed is selected and no new test predictions are generated or scored.

## Protocol fixed before replication results

`outputs/experiments/spatial_replication/protocol.json` records the gate, seeds, conditional mixed-loss stage and selection rule. All runs retain the original data windows, twelve input steps, training normalizer, graph construction and reliability-weighted messages. Each variant has 76,569 parameters. Optimizer, scheduler, clipping, fresh one-million-row sampling, batch size 2048, 50-epoch cap and early-stopping patience eight match the original follow-up. Runs use CPU and four PyTorch threads; concurrent runs have separate seeds and artifact directories.

Within each seed, the control and spatial model initialize identically; the control zeros messages before its bias-free encoder. Spatial MAE and spatial mixed loss use the same model initialization. Sample hashes match for all common epochs within each seed, including the mixed-loss trial when present. Sample plans differ across seeds. The control isolates access to neighbour messages within this architecture; seed replication does not isolate road direction, reliability weighting or individual message channels.

Every checkpoint minimizes mean validation MAE over 15, 30 and 60 minutes. The mixed objective retains the earlier fixed formula `(0.8 * MAE + 0.2 * MSE / training_std) / training_std`. No coefficient, seed or checkpoint is chosen using test scores or secondary metrics. The mixed-loss stage runs only after both new spatial pairs pass the gate.

## Matched spatial results

{markdown_table(gate_table.round(4))}

Seed 42 was used to develop this hypothesis. Seeds 43 and 44 are repetitions on the same validation period, with new initialization and training-row draws. These are not three independent traffic periods or a significance test.

## All validation results

Each run's metric is the arithmetic mean of its three horizon metrics. The summary reports the mean and sample standard deviation across three runs. Standard deviations describe seed sensitivity on this fixed validation set, not confidence intervals for future traffic. Mean RMSE is not pooled RMSE.

{markdown_table(display.round(4).reset_index(names="Model"))}

{markdown_table(per_seed_display.round(4))}

The fixed neighbour-boosting reference has mean validation MAE {boost_reference["mae"]:.4f} mph, mean RMSE {boost_reference["rmse"]:.4f} mph, mean low-speed MAE {boost_reference["low_speed_mae"]:.4f} mph and 60-minute drop MAE {boost_reference["drop_mae_60"]:.4f} mph. These values come from the earlier single fit, not three new boosting runs. Its saved predictions reproduce all nine recorded validation MAE/RMSE/MAPE scores within the original rounding tolerance.

{mixed_text}

![Validation metrics for each matched seed](../outputs/experiments/spatial_replication/validation_replication.png)

For the original seed-42 diagnosis, large speed drops dominate the identity MLP's excess squared error over boosting. This report retains the same retrospective drop definition and records 60-minute drop MAE for every seed in `per_seed.csv`; all transition and missing-input groups appear in `partitions.csv`. The group uses observed future targets and cannot serve as a forecast-time input or an incident detector.

## Verification and limits

Saved predictions must match validation starts, horizons, shape and finite-value checks. Every checkpoint is restored and every saved validation prediction must match exactly. Their scores reproduce selected checkpoint MAE within 0.000001 mph. Checkpoint hashes match selection records. Every subgroup partition reproduces count, MAE, RMSE, MAPE and squared-error totals; {verified} partitions are verified. All {len(previous)} protected earlier result/checkpoint artifacts retain their recorded hashes. Evaluate-only restoration checks exact prediction equality before rewriting an existing prediction file.

Three seeds provide a limited check of initialization and sampled-training sensitivity. Validation is reused for model development, and the benchmark split retains eleven overlapping target timestamps at its boundaries. No claim of an unseen holdout, independent forecast-target pairs, or causal traffic propagation follows from these results. Repeated periods with purged boundaries or new data remain necessary for stronger generalization claims. {cap_hits} of the {len(selection)} selected checkpoints occur at epoch 50; convergence is not established for those runs.

## Reproduction

```sh
uv run --extra neural scripts/extend_neural.py --seed 43 --variants spatial_control spatial
uv run --extra neural scripts/extend_neural.py --seed 44 --variants spatial_control spatial
uv run --extra neural scripts/summarize_spatial_replication.py --gate-only
# Only if the gate passes, run each of seeds 42, 43 and 44 with:
uv run --extra neural scripts/extend_neural.py --seed 42 --variants spatial_mixed --output-dir outputs/experiments/spatial_replication
# Verify each trial by repeating its command with --evaluate-only.
uv run --extra neural scripts/summarize_spatial_replication.py
```

Spatial/control checkpoints remain under `outputs/experiments/neural_followup/seed_<seed>/`. New spatial mixed-loss checkpoints and this report's tables, protocol, preservation manifest and verification live under `outputs/experiments/spatial_replication/`. Prediction arrays remain under ignored `outputs/predictions/`. A temporal convolution or adaptive-graph architecture would be a separate next experiment.
"""
    (ROOT / "docs/spatial_replication.md").write_text(report)
    html = (
        '''<!doctype html><html><head><meta charset="utf-8"><style>
body { margin:0; font-family:var(--font-sans,sans-serif); color:var(--foreground); }
p { font-size:14px; line-height:1.5; } img { width:100%; height:auto; display:block; }
</style></head><body><p>Matched seeds on the same validation period. Every checkpoint was selected on mean MAE; each line connects the same seed across model variants.</p><img src="'''
        + str(OUT / "validation_replication.png")
        + """" alt="Validation errors across matched seeds"><p>Seed variation does not measure generalization to another traffic period.</p></body></html>"""
    )
    (OUT / "comparison.html").write_text(html)
    print(markdown_table(display.round(4).reset_index(names="Model")))
    print(mixed_text)
    print(
        f"Verified {verified} partitions and preserved {len(previous)} prior artifacts."
    )


if __name__ == "__main__":
    main()
