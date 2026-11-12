"""Report the attention benchmark and purged rolling evaluation separately."""

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
from traffic.attention import AttentionData, AttentionMLP, purged_period
from traffic.data import IN_STEPS, REPORT_HORIZONS, ROOT, STEPS_PER_DAY, load_metr_la
from traffic.identity import gap_age
from traffic.reporting import markdown_table

OUT = ROOT / "outputs/experiments/attention"
LABELS = {
    "mlp": "Neighbour MLP",
    "uniform": "Uniform-attention control",
    "attention": "Neighbour Transformer",
}
COLORS = {"mlp": "#2563eb", "uniform": "#d97706", "attention": "#059669"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-only", action="store_true")
    args = parser.parse_args()
    original = load_metr_la()
    periods = ["benchmark"] if args.benchmark_only else ["benchmark", "early", "late"]
    records, grouped, selections, samples, contracts, weights = [], [], [], [], {}, []
    torch.set_num_threads(4)
    checked, checkpoint_hashes, prediction_hashes = 0, {}, {}
    for period in periods:
        boundary = int(original.splits["val"][0]) + IN_STEPS
        ds = (
            original
            if period == "benchmark"
            else purged_period(
                original, boundary - (14 * STEPS_PER_DAY if period == "early" else 0)
            )
        )
        contract = json.loads((OUT / period / "contract.json").read_text())
        contracts[period] = contract
        assert contract["data_sha256"] == hashlib.sha256(ds.speed.tobytes()).hexdigest()
        assert (
            contract["graph_sha256"]
            == hashlib.sha256(ds.adjacency.tobytes()).hexdigest()
        )
        np.testing.assert_allclose(
            contract["normalizer"], ds.normalizer(), rtol=0, atol=1e-7
        )
        if period != "benchmark":
            assert (
                contract["splits"]["train"]["target_last"]
                < contract["splits"]["val"]["target_first"]
            )
            assert (
                contract["splits"]["val"]["target_last"]
                < contract["splits"]["evaluation"]["target_first"]
            )
            assert (
                contract["splits"]["evaluation"]["target_last"]
                < original.splits["test"][0] + IN_STEPS
            )
        histories = {}
        for variant in LABELS:
            selected = json.loads(
                (OUT / period / variant / "selection.json").read_text()
            )
            digest = hashlib.sha256(
                (ROOT / selected["checkpoint_path"]).read_bytes()
            ).hexdigest()
            assert digest == selected["checkpoint_sha256"]
            checkpoint_hashes[f"{period}_{variant}"] = digest
            selections.append(dict(period=period, **selected))
            histories[variant] = pd.read_csv(ROOT / selected["history_path"])
            for split in ["val"] + ([] if period == "benchmark" else ["evaluation"]):
                starts = ds.splits[split]
                origins = starts + IN_STEPS - 1
                y = ds.speed[origins[:, None] + np.array(REPORT_HORIZONS)[None]]
                path = (
                    ROOT
                    / f"outputs/predictions/attention_{period}_{variant}_{split}.npz"
                )
                p = load_predictions(path, starts, np.array(REPORT_HORIZONS), y.shape)
                prediction_hashes[f"{period}_{variant}_{split}"] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                missing = ds.missing[starts[:, None] + np.arange(IN_STEPS)].sum(1)
                age = gap_age(ds.speed)[origins]
                scores = []
                for i, h in enumerate(REPORT_HORIZONS):
                    aggregate = residual_summary(p[:, i], y[:, i])
                    scores.append(aggregate["mae"])
                    records.append(
                        dict(
                            period=period,
                            variant=variant,
                            split=split,
                            minutes=h * 5,
                            **aggregate,
                        )
                    )
                    for partition, (labels, names) in forecast_partitions(
                        ds.speed[origins], y[:, i], missing, age
                    ).items():
                        rows = []
                        for j, name in enumerate(names):
                            mask = np.broadcast_to(labels, y[:, i].shape) == j
                            result = residual_summary(p[:, i][mask], y[:, i][mask])
                            rows.append(result)
                            grouped.append(
                                dict(
                                    period=period,
                                    variant=variant,
                                    split=split,
                                    minutes=h * 5,
                                    partition=partition,
                                    group=name,
                                    sse_share=result["sse"] / aggregate["sse"],
                                    **result,
                                )
                            )
                        verify_partition(rows, aggregate)
                        np.testing.assert_allclose(
                            sum(r["sse"] for r in rows), aggregate["sse"], rtol=1e-12
                        )
                        checked += 1
                if split == "val":
                    np.testing.assert_allclose(
                        np.mean(scores), selected["val_mae"], rtol=0, atol=1e-6
                    )
        common = min(map(len, histories.values()))
        for frame in histories.values():
            pd.testing.assert_series_equal(
                frame.sample_sha256[:common],
                histories["mlp"].sample_sha256[:common],
                check_names=False,
            )
        samples.append(
            dict(
                period=period,
                matched_epochs=common,
                epochs={v: len(h) for v, h in histories.items()},
            )
        )
        # Fixed input-only diagnostic sample; never used for model selection.
        data = AttentionData(ds)
        selected_rows = np.random.default_rng(0).choice(
            data.size("val"), min(10000, data.size("val")), replace=False
        )
        inputs, _ = data.batch("val", selected_rows)
        _, _, _, _, histories_x, nodes, edges, valid = inputs
        for variant in ["uniform", "attention"]:
            choice = json.loads((OUT / period / variant / "selection.json").read_text())
            model = AttentionMLP(ds.n_sensors, uniform=variant == "uniform").eval()
            model.load_state_dict(
                torch.load(ROOT / choice["checkpoint_path"], weights_only=True)[
                    "state_dict"
                ]
            )
            with torch.no_grad():
                tokens = model.token_projection(
                    torch.cat([histories_x, model.sensor(nodes), edges], dim=2)
                )
                q, key, _ = (
                    model.encoder.qkv(model.encoder.norm1(tokens))
                    .reshape(len(tokens), 4, 3, 2, 8)
                    .permute(2, 0, 3, 1, 4)
                    .unbind(0)
                )
                logits = q[:, :, :1] @ key.transpose(-2, -1) / np.sqrt(8)
                if variant == "uniform":
                    logits = logits * 0
                probability = (
                    logits.masked_fill(~valid[:, None, None, :], -torch.inf)
                    .softmax(-1)
                    .squeeze(2)
                )
                connected = valid.sum(1) > 1
                entropy = -(probability * probability.clamp_min(1e-12).log()).sum(-1)
                normalized = (
                    entropy[connected] / valid.sum(1)[connected, None].float().log()
                )
                self_weight = probability[connected, :, 0]
                weights.append(
                    dict(
                        period=period,
                        variant=variant,
                        sample_rows=len(selected_rows),
                        sample_sha256=hashlib.sha256(
                            selected_rows.tobytes()
                        ).hexdigest(),
                        connected_rows=int(connected.sum()),
                        normalized_entropy=float(normalized.mean()),
                        self_weight_mean=float(self_weight.mean()),
                        self_weight_std=float(self_weight.std()),
                    )
                )
        assert next(
            s["parameters"]
            for s in selections
            if s["period"] == period and s["variant"] == "attention"
        ) == next(
            s["parameters"]
            for s in selections
            if s["period"] == period and s["variant"] == "uniform"
        )
    pd.DataFrame(weights).to_csv(OUT / "attention_diagnostics.csv", index=False)
    frame, partitions = pd.DataFrame(records), pd.DataFrame(grouped)
    frame.to_csv(OUT / "metrics.csv", index=False)
    for period in periods:
        if period == "benchmark":
            continue
        saved = (
            pd.read_csv(OUT / period / "evaluation.csv")
            .set_index(["variant", "minutes"])
            .sort_index()
        )
        computed = (
            frame[(frame.period == period) & (frame.split == "evaluation")]
            .set_index(["variant", "minutes"])
            .sort_index()
        )
        np.testing.assert_array_equal(saved.index.to_numpy(), computed.index.to_numpy())
        np.testing.assert_allclose(
            saved[["mae", "rmse", "mape", "n"]],
            computed[["mae", "rmse", "mape", "n"]],
            rtol=0,
            atol=1e-10,
        )
    partitions.to_csv(OUT / "partitions.csv", index=False)
    pd.DataFrame(selections).to_csv(OUT / "selection.csv", index=False)
    cap_hits = sum(s["best_epoch"] == 50 for s in selections)
    means = frame.groupby(["period", "split", "variant"])[["mae", "rmse"]].mean()
    low = partitions[
        (partitions.partition == "speed") & (partitions.group == "below_40")
    ]
    means["low_speed_mae"] = low.groupby(["period", "split", "variant"]).mae.mean()
    means.to_csv(OUT / "comparison.csv")
    benchmark = means.loc[("benchmark", "val")].loc[list(LABELS)]
    bench_display = benchmark.rename(index=LABELS).rename(
        columns={
            "mae": "Mean MAE",
            "rmse": "Mean RMSE",
            "low_speed_mae": "Mean low-speed MAE",
        }
    )
    plot_periods = (
        ["benchmark"] if args.benchmark_only else ["benchmark", "early", "late"]
    )
    fig, axes = plt.subplots(
        len(plot_periods), 2, figsize=(8, 3.5 * len(plot_periods)), squeeze=False
    )
    for row, period in enumerate(plot_periods):
        split = "val" if period == "benchmark" else "evaluation"
        subset = frame[(frame.period == period) & (frame.split == split)]
        for variant in LABELS:
            model = subset[subset.variant == variant].sort_values("minutes")
            for col, metric in enumerate(["mae", "rmse"]):
                axes[row, col].plot(
                    model.minutes,
                    model[metric],
                    marker="o",
                    color=COLORS[variant],
                    label=LABELS[variant],
                )
        for col, metric in enumerate(["mae", "rmse"]):
            axes[row, col].set(
                title=f"{period.capitalize()} {'tuning' if period == 'benchmark' else 'evaluation'} {metric.upper()}",
                xlabel="Horizon (minutes)",
                ylabel="mph",
                xticks=[15, 30, 60],
            )
            axes[row, col].grid(alpha=0.2)
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=9
    )
    fig.tight_layout(rect=(0, 0.04 if len(plot_periods) > 1 else 0.12, 1, 1))
    fig.savefig(OUT / "comparison.png", dpi=160)
    plt.close(fig)
    manifest = json.loads((OUT / "prior_artifacts.json").read_text())
    for path, digest in manifest.items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    verification = dict(
        scope="benchmark validation and purged tuning/evaluation; original benchmark test not scored",
        partition_checks=checked,
        matched_samples=samples,
        preserved_prior_artifacts=len(manifest),
        checkpoint_sha256=checkpoint_hashes,
        prediction_sha256=prediction_hashes,
        disjoint_purged_target_periods=not args.benchmark_only,
    )
    (OUT / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    gain_uniform = 100 * (
        1 - benchmark.loc["attention", "mae"] / benchmark.loc["uniform", "mae"]
    )
    gain_mlp = 100 * (
        1 - benchmark.loc["attention", "mae"] / benchmark.loc["mlp", "mae"]
    )
    rolling_text = (
        "The two purged rolling periods are the next stage of this fixed experiment."
    )
    if not args.benchmark_only:
        rolling = means.xs("evaluation", level="split").loc[["early", "late"]]
        rolling.to_csv(OUT / "rolling_summary.csv")
        rolling_display = rolling.reset_index()
        rolling_display["variant"] = rolling_display.variant.map(LABELS)
        rolling_display = rolling_display.rename(
            columns={
                "period": "Period",
                "variant": "Model",
                "mae": "Mean MAE",
                "rmse": "Mean RMSE",
                "low_speed_mae": "Mean low-speed MAE",
            }
        )
        fold_rows = []
        for period in ["early", "late"]:
            g = rolling.loc[period]
            fold_rows.append(
                dict(
                    period=period,
                    attention_gain_vs_uniform_pct=100
                    * (1 - g.loc["attention", "mae"] / g.loc["uniform", "mae"]),
                    attention_gain_vs_mlp_pct=100
                    * (1 - g.loc["attention", "mae"] / g.loc["mlp", "mae"]),
                )
            )
        fold_frame = pd.DataFrame(fold_rows)
        fold_frame.to_csv(OUT / "rolling_gains.csv", index=False)
        rolling_text = (
            markdown_table(rolling_display.round(4))
            + "\n\n"
            + markdown_table(fold_frame.round(3))
            + "\n\nPositive gains favour the Transformer. Negative gains mean higher error. These are fixed-protocol comparisons; outer scores do not trigger setting changes or checkpoint reselection."
        )
    periods_rows = []
    for period in periods:
        for split, values in contracts[period]["splits"].items():
            periods_rows.append(
                dict(
                    period=period,
                    split=split,
                    first_target=values["first_timestamp"],
                    last_target=values["last_timestamp"],
                    windows=values["windows"],
                )
            )
    report = f"""# Neighbour attention and purged rolling evaluation

A compact Transformer is compared with a matched uniform-attention intervention and the stronger neighbour MLP. This tests learned weighting of individual sensor histories. It is not a temporal Transformer over twelve time tokens or a reproduction of a published traffic model. Settings are fixed in `outputs/experiments/attention/protocol.json` before fitting.

## Architecture and controlled question

The encoder uses the residual multihead attention and feedforward pattern introduced in [Attention Is All You Need](https://arxiv.org/abs/1706.03762). Restricting attention to road-neighbour tokens draws on the idea of graph-restricted attention in [Graph Attention Networks](https://arxiv.org/abs/1710.10903); this implementation uses dot-product attention rather than the original GAT mechanism. No published result is reproduced or used as a directly comparable score.

Each prediction row has four tokens: the local sensor plus up to three neighbours chosen by the largest weight in either road direction, with stable ordering and zero self-weights. Each token contains twelve normalized speeds and flags, log consecutive missing age at the origin, the shared 16-dimensional sensor embedding, incoming/outgoing road weights and a self indicator. Invalid road-neighbour slots are masked; the local token is always valid. Measurement gaps are represented by forward-filled speeds, flags and gap age. Token histories preserve temporal order by flattening the twelve readings. All sensor inputs end at the forecast origin.

A linear projection produces 16 channels. One pre-normalized Transformer encoder block has two attention heads, a 32-channel feedforward layer, residual connections and dropout 0.1 outside attention. Scaled dot-product attention uses zero dropout in both variants. The local output adds 16 hidden channels to the existing local and pooled-neighbour history representation. Two 144-channel residual prediction blocks retain the sensor/target-calendar/horizon features and local persistence anchor. The zero-initialized head reproduces persistence before training. Both attention variants have 97,001 parameters; the neighbour MLP has 76,569.

The control zeros queries and keys before attention, producing uniform weights over valid tokens. Every other parameter, input, initialized value, optimizer policy and sample plan matches. Query/key gradients are zero in the control. A gain over this control supports learned weighting within this architecture; it does not isolate time-varying attention from identity-based static preferences. Differences against the smaller MLP also bundle richer individual-neighbour inputs and extra capacity. Attention weights are not treated as causal explanations.

## Training and benchmark comparison

All trials use seed 42, masked MAE divided by training standard deviation, AdamW (learning rate 0.001, weight decay 0.0001), gradient clipping at 1, fresh one-million-row samples each epoch, batch size 2048 and a 50-epoch cap. ReduceLROnPlateau uses factor 0.5, patience three and threshold 0.0001; early stopping uses patience eight. Checkpoints minimize mean tuning MAE over 15, 30 and 60 minutes. Uniform and learned attention start identically. Sampling hashes match for all common epochs within each period. Training logic is shared with the earlier neural follow-up.

The benchmark MLP reuses the earlier unchanged seed-42 spatial MAE checkpoint. Its validation predictions must reproduce the earlier saved array exactly. Attention candidates retain the same benchmark data and budget. This benchmark retains the previously disclosed eleven overlapping target timestamps and reused validation period.

{markdown_table(bench_display.round(4).reset_index(names="Model"))}

On benchmark mean validation MAE, the Transformer has a {gain_uniform:+.2f}% gain relative to uniform attention and {gain_mlp:+.2f}% relative to the MLP. Positive means lower error; negative means higher error. RMSE and low-speed MAE are secondary measurements, not alternate selection criteria.

A fixed input-only sample of 10,000 tuning rows per period records normalized local-query attention entropy and self-weight statistics in `attention_diagnostics.csv`. Entropy one means uniform attention over valid tokens; zero means concentration on one token. Statistics exclude sensors with no valid neighbour and average both heads. Different weights alone do not establish forecasting value or causal importance; the matched forecast comparison is the evidence for utility. This diagnostic does not select models or change settings.

## Purged period design

The earlier evaluation starts fourteen days before the benchmark validation target start. The later evaluation starts at that benchmark validation target start. Each has a preceding seven-day tuning period and an expanding training prefix. Every twelve-step target sequence must remain wholly inside its assigned period. Histories may use earlier observed speeds, as in rolling forecasting. Full target ranges do not overlap across training, tuning or evaluation. Initial-gap filling and normalization are refitted to that fold's training prefix. Every model is trained from scratch for each rolling period.

All three checkpoints for a period must be selected using only tuning data before any evaluation score is exposed. Evaluation windows have seven calendar days of targets and remain before the original benchmark test targets. The architecture, loss, seed and training settings are held fixed regardless of the benchmark result. The two periods evaluate temporal sensitivity, not the isolated effect of purging: the training spans and checkpoint-selection periods also differ from the original benchmark.

{markdown_table(pd.DataFrame(periods_rows))}

{rolling_text}

![Attention comparison by horizon](../outputs/experiments/attention/comparison.png)

## Verification and limits

Input tests cover masking, missing neighbours, permutation of neighbour tokens, uniform query/key gradients, initialization, prediction alignment, checkpoint restoration, future-speed perturbations, disjoint target ranges and training-only preprocessing. Each trial restores its selected checkpoint before saving validation predictions. Repeated restoration requires exact equality with an existing prediction array. Evaluation restores and verifies all tuning predictions before scoring a period. Prediction metadata, checkpoint hashes and all {checked} subgroup partitions are checked in this report. All {len(manifest)} protected prior artifacts retain their hashes.

This is one seed per model and period. The dataset periods have already been used in earlier project work, so this is retrospective temporal sensitivity with correct fold-local fitting, not pristine unseen data. The later fold may train on the earlier fold's evaluation week, as permitted in expanding-window evaluation; folds are not independent replications. The protocol forbids outer-score tuning, but global architecture choices still reflect prior dataset exploration. Small width, one encoder block, three neighbours and the fixed budget limit claims about Transformers in general. {cap_hits} of {len(selections)} selected checkpoints occur at epoch 50; convergence is not established for those runs. Further seeds, longer history, a temporal Transformer, other neighbourhood sizes and a new dataset are separate experiments. No original benchmark test scores are added.

## Further experiments worth separating

- Test temporal attention with a longer history while giving its MLP control the same history. The current model attends across four sensor histories, so it does not test the value of attention across time tokens.
- A fuller traffic-specific reference is [STAEformer](https://arxiv.org/abs/2308.10425). Its [official implementation](https://github.com/XDZhelheim/STAEformer/blob/main/model/STAEformer.py) combines adaptive embeddings with separate temporal and spatial attention over joint network windows. That would require a different full-network batching design and a separate reproduction; this pilot is not evidence about its attainable performance.
- Supply the training weekly-profile features to both neural variants using the existing whole-week cross-fitting rule. Boosting already has this information. This tests feature parity before attributing the remaining gap solely to architecture.

## Reproduction

```sh
uv run --extra neural scripts/try_attention.py --prepare
uv run --extra neural scripts/try_attention.py --period benchmark --variant mlp
uv run --extra neural scripts/try_attention.py --period benchmark --variant uniform
uv run --extra neural scripts/try_attention.py --period benchmark --variant attention
uv run --extra neural scripts/summarize_attention.py --benchmark-only
# For each of early and late, fit each of mlp, uniform and attention first:
uv run --extra neural scripts/try_attention.py --period early --variant mlp
# After all three fits for that period:
uv run --extra neural scripts/try_attention.py --period early --evaluate-period
# Repeat for late, then:
uv run --extra neural scripts/summarize_attention.py
```

Trial histories, selected checkpoints and immutable contracts live in `outputs/experiments/attention/<period>/<variant>/`. Saved prediction arrays stay under ignored `outputs/predictions/`. Repeat a fitting command with `--evaluate-only` to check exact restoration without fitting again.
"""
    (ROOT / "docs/attention.md").write_text(report)
    html = (
        '''<!doctype html><html><head><meta charset="utf-8"><style>body{margin:0;font-family:var(--font-sans,sans-serif);color:var(--foreground)}p{font-size:14px;line-height:1.5}img{display:block;width:100%;height:auto}</style></head><body><p>Benchmark tuning and purged evaluation are shown separately. Checkpoints minimize tuning MAE. Uniform attention retains the same inputs and parameters as the learned-attention model.</p><img src="'''
        + str(OUT / "comparison.png")
        + """" alt="Attention and rolling-period comparison"><p>One seed per trial. Reused dataset periods; the original benchmark test is not scored.</p></body></html>"""
    )
    (OUT / "comparison.html").write_text(html)
    print(markdown_table(bench_display.round(4).reset_index(names="Model")))
    if not args.benchmark_only:
        print(rolling_text)
    print(f"Verified {checked} partitions and {len(manifest)} preserved artifacts.")


if __name__ == "__main__":
    main()
