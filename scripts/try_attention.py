"""Attention first, then purged rolling periods with separate checkpoint tuning.

uv run --extra neural scripts/try_attention.py --prepare
uv run --extra neural scripts/try_attention.py --period benchmark --variant attention
uv run --extra neural scripts/try_attention.py --period benchmark --variant uniform
uv run --extra neural scripts/try_attention.py --period early --variant mlp
uv run --extra neural scripts/try_attention.py --period early --evaluate-period
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform

import numpy as np
import pandas as pd
import torch

from traffic.analysis import load_predictions
from traffic.attention import AttentionData, AttentionMLP, purged_period
from traffic.data import IN_STEPS, REPORT_HORIZONS, ROOT, STEPS_PER_DAY, load_metr_la
from traffic.identity import predict_identity
from traffic.metrics import masked_errors
from traffic.spatial_identity import NeighbourIdentityMLP
from traffic.training import fit_forecaster

OUT = ROOT / "outputs/experiments/attention"
VARIANTS = ("mlp", "uniform", "attention")
PERIODS = ("benchmark", "early", "late")


def dataset_for(ds, period):
    if period == "benchmark":
        return ds
    last = int(ds.splits["val"][0]) + IN_STEPS
    return purged_period(ds, last - (14 * STEPS_PER_DAY if period == "early" else 0))


def model_for(variant, n_sensors):
    return (
        NeighbourIdentityMLP(n_sensors)
        if variant == "mlp"
        else AttentionMLP(n_sensors, uniform=variant == "uniform")
    )


def truth_for(ds, split):
    return ds.speed[
        ds.splits[split][:, None] + IN_STEPS + np.array(REPORT_HORIZONS)[None] - 1
    ]


def write_contract(ds, period):
    splits = {k: v for k, v in ds.splits.items() if k != "test"}
    contract = dict(
        period=period,
        data_sha256=hashlib.sha256(ds.speed.tobytes()).hexdigest(),
        graph_sha256=hashlib.sha256(ds.adjacency.tobytes()).hexdigest(),
        sensor_ids=ds.sensor_ids,
        normalizer=list(ds.normalizer()),
        horizons=list(REPORT_HORIZONS),
        seed=42,
        epochs=50,
        rows_per_epoch=1_000_000,
        batch_size=2048,
        threads=4,
        objective="masked MAE / training std",
        selection="minimum mean tuning MAE over horizons",
        optimizer="AdamW .001 wd=.0001; clip=1; plateau factor=.5 patience=3 threshold=.0001; early stop=8",
        attention="4 sensor-history tokens, 16 channels, 2 heads, 1 pre-norm encoder, FFN 32; no attention dropout; residual dropout .1",
        control="zero queries and keys; uniform weights over valid tokens; same inputs, parameters and initialization",
        splits={
            k: dict(
                windows=len(v),
                starts_sha256=hashlib.sha256(v.tobytes()).hexdigest(),
                target_first=int(v[0] + IN_STEPS),
                target_last=int(v[-1] + IN_STEPS + 11),
                first_timestamp=str(ds.timestamps[v[0] + IN_STEPS]),
                last_timestamp=str(ds.timestamps[v[-1] + IN_STEPS + 11]),
            )
            for k, v in splits.items()
        },
        python=platform.python_version(),
        torch=str(torch.__version__),
    )
    folder = OUT / period
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "contract.json"
    if path.exists():
        if json.loads(path.read_text()) != contract:
            raise ValueError(f"Contract changed: {period}")
    else:
        path.write_text(json.dumps(contract, indent=2) + "\n")
    return contract


def prediction_path(period, variant, split):
    return ROOT / f"outputs/predictions/attention_{period}_{variant}_{split}.npz"


def restore(ds, data, period, variant):
    directory = OUT / period / variant
    selected = json.loads((directory / "selection.json").read_text())
    path = ROOT / selected["checkpoint_path"]
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest() == selected["checkpoint_sha256"]
    )
    checkpoint = torch.load(path, weights_only=True)
    assert checkpoint["n_sensors"] == ds.n_sensors
    assert checkpoint["seed"] == 42 and checkpoint["objective"] == "mae"
    assert checkpoint["variant"] == (
        "spatial" if period == "benchmark" and variant == "mlp" else variant
    )
    np.testing.assert_allclose(
        [checkpoint["mean"], checkpoint["std"]], ds.normalizer(), rtol=0, atol=1e-7
    )
    model = model_for(variant, ds.n_sensors)
    model.load_state_dict(checkpoint["state_dict"])
    pred = predict_identity(model, data, "val")
    score = float(
        np.mean(
            [
                masked_errors(pred[:, i], truth_for(ds, "val")[:, i])["mae"]
                for i in range(3)
            ]
        )
    )
    np.testing.assert_allclose(score, selected["val_mae"], rtol=0, atol=1e-6)
    existing = prediction_path(period, variant, "val")
    if existing.exists():
        np.testing.assert_array_equal(
            pred,
            load_predictions(
                existing, ds.splits["val"], np.array(REPORT_HORIZONS), pred.shape
            ),
        )
        print(
            f"{period} {variant}: validation predictions restore exactly.", flush=True
        )
    return model, pred


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--period", choices=PERIODS, default="benchmark")
    parser.add_argument("--variant", choices=VARIANTS, default="attention")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--evaluate-period", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    original = load_metr_la()
    if args.prepare:
        OUT.mkdir(parents=True, exist_ok=True)
        protocol = dict(
            sequence="benchmark attention comparison first, then two purged rolling periods",
            variants=list(VARIANTS),
            seed=42,
            budget="50 epochs, fresh 1m rows/epoch, batch 2048; same optimizer and stopping policy",
            fixed_architecture="16-channel, 2-head, one-block Transformer over 4 sensor histories; uniform-attention control",
            periods="early: benchmark tuning target start minus 14 days; late: benchmark tuning target start; each has 7-day tuning then 7-day evaluation",
            outer_policy="select all checkpoints on inner tuning only; evaluate each outer period after all three fits; no outer-score tuning or seed selection",
            benchmark="MLP reuses the unchanged seed-42 MAE checkpoint and verifies exact validation predictions",
            limits="retrospective sensitivity on already-used dataset periods; no claim of pristine unseen data; original benchmark test never scored",
        )
        path = OUT / "protocol.json"
        if path.exists():
            assert json.loads(path.read_text()) == protocol
        else:
            path.write_text(json.dumps(protocol, indent=2) + "\n")
        manifest = OUT / "prior_artifacts.json"
        if not manifest.exists():
            files = [
                p
                for p in (ROOT / "outputs").rglob("*")
                if p.is_file() and "predictions" not in p.parts and OUT not in p.parents
            ]
            manifest.write_text(
                json.dumps(
                    {
                        str(p.relative_to(ROOT)): hashlib.sha256(
                            p.read_bytes()
                        ).hexdigest()
                        for p in sorted(files)
                    },
                    indent=2,
                )
                + "\n"
            )
        for period in PERIODS:
            contract = write_contract(dataset_for(original, period), period)
            print(period, contract["splits"], flush=True)
        return
    if not (OUT / "protocol.json").exists():
        raise FileNotFoundError("Run --prepare before fitting")
    ds = dataset_for(original, args.period)
    write_contract(ds, args.period)
    data = AttentionData(ds)
    directory = OUT / args.period / args.variant
    directory.mkdir(exist_ok=True)
    if args.evaluate_period:
        if args.period == "benchmark":
            parser.error("The benchmark has no outer evaluation in this experiment")
        # Require every inner-selected checkpoint before exposing any outer score.
        for variant in VARIANTS:
            assert (OUT / args.period / variant / "selection.json").is_file(), variant
        rows = []
        for variant in VARIANTS:
            model, _ = restore(ds, data, args.period, variant)
            pred = predict_identity(model, data, "evaluation")
            path = prediction_path(args.period, variant, "evaluation")
            if path.exists():
                np.testing.assert_array_equal(
                    pred,
                    load_predictions(
                        path,
                        ds.splits["evaluation"],
                        np.array(REPORT_HORIZONS),
                        pred.shape,
                    ),
                )
            np.savez_compressed(
                path,
                pred=pred,
                starts=ds.splits["evaluation"],
                horizons=REPORT_HORIZONS,
            )
            for i, h in enumerate(REPORT_HORIZONS):
                rows.append(
                    dict(
                        variant=variant,
                        minutes=h * 5,
                        **masked_errors(pred[:, i], truth_for(ds, "evaluation")[:, i]),
                    )
                )
        pd.DataFrame(rows).to_csv(OUT / args.period / "evaluation.csv", index=False)
        print(pd.DataFrame(rows).to_string(index=False), flush=True)
        return
    old_reference = args.period == "benchmark" and args.variant == "mlp"
    if not args.evaluate_only:
        if old_reference:
            checkpoint_path = (
                ROOT / "outputs/experiments/neural_followup/seed_42/spatial/model.pt"
            )
            history_path = checkpoint_path.parent / "history.csv"
        else:
            torch.manual_seed(42)
            fit_forecaster(
                model_for(args.variant, ds.n_sensors),
                data,
                truth_for(ds, "val"),
                seed=42,
                epochs=50,
                variant=args.variant,
                directory=directory,
            )
            checkpoint_path = directory / "model.pt"
            history_path = directory / "history.csv"
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        selection = dict(
            variant=args.variant,
            best_epoch=checkpoint["best_epoch"],
            val_mae=checkpoint["val_mae"],
            parameters=sum(
                p.numel() for p in model_for(args.variant, ds.n_sensors).parameters()
            ),
            checkpoint_path=str(checkpoint_path.relative_to(ROOT)),
            checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            history_path=str(history_path.relative_to(ROOT)),
        )
        (directory / "selection.json").write_text(
            json.dumps(selection, indent=2) + "\n"
        )
    _, pred = restore(ds, data, args.period, args.variant)
    if old_reference:
        prior = ROOT / "outputs/predictions/followup_spatial_seed42_val.npz"
        np.testing.assert_array_equal(
            pred,
            load_predictions(
                prior, ds.splits["val"], np.array(REPORT_HORIZONS), pred.shape
            ),
        )
    np.savez_compressed(
        prediction_path(args.period, args.variant, "val"),
        pred=pred,
        starts=ds.splits["val"],
        horizons=REPORT_HORIZONS,
    )
    print((directory / "selection.json").read_text(), flush=True)


if __name__ == "__main__":
    main()
