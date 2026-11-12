"""Validation-only loss and road-message experiments; never score the test split.

uv run --extra neural scripts/extend_neural.py
uv run --extra neural scripts/extend_neural.py --evaluate-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from traffic.analysis import load_predictions
from traffic.data import IN_STEPS, REPORT_HORIZONS, ROOT, load_metr_la
from traffic.identity import IdentityData, IdentityMLP, predict_identity
from traffic.metrics import masked_errors
from traffic.spatial_identity import (
    NeighbourIdentityData,
    NeighbourIdentityMLP,
)
from traffic.training import fit_forecaster

OBJECTIVES = {
    "huber": "huber",
    "mixed": "mixed",
    "spatial_control": "mae",
    "spatial": "mae",
    "spatial_mixed": "mixed",
}
VARIANTS = tuple(OBJECTIVES)
OUT = ROOT / "outputs/experiments/neural_followup"


def make_model(variant, n_sensors):
    if variant.startswith("spatial"):
        return NeighbourIdentityMLP(
            n_sensors, use_neighbours=variant != "spatial_control"
        )
    return IdentityMLP(n_sensors)


def scores(pred, truth):
    return [masked_errors(pred[:, i], truth[:, i]) for i in range(3)]


def train(data, truth, args, variant, directory):
    torch.manual_seed(args.seed)
    model = make_model(variant, data.n_sensors)
    fit_forecaster(
        model,
        data,
        truth,
        seed=args.seed,
        epochs=args.epochs,
        variant=variant,
        directory=directory,
        objective=OBJECTIVES[variant],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS[:4])
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    if args.epochs <= 0 or args.threads <= 0:
        parser.error("epochs and threads must be positive")
    torch.set_num_threads(args.threads)
    ds = load_metr_la()
    data = (
        NeighbourIdentityData(ds, "cpu")
        if any(v.startswith("spatial") for v in args.variants)
        else IdentityData(ds, "cpu")
    )
    starts, hs = ds.splits["val"], np.array(REPORT_HORIZONS)
    truth = ds.speed[starts[:, None] + IN_STEPS + hs[None] - 1]
    out = args.output_dir / f"seed_{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    contract = dict(
        scope="validation only; no test scoring",
        data_sha256=hashlib.sha256(ds.speed.tobytes()).hexdigest(),
        adjacency_sha256=hashlib.sha256(ds.adjacency.tobytes()).hexdigest(),
        sensor_ids=ds.sensor_ids,
        horizons=list(REPORT_HORIZONS),
        normalizer=[data.mean, data.std],
        seed=args.seed,
        epochs=args.epochs,
        threads=args.threads,
        selection="minimum mean validation MAE across horizons; no RMSE-based reselection",
        optimizer="AdamW lr=.001 wd=.0001; clip=1; plateau factor=.5 patience=3 threshold=.0001; early stopping=8",
        sampling="1,000,000 fresh rows without replacement per epoch; batch 2048; matched seed plan",
        objectives={
            "huber": "smooth L1 beta=5 mph / std",
            "mixed": "(.8*MAE + .2*MSE/std)/std",
            "spatial": "MAE/std",
            "spatial_control": "MAE/std",
        },
        python=platform.python_version(),
        torch=str(torch.__version__),
    )
    contract_path = out / "contract.json"
    if "spatial_mixed" in args.variants:
        contract["objectives"]["spatial_mixed"] = contract["objectives"]["mixed"]
    if contract_path.exists():
        if json.loads(contract_path.read_text()) != contract:
            raise ValueError(
                "Experiment contract differs; preserve the prior experiment"
            )
    elif args.evaluate_only:
        raise FileNotFoundError(contract_path)
    else:
        contract_path.write_text(json.dumps(contract, indent=2) + "\n")
    baseline_history = (
        pd.read_csv(
            ROOT / "outputs/experiments/identity_mlp/seed_42/identity/history.csv"
        )
        if args.seed == 42
        else None
    )
    selections, rows = [], []
    for variant in args.variants:
        directory = out / variant
        directory.mkdir(exist_ok=True)
        if not args.evaluate_only:
            train(data, truth, args, variant, directory)
        checkpoint = torch.load(directory / "model.pt", weights_only=True)
        assert checkpoint["variant"] == variant and checkpoint["seed"] == args.seed
        assert checkpoint["objective"] == OBJECTIVES[variant]
        np.testing.assert_allclose(
            [checkpoint["mean"], checkpoint["std"]],
            [data.mean, data.std],
            rtol=0,
            atol=1e-7,
        )
        model = make_model(variant, ds.n_sensors)
        model.load_state_dict(checkpoint["state_dict"])
        pred = predict_identity(model, data, "val")
        measured = scores(pred, truth)
        np.testing.assert_allclose(
            np.mean([r["mae"] for r in measured]),
            checkpoint["val_mae"],
            rtol=0,
            atol=1e-6,
        )
        path = ROOT / f"outputs/predictions/followup_{variant}_seed{args.seed}_val.npz"
        if args.evaluate_only and path.exists():
            np.testing.assert_array_equal(
                load_predictions(path, starts, hs, truth.shape), pred
            )
        np.savez_compressed(path, pred=pred, starts=starts, horizons=hs)
        history = pd.read_csv(directory / "history.csv")
        if baseline_history is not None:
            common = min(len(history), len(baseline_history))
            pd.testing.assert_series_equal(
                history.sample_sha256[:common],
                baseline_history.sample_sha256[:common],
                check_names=False,
            )
        selections.append(
            dict(
                variant=variant,
                seed=args.seed,
                best_epoch=checkpoint["best_epoch"],
                val_mae=checkpoint["val_mae"],
                parameters=sum(p.numel() for p in model.parameters()),
                matched_baseline_sample_epochs=common
                if baseline_history is not None
                else 0,
                checkpoint_sha256=hashlib.sha256(
                    (directory / "model.pt").read_bytes()
                ).hexdigest(),
            )
        )
        rows.extend(
            dict(variant=variant, minutes=int(h * 5), **r) for h, r in zip(hs, measured)
        )
    for filename, records, keys in [
        ("selection.csv", selections, ["variant", "seed"]),
        ("validation.csv", rows, ["variant", "minutes"]),
    ]:
        path = out / filename
        frame = pd.DataFrame(records)
        if path.exists():
            frame = pd.concat([pd.read_csv(path), frame]).drop_duplicates(
                keys, keep="last"
            )
        frame.to_csv(path, index=False)
    print(pd.DataFrame(selections).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
