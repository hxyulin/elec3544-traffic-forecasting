"""Bounded, validation-only AI-assisted spatial replication. Run --prepare first."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from traffic.data import (
    IN_STEPS,
    OUT_STEPS,
    REPORT_HORIZONS,
    ROOT,
    causal_fill,
    load_metr_la,
)
from traffic.identity import predict_identity
from traffic.metrics import masked_errors
from traffic.multihop import MultihopData, MultihopMLP
from traffic.training import fit_forecaster

OUT = ROOT / "outputs/experiments/capacity_budget"
VARIANTS = ("one", "two")


def digest(value):
    return hashlib.sha256(value).hexdigest()


def dataset():
    ds = load_metr_la()
    # Retain benchmark validation period, purge all intersecting train targets.
    boundary = int(ds.splits["val"][0]) + IN_STEPS
    splits = {
        "train": ds.splits["train"][
            ds.splits["train"] + IN_STEPS + OUT_STEPS <= boundary
        ],
        "val": ds.splits["val"],
    }
    return replace(ds, splits=splits, filled=causal_fill(ds.speed, boundary))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), default=42)
    parser.add_argument("--size", choices=("current", "wide"), default="current")
    args = parser.parse_args()
    torch.set_num_threads(4)
    ds = dataset()
    OUT.mkdir(parents=True, exist_ok=True)
    contract = dict(
        status="replication",
        assistance="AI-assisted exploratory material; student review required",
        variants=VARIANTS,
        sizes={"current": [64, 32, 128], "wide": [128, 64, 224]},
        prefix_budget=50,
        seeds=[42, 43, 44],
        epochs=100,
        rows_per_epoch=1_000_000,
        batch_size=2048,
        selection="minimum mean validation MAE across 3,6,12 steps; no test evaluation",
        optimizer="shared fit_forecaster defaults; masked MAE; train-only normalization",
        confounds="three seeds; capped training; reused validation period; wider encoder than historical baseline; no historical results reused",
        geometry="weighted east/north displacement, straight distance, listed road distance in km for each orientation; no external road metadata",
        source={
            str(p.relative_to(ROOT)): digest(p.read_bytes())
            for p in sorted((ROOT / "src").rglob("*.py"))
        }
        | {"scripts/try_capacity.py": digest(Path(__file__).read_bytes())},
        data={
            p.name: digest(p.read_bytes())
            for p in sorted((ROOT / "data/raw/extracted").glob("*"))
            if p.is_file()
        },
        arrays={
            k: digest(v.tobytes())
            for k, v in dict(
                speed=ds.speed,
                adjacency=ds.adjacency,
                distance=ds.distance,
                locations=ds.locations[["latitude", "longitude"]].to_numpy(),
                **ds.splits,
            ).items()
        },
        normalizer=ds.normalizer(),
        splits={k: [int(v[0]), int(v[-1]), len(v)] for k, v in ds.splits.items()},
    )
    contract["initialization"] = {}
    for size in ("current", "wide"):
        for seed in (42, 43, 44):
            values = []
            for variant in VARIANTS:
                torch.manual_seed(seed)
                candidate = MultihopMLP(ds.n_sensors, variant, size)
                values.append(
                    digest(
                        b"".join(
                            name.encode() + param.detach().numpy().tobytes()
                            for name, param in candidate.named_parameters()
                        )
                    )
                )
            assert len(set(values)) == 1
            contract["initialization"][f"{size}/{seed}"] = dict(
                sha256=values[0],
                parameters=sum(p.numel() for p in candidate.parameters()),
            )
    contract["policy"] = dict(
        patience=8,
        lr=0.001,
        weight_decay=1e-4,
        scheduler="ReduceLROnPlateau factor .5 patience 3 threshold 1e-4",
        clipping=1.0,
        timeout_per_run_seconds=900,
        overall_timeout_seconds=10800,
    )
    path = OUT / "protocol.json"
    contract = json.loads(json.dumps(contract))
    if args.prepare:
        if path.exists():
            assert json.loads(path.read_text()) == contract
        else:
            path.write_text(json.dumps(contract, indent=2) + "\n")
        print("Protocol frozen", flush=True)
        return
    assert json.loads(path.read_text()) == contract, (
        "Run --prepare before fitting; source/data contract must match"
    )
    assert args.variant
    directory = OUT / f"seed{args.seed}" / args.size / args.variant
    directory.mkdir(parents=True, exist_ok=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    setup_started = time.perf_counter()
    data = MultihopData(ds, device)
    model = MultihopMLP(ds.n_sensors, args.variant, args.size).to(device)
    torch.cuda.synchronize() if device == "cuda" else None
    feature_setup_seconds = time.perf_counter() - setup_started
    environment = dict(
        python=platform.python_version(),
        torch=str(torch.__version__),
        numpy=np.__version__,
        cuda=torch.version.cuda,
        device=torch.cuda.get_device_name() if device == "cuda" else "cpu",
        threads=4,
        parameters=sum(p.numel() for p in model.parameters()),
        packages="\n".join(
            line
            for line in subprocess.check_output(
                ["uv", "pip", "freeze"], text=True
            ).splitlines()
            if not line.startswith("-e ")
        ),
    )
    (directory / "environment.json").write_text(
        json.dumps(environment, indent=2) + "\n"
    )
    truth = ds.speed[
        ds.splits["val"][:, None] + IN_STEPS + np.array(REPORT_HORIZONS) - 1
    ]
    torch.cuda.synchronize() if device == "cuda" else None
    started = time.perf_counter()
    fit_forecaster(
        model,
        data,
        truth,
        seed=args.seed,
        epochs=100,
        variant=args.variant,
        directory=directory,
        rows_per_epoch=1_000_000,
        record_prediction=True,
        prefix_epoch=50,
    )
    import pandas as pd

    history = pd.read_csv(directory / "history.csv")
    for checkpoint_file, prediction_file, selection_file in [
        ("model.pt", "best_prediction.npy", "selection.json"),
        ("prefix_model.pt", "prefix_prediction.npy", "prefix_selection.json"),
    ]:
        checkpoint = torch.load(directory / checkpoint_file, weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])
        prediction = predict_identity(model, data, "val")
        stored = np.load(directory / prediction_file)
        assert np.array_equal(stored, prediction), "Restored predictions differ"
        assert (
            float(
                np.mean(
                    [
                        masked_errors(prediction[:, i], truth[:, i])["mae"]
                        for i in range(3)
                    ]
                )
            )
            == checkpoint["val_mae"]
        )
        (directory / selection_file).write_text(
            json.dumps(
                dict(
                    restored_prediction_exact=True,
                    prediction_sha256=digest(prediction.tobytes()),
                    best_epoch=checkpoint["best_epoch"],
                    val_mae=checkpoint["val_mae"],
                    checkpoint_sha256=digest(
                        (directory / checkpoint_file).read_bytes()
                    ),
                    horizons={
                        str(h): masked_errors(prediction[:, i], truth[:, i])
                        for i, h in enumerate(REPORT_HORIZONS)
                    },
                ),
                indent=2,
            )
            + "\n"
        )
    (directory / "convergence.json").write_text(
        json.dumps(
            dict(
                observed_epochs=len(history),
                cap_reached=len(history) == 100,
                early_stopped=len(history) < 100,
                prefix_score=float(history.loc[history.epoch <= 50, "val_mae"].min()),
                final_score=float(history.val_mae.min()),
            ),
            indent=2,
        )
        + "\n"
    )
    torch.cuda.synchronize() if device == "cuda" else None
    (directory / "runtime.json").write_text(
        json.dumps(
            dict(
                seconds=time.perf_counter() - started,
                feature_setup_seconds=feature_setup_seconds,
                completed=True,
            )
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
