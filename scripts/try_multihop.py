"""Bounded, validation-only AI-assisted spatial pilot. Run --prepare first."""

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

OUT = ROOT / "outputs/experiments/multihop_geometry_pilot"
VARIANTS = ("local", "one", "two", "geometry")


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
    args = parser.parse_args()
    torch.set_num_threads(4)
    ds = dataset()
    OUT.mkdir(parents=True, exist_ok=True)
    contract = dict(
        status="pilot",
        assistance="AI-assisted exploratory material; student review required",
        variants=VARIANTS,
        seed=42,
        epochs=3,
        rows_per_epoch=100_000,
        batch_size=2048,
        selection="minimum mean validation MAE across 3,6,12 steps; no test evaluation",
        optimizer="shared fit_forecaster defaults; masked MAE; train-only normalization",
        confounds="one seed; short budget; reused validation period; wider encoder than historical baseline; no historical results reused",
        geometry="weighted east/north displacement, straight distance, listed road distance in km for each orientation; no external road metadata",
        source={
            str(p.relative_to(ROOT)): digest(p.read_bytes())
            for p in sorted((ROOT / "src").rglob("*.py"))
        }
        | {"scripts/try_multihop.py": digest(Path(__file__).read_bytes())},
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
    directory = OUT / args.variant
    directory.mkdir(exist_ok=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    data = MultihopData(ds, device)
    model = MultihopMLP(ds.n_sensors, args.variant).to(device)
    environment = dict(
        python=platform.python_version(),
        torch=str(torch.__version__),
        numpy=np.__version__,
        cuda=torch.version.cuda,
        device=torch.cuda.get_device_name() if device == "cuda" else "cpu",
        threads=4,
        parameters=sum(p.numel() for p in model.parameters()),
        packages=subprocess.check_output(["uv", "pip", "freeze"], text=True),
    )
    (directory / "environment.json").write_text(
        json.dumps(environment, indent=2) + "\n"
    )
    truth = ds.speed[
        ds.splits["val"][:, None] + IN_STEPS + np.array(REPORT_HORIZONS) - 1
    ]
    started = time.perf_counter()
    fit_forecaster(
        model,
        data,
        truth,
        seed=42,
        epochs=3,
        variant=args.variant,
        directory=directory,
        rows_per_epoch=100_000,
    )
    checkpoint = torch.load(directory / "model.pt", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    prediction = predict_identity(model, data, "val")
    (directory / "selection.json").write_text(
        json.dumps(
            dict(
                best_epoch=checkpoint["best_epoch"],
                val_mae=checkpoint["val_mae"],
                checkpoint_sha256=digest((directory / "model.pt").read_bytes()),
                horizons={
                    str(h): masked_errors(prediction[:, i], truth[:, i])
                    for i, h in enumerate(REPORT_HORIZONS)
                },
            ),
            indent=2,
        )
        + "\n"
    )
    (directory / "runtime.json").write_text(
        json.dumps(dict(seconds=time.perf_counter() - started, completed=True)) + "\n"
    )


if __name__ == "__main__":
    main()
