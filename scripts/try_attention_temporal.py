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

from traffic.attention_followups import TemporalMLP
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
from traffic.spatial_identity import NeighbourIdentityData
from traffic.training import fit_forecaster

OUT = ROOT / "outputs/experiments/attention_temporal"
VARIANTS = ("sinusoidal", "rope")


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
    args = parser.parse_args()
    torch.set_num_threads(4)
    ds = dataset()
    OUT.mkdir(parents=True, exist_ok=True)
    contract = dict(
        status="replication",
        assistance="AI-assisted exploratory material; student review required",
        variants=VARIANTS,
        seeds=[42, 43, 44],
        epochs=50,
        rows_per_epoch=1_000_000,
        batch_size=2048,
        selection="minimum mean validation MAE across 3,6,12 steps; no test evaluation",
        optimizer="shared fit_forecaster defaults; masked MAE; train-only normalization",
        confounds="three seeds; capped training; reused validation period; wider encoder than historical baseline; no historical results reused",
        design="See docs/attention_followups.md; 12 historical timestamp tokens; width32 heads4 FFN64; last token pooling; sin versus rotary",
        source={
            str(p.relative_to(ROOT)): digest(p.read_bytes())
            for p in sorted((ROOT / "src").rglob("*.py"))
        }
        | {"scripts/try_attention_temporal.py": digest(Path(__file__).read_bytes())},
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
    contract["parameters"] = sum(
        p.numel() for p in TemporalMLP(ds.n_sensors, VARIANTS[0]).parameters()
    )
    contract["initialization"] = {}
    for seed in (42, 43, 44):
        values = []
        for variant in VARIANTS:
            torch.manual_seed(seed)
            candidate = TemporalMLP(ds.n_sensors, variant)
            values.append(
                digest(
                    b"".join(
                        name.encode() + param.detach().numpy().tobytes()
                        for name, param in candidate.named_parameters()
                    )
                )
            )
        assert len(set(values)) == 1
        contract["initialization"][str(seed)] = values[0]
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
    directory = OUT / f"seed{args.seed}" / args.variant
    directory.mkdir(parents=True, exist_ok=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    setup_started = time.perf_counter()
    data = NeighbourIdentityData(ds, device)
    model = TemporalMLP(ds.n_sensors, args.variant).to(device)
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
        epochs=50,
        variant=args.variant,
        directory=directory,
        rows_per_epoch=1_000_000,
        record_prediction=True,
    )
    checkpoint = torch.load(directory / "model.pt", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    prediction = predict_identity(model, data, "val")
    stored = np.load(directory / "best_prediction.npy")
    assert np.array_equal(stored, prediction), "Restored predictions differ"
    assert (
        float(
            np.mean(
                [masked_errors(prediction[:, i], truth[:, i])["mae"] for i in range(3)]
            )
        )
        == checkpoint["val_mae"]
    )
    (directory / "selection.json").write_text(
        json.dumps(
            dict(
                restored_prediction_exact=True,
                prediction_sha256=digest(prediction.tobytes()),
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
