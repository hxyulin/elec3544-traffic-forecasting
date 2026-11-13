"""Bounded AI-assisted validation-only seq2seq study; no test evaluation."""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from try_capacity import dataset

from traffic.data import ROOT
from traffic.dcrnn import DCRNN, batches, masked_mae

OUT = ROOT / "outputs/experiments/teacher_schedule"


def sha(b):
    return hashlib.sha256(b).hexdigest()


def weights(model):
    return sha(
        b"".join(
            k.encode() + v.detach().cpu().numpy().tobytes()
            for k, v in model.named_parameters()
        )
    )


def predict(model, ds, device):
    model.eval()
    results = []
    with torch.no_grad():
        for starts in np.array_split(
            ds.splits["val"], max(1, int(np.ceil(len(ds.splits["val"]) / 64)))
        ):
            x, _ = batches(ds, starts, device)
            results.append(model(x).cpu().numpy())
    mean, std = ds.normalizer()
    return np.concatenate(results) * std + mean


def probability(schedule, update, updates_per_epoch):
    if schedule == "A":
        return float(2000 / (2000 + np.exp(update / 2000)))
    return max(0.0, 1 - update / (30 * updates_per_epoch))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["pilot", "prepare", "fit"], required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--disabled", action="store_true")
    p.add_argument("--schedule", choices=["A", "B"], default="A")
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    ds = dataset()
    device = "cuda"
    model = DCRNN(ds.adjacency, not args.disabled).to(device)
    OUT.mkdir(parents=True, exist_ok=True)
    source = {
        str(f.relative_to(ROOT)): sha(f.read_bytes())
        for f in [
            ROOT / "src/traffic/dcrnn.py",
            Path(__file__),
            ROOT / "scripts/try_dcrnn.py",
            ROOT / "scripts/try_capacity.py",
            ROOT / "src/traffic/data.py",
        ]
    }
    if args.mode == "pilot":
        x, y = batches(ds, ds.splits["train"][:16], device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        torch.cuda.synchronize()
        t = time.perf_counter()
        losses = []
        for _ in range(40):
            model.train()
            optimizer.zero_grad()
            loss = masked_mae(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            losses.append(loss.item())
        torch.cuda.synchronize()
        seconds = (time.perf_counter() - t) / 40
        t = time.perf_counter()
        prediction = predict(model, ds, device)
        torch.cuda.synchronize()
        evaluation_seconds = time.perf_counter() - t
        result = dict(
            batch_size=16,
            steps=40,
            losses=losses,
            seconds_per_batch=seconds,
            evaluation_seconds=evaluation_seconds,
            finite=bool(np.isfinite(prediction).all()),
            peak_memory=torch.cuda.max_memory_allocated(),
            forecast_seconds_per_epoch=seconds
            * np.ceil((1_000_000 // ds.n_sensors) / 16)
            + evaluation_seconds,
        )
        (OUT / "pilot.json").write_text(json.dumps(result, indent=2) + "\n")
        print(result, flush=True)
        return
    assert json.loads((OUT / "pilot.json").read_text())["finite"]
    # Every candidate receives the same fixed budget.
    epochs = 50
    contract = dict(
        source=source,
        architecture="one layer hidden32; two Chebyshev steps per transposed random walk; independent direction reset; 5 slots",
        objective="all12 observed masked MAE",
        selection="mean validation MAE steps3/6/12",
        epochs=epochs,
        patience=None,
        batch_size=16,
        windows_per_epoch=min(1_000_000 // ds.n_sensors, len(ds.splits["train"])),
        seeds=[42, 43, 44],
        sampling="numpy default_rng(seed) distinct train windows per epoch; torch CPU Generator(seed+1000), scalar per decoder step, missing fallback prediction",
        scheduled_sampling={
            "A": "2000/(2000+exp(u/2000))",
            "B": "max(0,1-u/(30*updates_per_epoch)))",
        },
        update_index="zero based before optimizer update",
        selection_scope="best whole50; retained best<=30 and best>30",
        optimizer="AdamW lr.001 wd1e-4 clip1 plateau factor.5 patience3 threshold1e-4",
        normalizer=ds.normalizer(),
        splits={k: [int(v[0]), int(v[-1]), len(v)] for k, v in ds.splits.items()},
        arrays={
            k: sha(v.tobytes())
            for k, v in dict(
                speed=ds.speed, adjacency=ds.adjacency, filled=ds.filled, **ds.splits
            ).items()
        },
        data={
            f.name: sha(f.read_bytes())
            for f in (ROOT / "data/raw/extracted").iterdir()
            if f.is_file()
        },
        run_timeout=1800,
        total_timeout=10800,
        assistance="AI-assisted exploratory work for student inspection",
    )
    contract["initialization"] = {}
    for seed in (42, 43, 44):
        torch.manual_seed(seed)
        a = DCRNN(ds.adjacency)
        torch.manual_seed(seed)
        b = DCRNN(ds.adjacency, False)
        assert weights(a) == weights(b)
        contract["initialization"][str(seed)] = weights(a)
    protocol = OUT / "protocol.json"
    if args.mode == "prepare":
        assert not protocol.exists()
        protocol.write_text(json.dumps(contract, indent=2) + "\n")
        print("Frozen", epochs, "epochs", flush=True)
        return
    assert json.loads(protocol.read_text()) == json.loads(json.dumps(contract))
    torch.manual_seed(args.seed)
    model = DCRNN(ds.adjacency, not args.disabled).to(device)
    assert weights(model) == contract["initialization"][str(args.seed)]
    directory = (
        OUT
        / args.schedule
        / f"seed{args.seed}"
        / ("disabled" if args.disabled else "enabled")
    )
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "environment.json").write_text(
        json.dumps(
            dict(
                python=platform.python_version(),
                torch=str(torch.__version__),
                cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(),
                parameters=sum(p.numel() for p in model.parameters()),
            ),
            indent=2,
        )
        + "\n"
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=3
    )
    rng = np.random.default_rng(args.seed)
    sampling = torch.Generator().manual_seed(args.seed + 1000)
    audit_rng = torch.Generator().manual_seed(args.seed + 1000)
    truth = ds.speed[ds.splits["val"][:, None] + 12 + np.array([2, 5, 11])]
    best, stale, step, history = float("inf"), 0, 0, []
    torch.cuda.synchronize()
    start = time.perf_counter()
    phase_best = {"prefix": float("inf"), "post": float("inf")}
    for epoch in range(1, epochs + 1):
        chosen = rng.choice(
            ds.splits["train"], contract["windows_per_epoch"], replace=False
        )
        exposure = 0
        forced = eligible = forced_observed = 0
        epoch_p = []
        model.train()
        losses = []
        rng_before = sha(sampling.get_state().numpy().tobytes())
        for indices in np.array_split(chosen, int(np.ceil(len(chosen) / 16))):
            x, y = batches(ds, indices, device)
            exposure += int(torch.isfinite(y).sum())
            prob = probability(
                args.schedule, step, int(np.ceil(contract["windows_per_epoch"] / 16))
            )
            epoch_p.append(prob)
            draws = torch.rand(11, generator=audit_rng)
            forced += int((draws < prob).sum())
            eligible += 11
            forced_observed += int(torch.isfinite(y[:, :11])[:, draws < prob].sum())
            optimizer.zero_grad()
            loss = masked_mae(model(x, y, prob, sampling), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            losses.append(loss.item())
            step += 1
        assert torch.equal(sampling.get_state(), audit_rng.get_state())
        torch.cuda.synchronize()
        prediction = predict(model, ds, device)[:, [2, 5, 11]]
        metrics = []
        for i in range(3):
            error = (prediction[:, i] - truth[:, i])[truth[:, i] > 0].astype(float)
            metrics.append(
                dict(
                    mae=float(abs(error).mean()), rmse=float(np.sqrt((error**2).mean()))
                )
            )
        assert np.isfinite(prediction).all()
        score = float(np.mean([m["mae"] for m in metrics]))
        scheduler.step(score)
        history.append(
            dict(
                epoch=epoch,
                probability_first=epoch_p[0],
                probability_last=epoch_p[-1],
                forced_transitions=forced,
                eligible_transitions=eligible,
                forced_observed=forced_observed,
                metrics=json.dumps(metrics),
                loss=float(np.mean(losses)),
                val_mae=score,
                windows=len(chosen),
                observed_targets=exposure,
                sample_sha256=sha(chosen.tobytes()),
                sampling_before=rng_before,
                sampling_after=sha(sampling.get_state().numpy().tobytes()),
                seconds=time.perf_counter() - start,
                lr=optimizer.param_groups[0]["lr"],
            )
        )
        pd.DataFrame(history).to_csv(directory / "history.csv", index=False)
        print(args.seed, args.disabled, epoch, score, flush=True)
        phase = "prefix" if epoch <= 30 else "post"
        if score < phase_best[phase]:
            phase_best[phase] = score
            torch.save(
                dict(state_dict=model.state_dict(), epoch=epoch, metrics=metrics),
                directory / f"{phase}.pt",
            )
            np.save(directory / f"{phase}.npy", prediction)
        if score < best:
            best, stale = score, 0
            torch.save(
                dict(state_dict=model.state_dict(), epoch=epoch, metrics=metrics),
                directory / "model.pt",
            )
            np.save(directory / "prediction.npy", prediction)
        else:
            stale += 1
        if time.perf_counter() - start > 1750:
            raise TimeoutError("fit budget exhausted")
    for phase in ("prefix", "post"):
        saved = torch.load(directory / f"{phase}.pt", weights_only=True)
        model.load_state_dict(saved["state_dict"])
        actual = predict(model, ds, device)[:, [2, 5, 11]]
        assert np.array_equal(actual, np.load(directory / f"{phase}.npy"))
        (directory / f"{phase}.json").write_text(
            json.dumps(
                dict(
                    epoch=saved["epoch"],
                    metrics=saved["metrics"],
                    restored_exact=True,
                    prediction_sha256=sha(actual.tobytes()),
                    checkpoint_sha256=sha((directory / f"{phase}.pt").read_bytes()),
                ),
                indent=2,
            )
            + "\n"
        )
    checkpoint = torch.load(directory / "model.pt", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    restored = predict(model, ds, device)[:, [2, 5, 11]]
    assert np.array_equal(restored, np.load(directory / "prediction.npy"))
    torch.cuda.synchronize()
    (directory / "result.json").write_text(
        json.dumps(
            dict(
                completed=True,
                epochs=len(history),
                best_epoch=checkpoint["epoch"],
                metrics=checkpoint["metrics"],
                restored_exact=True,
                prediction_sha256=sha(restored.tobytes()),
                checkpoint_sha256=sha((directory / "model.pt").read_bytes()),
                seconds=time.perf_counter() - start,
                peak_memory=torch.cuda.max_memory_allocated(),
                windows=sum(h["windows"] for h in history),
                observed_targets=sum(h["observed_targets"] for h in history),
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
