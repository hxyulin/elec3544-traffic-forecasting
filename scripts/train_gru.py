"""Train validation-selected per-sensor GRUs and evaluate the selected checkpoint.

    uv run --extra neural scripts/train_gru.py
    uv run --extra neural scripts/train_gru.py --evaluate-only
    uv run --extra neural scripts/train_gru.py --model graph_gru --device cpu
    uv run --extra neural scripts/train_gru.py --model graph_gru_no_graph --device cpu

The fixed training sample is shared by all trials. Only validation scores select
settings and epochs; test predictions are produced once after selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from traffic.data import REPORT_HORIZONS, ROOT, load_metr_la
from traffic.metrics import errors_by_horizon
from traffic.neural import GraphSensorGRU, NeuralData, SensorGRU, masked_mae, predict

OUT = ROOT / "outputs" / "neural"


def train_trial(data, truth, learning_rate, args):
    torch.manual_seed(args.seed)
    model_type = SensorGRU if args.model == "gru" else GraphSensorGRU
    model = model_type(args.hidden_size).to(data.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Fixed rows, then independently shuffled each epoch. Missing horizons stay masked.
    rng = np.random.default_rng(args.seed)
    selected = rng.choice(
        data.size("train"), min(args.train_rows, data.size("train")), replace=False
    )
    best_score, best_epoch, best_state = np.inf, 0, None
    rows = []
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        model.train()
        total_error, total_count = 0.0, 0
        for start in range(0, len(selected), args.batch_size):
            indices = selected[start : start + args.batch_size]
            x, y = data.batch("train", indices)
            optimizer.zero_grad(set_to_none=True)
            p = model(x) * data.std + data.mean
            loss = masked_mae(p, y) / data.std
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = int((y > 0).sum().item())
            total_error += float(loss.detach().item()) * data.std * count
            total_count += count
        prediction = predict(model, data, "val")
        scores = errors_by_horizon(prediction, truth, REPORT_HORIZONS)
        score = float(np.mean([r["mae"] for r in scores]))
        elapsed = time.perf_counter() - started
        rows.append(
            {
                "learning_rate": learning_rate,
                "epoch": epoch,
                "train_mae": total_error / total_count,
                "val_mae": score,
                "seconds": elapsed,
            }
        )
        print(
            f"lr={learning_rate} epoch={epoch}: train MAE {rows[-1]['train_mae']:.4f}, val MAE {score:.4f}, {elapsed:.1f}s",
            flush=True,
        )
        if not np.isfinite(score):
            raise RuntimeError("Non-finite validation error")
        if score < best_score:
            best_score, best_epoch = score, epoch
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        elif epoch - best_epoch >= args.patience:
            break
        rng.shuffle(selected)
    return best_score, best_epoch, best_state, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=["gru", "graph_gru", "graph_gru_no_graph"], default="gru"
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu", "mps", "cuda"], default="auto"
    )
    parser.add_argument(
        "--learning-rates", type=float, nargs="+", default=[0.001, 0.003]
    )
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--train-rows", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    for key in ("hidden_size", "epochs", "patience", "train_rows", "batch_size"):
        if getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if any(lr <= 0 for lr in args.learning_rates):
        parser.error("Learning rates must be positive")
    device = args.device
    if device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    torch.set_num_threads(4)
    ds = load_metr_la()
    graph_mode = {"gru": "none", "graph_gru": "road", "graph_gru_no_graph": "identity"}[
        args.model
    ]
    data = NeuralData(ds, device, graph_mode)
    out = OUT if args.model == "gru" else OUT / args.model
    checkpoint_path = out / f"{args.model}.pt"
    adjacency_hash = hashlib.sha256(ds.adjacency.tobytes()).hexdigest()
    # Keep test truth and predictions out of the trial-selection loop.
    val_truth = ds.windows("val")[1]
    out.mkdir(parents=True, exist_ok=True)
    if not args.evaluate_only:
        histories, trials = [], []
        best = np.inf
        for lr in args.learning_rates:
            score, epoch, state, history = train_trial(data, val_truth, lr, args)
            histories.extend(history)
            trials.append({"learning_rate": lr, "best_epoch": epoch, "val_mae": score})
            pd.DataFrame(histories).to_csv(out / "history.csv", index=False)
            pd.DataFrame(trials).to_csv(out / "tuning.csv", index=False)
            if score < best:
                best = score
                torch.save(
                    {
                        "state_dict": state,
                        "model": args.model,
                        "graph_mode": graph_mode,
                        "adjacency_sha256": adjacency_hash,
                        "hidden_size": args.hidden_size,
                        "learning_rate": lr,
                        "best_epoch": epoch,
                        "val_mae": score,
                        "mean": data.mean,
                        "std": data.std,
                        "args": vars(args),
                        "sensor_ids": ds.sensor_ids,
                        "horizons": list(REPORT_HORIZONS),
                        "data_sha256": hashlib.sha256(ds.speed.tobytes()).hexdigest(),
                    },
                    checkpoint_path,
                )
        fig, ax = plt.subplots(figsize=(7, 4))
        for lr, part in pd.DataFrame(histories).groupby("learning_rate"):
            ax.plot(part.epoch, part.val_mae, label=f"Validation, lr={lr}")
            ax.plot(
                part.epoch,
                part.train_mae,
                linestyle="--",
                alpha=0.7,
                label=f"Training, lr={lr}",
            )
        ax.set(xlabel="Epoch", ylabel="MAE (mph)")
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(out / "learning_curves.png", dpi=160)
        plt.close(fig)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("model", "gru") != args.model:
        raise ValueError("Checkpoint model does not match the requested model")
    if args.model != "gru" and checkpoint["adjacency_sha256"] != adjacency_hash:
        raise ValueError("Checkpoint was trained on a different sensor graph")
    if checkpoint["sensor_ids"] != ds.sensor_ids or checkpoint["horizons"] != list(
        REPORT_HORIZONS
    ):
        raise ValueError("Checkpoint sensor order or horizons do not match the dataset")
    if checkpoint["data_sha256"] != hashlib.sha256(ds.speed.tobytes()).hexdigest():
        raise ValueError("Checkpoint was trained on a different dataset")
    np.testing.assert_allclose(
        [data.mean, data.std],
        [checkpoint["mean"], checkpoint["std"]],
        rtol=0,
        atol=1e-7,
    )
    model_type = SensorGRU if args.model == "gru" else GraphSensorGRU
    model = model_type(checkpoint["hidden_size"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    rows = []
    for split in ("val", "test"):
        prediction = predict(model, data, split)
        if not np.isfinite(prediction).all():
            raise ValueError("Non-finite predictions")
        rows.extend(
            {"model": args.model, "split": split, **r}
            for r in errors_by_horizon(
                prediction, ds.windows(split)[1], REPORT_HORIZONS
            )
        )
        if split == "test":
            directory = ROOT / "outputs" / "predictions"
            directory.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                directory / f"{args.model}.npz",
                pred=prediction,
                starts=ds.splits[split],
                horizons=np.array(REPORT_HORIZONS),
            )
    metrics = ROOT / "outputs" / "results" / "metrics.csv"
    old = pd.read_csv(metrics) if metrics.exists() else pd.DataFrame(columns=["model"])
    metrics.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(
        [old[old.model != args.model], pd.DataFrame(rows)], ignore_index=True
    ).to_csv(metrics, index=False, float_format="%.4f")
    metadata = {
        "model": args.model,
        "graph_mode": graph_mode,
        "adjacency_sha256": adjacency_hash,
        "torch": str(torch.__version__),
        "python": platform.python_version(),
        "device": device,
        "parameters": sum(p.numel() for p in model.parameters()),
        "selected_learning_rate": checkpoint["learning_rate"],
        "best_epoch": checkpoint["best_epoch"],
        "selection_val_mae": checkpoint["val_mae"],
        "normalizer": {"mean": data.mean, "std": data.std},
        "training_args": checkpoint["args"],
        "data_sha256": checkpoint["data_sha256"],
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "limitations": [
            "One seed; device-specific numerical differences are possible.",
            "Fixed training sample of window/sensor rows, not every training row.",
            "Benchmark target periods overlap by 11 timestamps at boundaries.",
            "Direct outputs only at 15, 30 and 60 minutes; no recursive future inputs.",
        ],
    }
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
