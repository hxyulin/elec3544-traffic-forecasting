"""Validation-only comparison of residual MLPs with and without sensor identity.

    uv run --extra neural scripts/investigate_neural.py
    uv run --extra neural scripts/investigate_neural.py --evaluate-only

This runner never scores test predictions or modifies the published metrics.
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

from traffic.data import IN_STEPS, REPORT_HORIZONS, ROOT, load_metr_la
from traffic.identity import IdentityData, IdentityMLP, predict_identity
from traffic.metrics import masked_errors
from traffic.neural import masked_mae


def validation_scores(pred, true):
    return [masked_errors(pred[:, i], true[:, i]) for i in range(len(REPORT_HORIZONS))]


def fit(data, truth, args, variant, directory):
    torch.manual_seed(args.seed)
    model = IdentityMLP(data.n_sensors, use_identity=variant == "identity").to(
        args.device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, threshold=1e-4
    )
    rng = np.random.default_rng(args.seed)
    best = np.inf
    best_epoch = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        model.train()
        # Fresh rows each epoch; both variants share the same seeded sampling plan.
        selected = rng.choice(
            data.size("train"),
            min(args.rows_per_epoch, data.size("train")),
            replace=False,
        )
        sample_hash = hashlib.sha256(selected.tobytes()).hexdigest()
        error, count = 0.0, 0
        for start in range(0, len(selected), args.batch_size):
            inputs, y = data.batch("train", selected[start : start + args.batch_size])
            optimizer.zero_grad(set_to_none=True)
            p = model(*inputs) * data.std + data.mean
            loss = masked_mae(p, y) / data.std
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            n = int((y > 0).sum().item())
            error += float(loss.item()) * data.std * n
            count += n
        pred = predict_identity(model, data, "val")
        scores = validation_scores(pred, truth)
        score = float(np.mean([r["mae"] for r in scores]))
        if not np.isfinite(score):
            raise ValueError("Non-finite validation MAE")
        history.append(
            {
                "variant": variant,
                "seed": args.seed,
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_mae": error / count,
                "val_mae": score,
                "seconds": time.perf_counter() - started,
                "sample_sha256": sample_hash,
            }
        )
        pd.DataFrame(history).to_csv(directory / "history.csv", index=False)
        print(
            f"{variant}, epoch {epoch}: train {error / count:.4f}, val {score:.4f}, {history[-1]['seconds']:.1f}s",
            flush=True,
        )
        if score < best:
            best, best_epoch = score, epoch
            torch.save(
                {
                    "state_dict": {
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    },
                    "variant": variant,
                    "best_epoch": epoch,
                    "val_mae": score,
                    "mean": data.mean,
                    "std": data.std,
                    "n_sensors": data.n_sensors,
                    "settings": vars(args),
                    "scope": "validation only",
                },
                directory / "model.pt",
            )
        elif epoch - best_epoch >= args.patience:
            break
        scheduler.step(score)
    return history


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["shared", "identity"],
        default=["shared", "identity"],
    )
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--rows-per-epoch", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    if any(
        getattr(args, k) <= 0
        for k in ("epochs", "patience", "rows_per_epoch", "batch_size", "learning_rate")
    ):
        parser.error("Training counts and learning rate must be positive")
    torch.set_num_threads(4)
    ds = load_metr_la()
    data = IdentityData(ds, args.device)
    starts = ds.splits["val"]
    hs = np.array(REPORT_HORIZONS)
    truth = ds.speed[starts[:, None] + IN_STEPS + hs[None, :] - 1]
    out = ROOT / "outputs/experiments/identity_mlp" / f"seed_{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    contract = {
        "scope": "validation only; test results remain unchanged",
        "data_sha256": hashlib.sha256(ds.speed.tobytes()).hexdigest(),
        "sensor_ids": ds.sensor_ids,
        "horizons": list(REPORT_HORIZONS),
        "normalizer": [data.mean, data.std],
        "selection": "mean validation MAE across three horizons",
        "sampling": "fresh uniform sample without replacement each epoch; same prefix plan in both variants",
        "settings": {
            k: v
            for k, v in vars(args).items()
            if k not in ("evaluate_only", "variants")
        },
        "torch": str(torch.__version__),
        "python": platform.python_version(),
    }
    contract_path = out / "contract.json"
    if contract_path.exists():
        old = json.loads(contract_path.read_text())
        if old != contract:
            raise ValueError(
                "Existing experiment contract differs. Use another seed or preserve the prior experiment before changing its settings."
            )
    else:
        if args.evaluate_only:
            raise FileNotFoundError(contract_path)
        contract_path.write_text(json.dumps(contract, indent=2) + "\n")
    histories = []
    rows = []
    selections = []
    input_missing = ds.missing[starts[:, None] + np.arange(IN_STEPS)].sum(1) > 0
    for variant in args.variants:
        directory = out / variant
        directory.mkdir(exist_ok=True)
        if not args.evaluate_only:
            fit(data, truth, args, variant, directory)
        checkpoint = torch.load(
            directory / "model.pt", map_location="cpu", weights_only=True
        )
        if checkpoint["variant"] != variant or checkpoint["n_sensors"] != ds.n_sensors:
            raise ValueError("Checkpoint identity configuration differs")
        np.testing.assert_allclose(
            [checkpoint["mean"], checkpoint["std"]],
            [data.mean, data.std],
            rtol=0,
            atol=1e-7,
        )
        model = IdentityMLP(ds.n_sensors, use_identity=variant == "identity").to(
            args.device
        )
        model.load_state_dict(checkpoint["state_dict"])
        pred = predict_identity(model, data, "val")
        scores = validation_scores(pred, truth)
        np.testing.assert_allclose(
            np.mean([r["mae"] for r in scores]),
            checkpoint["val_mae"],
            atol=1e-6,
            rtol=0,
        )
        pred_dir = ROOT / "outputs/predictions"
        pred_dir.mkdir(exist_ok=True)
        np.savez_compressed(
            pred_dir / f"mlp_{variant}_seed{args.seed}_val.npz",
            pred=pred,
            starts=starts,
            horizons=hs,
        )
        selections.append(
            {
                "variant": variant,
                "seed": args.seed,
                "best_epoch": checkpoint["best_epoch"],
                "val_mae": checkpoint["val_mae"],
                "parameters": sum(p.numel() for p in model.parameters()),
            }
        )
        for i, h in enumerate(hs):
            y, p = truth[:, i], pred[:, i]
            masks = {
                "all_observed": y > 0,
                "below_40_mph": (y > 0) & (y < 40),
                "at_least_40_mph": y >= 40,
                "complete_input": (y > 0) & ~input_missing,
                "missing_input": (y > 0) & input_missing,
            }
            for condition, mask in masks.items():
                rows.append(
                    {
                        "variant": variant,
                        "seed": args.seed,
                        "minutes": int(h * 5),
                        "condition": condition,
                        **masked_errors(p[mask], y[mask]),
                    }
                )
        histories.append(pd.read_csv(directory / "history.csv"))
    # Merge selected variants without dropping records from earlier invocations.
    for filename, new, keys in [
        (
            "validation.csv",
            pd.DataFrame(rows),
            ["variant", "seed", "minutes", "condition"],
        ),
        ("selection.csv", pd.DataFrame(selections), ["variant", "seed"]),
    ]:
        path = out / filename
        if path.exists():
            new = pd.concat([pd.read_csv(path), new]).drop_duplicates(keys, keep="last")
        new.to_csv(path, index=False)
    for directory in out.iterdir():
        if (
            directory.is_dir()
            and (directory / "history.csv").exists()
            and directory.name not in args.variants
        ):
            histories.append(pd.read_csv(directory / "history.csv"))
    fig, ax = plt.subplots(figsize=(7, 4))
    for history in histories:
        name = history.variant.iloc[0]
        ax.plot(history.epoch, history.val_mae, label=name)
    ax.set(
        xlabel="Epoch",
        ylabel="Mean validation MAE (mph)",
        title="Validation only; fresh training sample each epoch",
    )
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "learning_curves.png", dpi=160)
    plt.close(fig)
    print(pd.DataFrame(selections).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
