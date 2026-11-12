"""Shared validation-selected training policy for the neural experiments."""

from __future__ import annotations

import hashlib
import time

import numpy as np
import pandas as pd
import torch

from .data import REPORT_HORIZONS
from .identity import predict_identity
from .metrics import masked_errors
from .neural import masked_mae
from .spatial_identity import forecast_loss


def fit_forecaster(
    model,
    data,
    truth,
    *,
    seed,
    epochs,
    variant,
    directory,
    objective="mae",
    rows_per_epoch=1_000_000,
    batch_size=2048,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=3, threshold=1e-4
    )
    rng = np.random.default_rng(seed)
    best, best_epoch, history = np.inf, 0, []
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        selected = rng.choice(
            data.size("train"), min(rows_per_epoch, data.size("train")), replace=False
        )
        total, count = 0.0, 0
        for start in range(0, len(selected), batch_size):
            inputs, y = data.batch("train", selected[start : start + batch_size])
            optimizer.zero_grad(set_to_none=True)
            pred = model(*inputs) * data.std + data.mean
            loss = forecast_loss(pred, y, data.std, objective)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            n = int((y > 0).sum())
            total += float(masked_mae(pred.detach(), y)) * n
            count += n
        predict = predict_identity(model, data, "val")
        measured = [
            masked_errors(predict[:, i], truth[:, i])
            for i in range(len(REPORT_HORIZONS))
        ]
        score = float(np.mean([r["mae"] for r in measured]))
        if not np.isfinite(score):
            raise ValueError("Non-finite validation score")
        history.append(
            dict(
                epoch=epoch,
                train_mae=total / count,
                val_mae=score,
                val_rmse=float(np.mean([r["rmse"] for r in measured])),
                learning_rate=optimizer.param_groups[0]["lr"],
                sample_sha256=hashlib.sha256(selected.tobytes()).hexdigest(),
                seconds=time.perf_counter() - started,
            )
        )
        pd.DataFrame(history).to_csv(directory / "history.csv", index=False)
        print(
            f"{variant} epoch {epoch}: validation MAE {score:.4f}, {history[-1]['seconds']:.1f}s",
            flush=True,
        )
        if score < best:
            best, best_epoch = score, epoch
            torch.save(
                dict(
                    state_dict={
                        k: v.detach().clone() for k, v in model.state_dict().items()
                    },
                    variant=variant,
                    seed=seed,
                    best_epoch=epoch,
                    val_mae=score,
                    mean=data.mean,
                    std=data.std,
                    n_sensors=data.n_sensors,
                    objective=objective,
                ),
                directory / "model.pt",
            )
        elif epoch - best_epoch >= 8:
            break
        scheduler.step(score)
