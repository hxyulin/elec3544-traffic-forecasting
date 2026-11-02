"""Error metrics that skip missing targets (speed 0), as in DCRNN and Graph WaveNet."""

from __future__ import annotations

import numpy as np


def masked_errors(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    mask = true > 0
    err = pred[mask].astype(np.float64) - true[mask]
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt(np.square(err).mean())),
        "mape": float((np.abs(err) / true[mask]).mean() * 100),
        "n": int(mask.sum()),
    }


def errors_by_horizon(
    pred: np.ndarray, true: np.ndarray, horizons: tuple[int, ...]
) -> list[dict[str, float]]:
    """pred and true are (S, H, N); pred holds only `horizons`, true holds all 12 steps."""
    rows = []
    for i, h in enumerate(horizons):
        rows.append(
            {
                "horizon": h,
                "minutes": 5 * h,
                **masked_errors(pred[:, i], true[:, h - 1]),
            }
        )
    return rows
