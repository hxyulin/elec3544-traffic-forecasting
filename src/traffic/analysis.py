"""Alignment checks and masked subgroup scores for saved forecasts."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_predictions(
    path: Path, starts: np.ndarray, horizons: np.ndarray, shape: tuple[int, ...]
) -> np.ndarray:
    with np.load(path) as saved:
        np.testing.assert_array_equal(saved["starts"], starts)
        np.testing.assert_array_equal(saved["horizons"], horizons)
        pred = saved["pred"]
    if pred.shape != shape or not np.isfinite(pred).all():
        raise ValueError(f"{path.stem}: predictions have invalid shape or values")
    return pred


def grouped_errors(
    pred: np.ndarray, true: np.ndarray, groups: np.ndarray, n_groups: int
) -> list[dict]:
    """Score a partition; group labels broadcast over window/sensor axes.

    Empty groups retain n=0 and NaN scores. Counts are observed forecast-target
    pairs, not independent samples. Accumulate errors in float64 like masked_errors.
    """
    if pred.shape != true.shape:
        raise ValueError("Prediction and target shapes differ")
    labels = np.broadcast_to(groups, true.shape)
    if (
        not np.issubdtype(labels.dtype, np.integer)
        or (labels < 0).any()
        or (labels >= n_groups).any()
    ):
        raise ValueError("Group labels must be integers in [0, n_groups)")
    mask = true > 0
    ids = labels[mask]
    error = pred[mask].astype(np.float64) - true[mask]
    n = np.bincount(ids, minlength=n_groups)
    values = []
    for weight in (np.abs(error), error**2, np.abs(error) / true[mask] * 100):
        total = np.bincount(ids, weights=weight, minlength=n_groups)
        values.append(np.divide(total, n, out=np.full(n_groups, np.nan), where=n > 0))
    mae, mse, mape = values
    return [
        {
            "group": i,
            "mae": float(mae[i]),
            "rmse": float(np.sqrt(mse[i])),
            "mape": float(mape[i]),
            "n": int(n[i]),
        }
        for i in range(n_groups)
    ]


def verify_partition(rows: list[dict], aggregate: dict) -> None:
    """Check that count-weighted subgroup scores reproduce the pooled scores."""
    observed = [r for r in rows if r["n"] > 0]
    n = sum(r["n"] for r in observed)
    if n != aggregate["n"]:
        raise ValueError("Subgroup counts do not partition observed targets")
    if not n:
        return
    recomposed = [
        sum(r["mae"] * r["n"] for r in observed) / n,
        np.sqrt(sum(r["rmse"] ** 2 * r["n"] for r in observed) / n),
        sum(r["mape"] * r["n"] for r in observed) / n,
    ]
    np.testing.assert_allclose(
        recomposed, [aggregate[k] for k in ("mae", "rmse", "mape")], rtol=0, atol=1e-9
    )
