"""Train the baselines and classical models, then score them on validation and test.

Writes outputs/results/metrics.csv and test predictions to outputs/predictions/<model>.npz.

    uv run scripts/evaluate.py                    # all models
    uv run scripts/evaluate.py persistence ridge  # a subset, merged into metrics.csv
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from traffic.data import REPORT_HORIZONS, ROOT, load_metr_la
from traffic.metrics import errors_by_horizon
from traffic.models import HistoricalAverage, Persistence, gradient_boosting, ridge

MODELS = {
    "persistence": Persistence,
    "historical_average": HistoricalAverage,
    "ridge_lags": lambda: ridge("ridge_lags", ("lags",)),
    "ridge": lambda: ridge("ridge", ("lags", "calendar", "profile")),
    "gradient_boosting": gradient_boosting,
}
RESULTS = ROOT / "outputs" / "results" / "metrics.csv"
PREDICTIONS = ROOT / "outputs" / "predictions"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*", choices=list(MODELS), default=list(MODELS))
    args = parser.parse_args()

    ds = load_metr_la()
    horizons = REPORT_HORIZONS
    truth = {split: ds.windows(split)[1] for split in ("val", "test")}
    rows = []
    PREDICTIONS.mkdir(parents=True, exist_ok=True)

    for key in args.models:
        model = MODELS[key]()
        start = time.perf_counter()
        if isinstance(model, (Persistence, HistoricalAverage)):
            model.fit(ds)
        else:
            model.fit(ds, horizons)
        fit_seconds = time.perf_counter() - start
        for split, y in truth.items():
            pred = model.predict(ds, split, horizons)
            for r in errors_by_horizon(pred, y, horizons):
                rows.append({"model": key, "split": split, **r})
            if split == "test":
                np.savez_compressed(
                    PREDICTIONS / f"{key}.npz",
                    pred=pred,
                    horizons=np.array(horizons),
                    starts=ds.splits["test"],
                )
        print(f"{key}: fitted in {fit_seconds:.0f}s")

    new = pd.DataFrame(rows)
    if RESULTS.exists():
        old = pd.read_csv(RESULTS)
        new = pd.concat([old[~old["model"].isin(args.models)], new], ignore_index=True)
    order = {k: i for i, k in enumerate(MODELS)}
    new = new.sort_values(
        ["split", "horizon", "model"],
        key=lambda c: c.map(order) if c.name == "model" else c,
    )
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(RESULTS, index=False, float_format="%.4f")

    test = new[new["split"] == "test"]
    table = test.pivot(index="model", columns="minutes", values=["mae", "rmse", "mape"])
    table = table.reindex([k for k in MODELS if k in table.index])
    print("\nTest set, masked metrics (mph; MAPE in %)")
    print(table.round(2).to_string())


if __name__ == "__main__":
    main()
