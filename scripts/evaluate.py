"""Train the baselines and classical models, then score them on validation and test.

Writes outputs/results/metrics.csv and test predictions to outputs/predictions/<model>.npz.

    uv run scripts/evaluate.py                    # all models, with tuned settings if present
    uv run scripts/evaluate.py persistence ridge  # a subset, merged into metrics.csv
    uv run scripts/evaluate.py --tune             # pick settings on validation MAE
    uv run scripts/evaluate.py --ablation         # feature-group ablation, outputs/results/ablation.csv

--tune writes outputs/results/tuning.csv (every setting tried) and best_params.json.
The test set is scored only for the chosen settings.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time

import numpy as np
import pandas as pd

from traffic.data import REPORT_HORIZONS, ROOT, load_metr_la
from traffic.metrics import errors_by_horizon
from traffic.models import (
    RIDGE_ALPHA,
    HistoricalAverage,
    Persistence,
    gradient_boosting,
    ridge,
)

RESULTS = ROOT / "outputs" / "results" / "metrics.csv"
TUNING = ROOT / "outputs" / "results" / "tuning.csv"
BEST = ROOT / "outputs" / "results" / "best_params.json"
ABLATION = ROOT / "outputs" / "results" / "ablation.csv"
PREDICTIONS = ROOT / "outputs" / "predictions"

# Each feature set adds one group to the one before. Gradient boosting also gets the
# sensor id in every set, so the sets differ only in the groups being tested.
FEATURE_SETS = {
    "lags": ("lags",),
    "+calendar": ("lags", "calendar"),
    "+profile": ("lags", "calendar", "profile"),
    "+neighbours": ("lags", "calendar", "profile", "neighbours"),
}
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
GB_GRID = {
    "max_leaf_nodes": (31, 63, 127),
    "learning_rate": (0.05, 0.1),
    "min_samples_leaf": (100, 400),
}


def load_best() -> dict:
    return json.loads(BEST.read_text()) if BEST.exists() else {}


def build_models(best: dict) -> dict:
    alphas = best.get("ridge_alpha", {})
    gb = best.get("gb_params")
    a = lambda s: alphas.get(s, RIDGE_ALPHA)
    return {
        "persistence": Persistence,
        "historical_average": HistoricalAverage,
        "ridge_lags": lambda: ridge("ridge_lags", FEATURE_SETS["lags"], a("lags")),
        "ridge": lambda: ridge("ridge", FEATURE_SETS["+profile"], a("+profile")),
        "ridge_neighbours": lambda: ridge(
            "ridge_neighbours", FEATURE_SETS["+neighbours"], a("+neighbours")
        ),
        "gradient_boosting": lambda: gradient_boosting(params=gb),
        "gradient_boosting_neighbours": lambda: gradient_boosting(
            "gradient_boosting_neighbours",
            FEATURE_SETS["+neighbours"] + ("sensor",),
            gb,
        ),
    }


def fit(model, ds, horizons) -> float:
    start = time.perf_counter()
    if isinstance(model, (Persistence, HistoricalAverage)):
        model.fit(ds)
    else:
        model.fit(ds, horizons)
    return time.perf_counter() - start


def score(model, ds, split: str, y: np.ndarray, horizons) -> list[dict]:
    return errors_by_horizon(model.predict(ds, split, horizons), y, horizons)


def mean_val_mae(model, ds, y_val, horizons) -> float:
    return float(np.mean([r["mae"] for r in score(model, ds, "val", y_val, horizons)]))


def tune(ds, horizons) -> None:
    """Choose settings by mean validation MAE over the three horizons."""
    y_val = ds.windows("val")[1]
    rows, best = [], {"ridge_alpha": {}}
    for name, groups in FEATURE_SETS.items():
        scores = {}
        for alpha in RIDGE_ALPHAS:
            m = ridge(f"ridge_{name}", groups, alpha)
            fit(m, ds, horizons)
            scores[alpha] = mean_val_mae(m, ds, y_val, horizons)
            rows.append(
                {
                    "model": f"ridge {name}",
                    "setting": f"alpha={alpha}",
                    "val_mae": scores[alpha],
                }
            )
            print(f"ridge {name} alpha={alpha}: val MAE {scores[alpha]:.4f}")
        best["ridge_alpha"][name] = min(scores, key=scores.get)

    keys = list(GB_GRID)
    gb_scores = {}
    for values in itertools.product(*GB_GRID.values()):
        params = dict(zip(keys, values))
        m = gradient_boosting(params=params)
        secs = fit(m, ds, horizons)
        gb_scores[values] = mean_val_mae(m, ds, y_val, horizons)
        rows.append(
            {
                "model": "gradient_boosting",
                "setting": json.dumps(params),
                "val_mae": gb_scores[values],
            }
        )
        print(f"gb {params}: val MAE {gb_scores[values]:.4f} ({secs:.0f}s)")
    best["gb_params"] = dict(zip(keys, min(gb_scores, key=gb_scores.get)))

    TUNING.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(TUNING, index=False, float_format="%.4f")
    BEST.write_text(json.dumps(best, indent=2))
    print("\nChosen on validation:", json.dumps(best))


def ablation(ds, horizons, best: dict) -> None:
    """Add one feature group at a time, plus the original leaky profile for comparison."""
    truth = {s: ds.windows(s)[1] for s in ("val", "test")}
    alphas = best.get("ridge_alpha", {})
    gb = best.get("gb_params")
    runs = []
    for name, groups in FEATURE_SETS.items():
        runs.append(
            ("ridge", name, ridge("ridge", groups, alphas.get(name, RIDGE_ALPHA)))
        )
        runs.append(
            (
                "gradient_boosting",
                name,
                gradient_boosting(groups=groups + ("sensor",), params=gb),
            )
        )
    leaky = FEATURE_SETS["+profile"]
    runs.append(
        (
            "ridge",
            "+profile (leaky)",
            ridge("ridge", leaky, alphas.get("+profile", RIDGE_ALPHA), fix_leak=False),
        )
    )
    runs.append(
        (
            "gradient_boosting",
            "+profile (leaky)",
            gradient_boosting(groups=leaky + ("sensor",), params=gb, fix_leak=False),
        )
    )

    rows = []
    for model_name, setname, m in runs:
        secs = fit(m, ds, horizons)
        for split, y in truth.items():
            for r in score(m, ds, split, y, horizons):
                rows.append(
                    {"model": model_name, "feature_set": setname, "split": split, **r}
                )
        print(f"{model_name} {setname}: {secs:.0f}s")
    out = pd.DataFrame(rows)
    ABLATION.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(ABLATION, index=False, float_format="%.4f")
    test = out[out["split"] == "test"]
    print("\nAblation, test set (mph)")
    print(
        test.pivot_table(
            index=["model", "feature_set"],
            columns="minutes",
            values=["mae", "rmse"],
            sort=False,
        )
        .round(2)
        .to_string()
    )


def main() -> None:
    best = load_best()
    models = build_models(best)
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*", choices=list(models), default=list(models))
    parser.add_argument(
        "--tune", action="store_true", help="choose settings on validation MAE"
    )
    parser.add_argument(
        "--ablation", action="store_true", help="feature-group ablation table"
    )
    args = parser.parse_args()

    ds = load_metr_la()
    horizons = REPORT_HORIZONS
    if args.tune:
        tune(ds, horizons)
        return
    if args.ablation:
        ablation(ds, horizons, best)
        return

    truth = {split: ds.windows(split)[1] for split in ("val", "test")}
    rows = []
    PREDICTIONS.mkdir(parents=True, exist_ok=True)

    for key in args.models:
        model = models[key]()
        fit_seconds = fit(model, ds, horizons)
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
    order = {k: i for i, k in enumerate(models)}
    new = new.sort_values(
        ["split", "horizon", "model"],
        key=lambda c: c.map(order) if c.name == "model" else c,
    )
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(RESULTS, index=False, float_format="%.4f")

    test = new[new["split"] == "test"]
    table = test.pivot(index="model", columns="minutes", values=["mae", "rmse", "mape"])
    table = table.reindex([k for k in models if k in table.index])
    print("\nTest set, masked metrics (mph; MAPE in %)")
    print(table.round(2).to_string())


if __name__ == "__main__":
    main()
