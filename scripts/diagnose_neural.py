"""Validation residual partitions and large-error contributions, with no test access.

--restore-boosting refits the existing selected reference to recover validation
predictions. It checks them against recorded validation metrics before saving.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd

from traffic.analysis import (
    forecast_partitions,
    load_predictions,
    residual_summary,
    verify_partition,
)
from traffic.data import IN_STEPS, REPORT_HORIZONS, ROOT, load_metr_la
from traffic.identity import gap_age
from traffic.metrics import masked_errors
from traffic.models import gradient_boosting

OUT = ROOT / "outputs/experiments/neural_followup"
PRED = ROOT / "outputs/predictions"


def restore_boosting(ds, truth, starts, hs):
    params = json.loads((ROOT / "outputs/results/best_params.json").read_text())[
        "gb_params"
    ]
    model = gradient_boosting(
        "gradient_boosting_neighbours",
        ("lags", "calendar", "profile", "neighbours", "sensor"),
        params,
    )
    model.fit(ds, REPORT_HORIZONS)
    pred = model.predict(ds, "val", REPORT_HORIZONS)
    reference = pd.read_csv(ROOT / "outputs/results/metrics.csv")
    reference = reference[
        (reference.model == "gradient_boosting_neighbours") & (reference.split == "val")
    ].set_index("minutes")
    for i, h in enumerate(hs):
        measured = masked_errors(pred[:, i], truth[:, i])
        for metric in ("mae", "rmse", "mape"):
            np.testing.assert_allclose(
                measured[metric],
                reference.loc[int(h * 5), metric],
                rtol=0,
                atol=0.000051,
            )
    np.savez_compressed(
        PRED / "followup_boosting_val.npz", pred=pred, starts=starts, horizons=hs
    )
    (OUT / "boosting_restoration.json").write_text(
        json.dumps(
            dict(
                scope="train fit and validation prediction only; selected settings unchanged",
                data_sha256=hashlib.sha256(ds.speed.tobytes()).hexdigest(),
                params=params,
                checks="all 9 validation MAE/RMSE/MAPE values match recorded rounded values within 0.000051",
                thread_limits={
                    k: os.environ.get(k)
                    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")
                },
            ),
            indent=2,
        )
        + "\n"
    )
    print("Restored neighbour boosting: all nine validation metrics match.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restore-boosting", action="store_true")
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()
    ds = load_metr_la()
    OUT.mkdir(parents=True, exist_ok=True)
    starts, hs = ds.splits["val"], np.array(REPORT_HORIZONS)
    origins = starts + IN_STEPS - 1
    truth = ds.speed[origins[:, None] + hs[None]]
    if args.restore_boosting:
        restore_boosting(ds, truth, starts, hs)
        return
    paths = {"identity": PRED / "mlp_identity_seed42_val.npz"}
    if (PRED / "followup_boosting_val.npz").exists():
        paths["boosting"] = PRED / "followup_boosting_val.npz"
    if not args.baseline_only:
        for variant in ("huber", "mixed", "spatial_control", "spatial"):
            paths[variant] = PRED / f"followup_{variant}_seed42_val.npz"
    predictions = {
        k: load_predictions(p, starts, hs, truth.shape) for k, p in paths.items()
    }
    predictions["persistence"] = np.repeat(ds.filled[origins, None], 3, axis=1)
    missing = ds.missing[starts[:, None] + np.arange(IN_STEPS)].sum(1)
    age = gap_age(ds.speed)[origins]
    rows, aggregate, tails, ranges, checked = [], [], [], [], 0
    train_observed = ds.speed[: ds.train_end]
    train_observed = train_observed[train_observed > 0]
    lower, upper = float(train_observed.min()), float(train_observed.max())
    for model, pred in predictions.items():
        for i, h in enumerate(hs):
            p, y = pred[:, i], truth[:, i]
            full = residual_summary(p, y)
            aggregate.append(dict(model=model, minutes=int(h * 5), **full))
            error = p[y > 0].astype(np.float64) - y[y > 0]
            observed_pred = p[y > 0]
            outside = (observed_pred < lower) | (observed_pred > upper)
            ranges.append(
                dict(
                    model=model,
                    minutes=int(h * 5),
                    train_min=lower,
                    train_max=upper,
                    pred_min=float(observed_pred.min()),
                    pred_max=float(observed_pred.max()),
                    outside_n=int(outside.sum()),
                    outside_fraction=float(outside.mean()),
                    outside_sse_share=float(
                        np.square(error[outside]).sum() / full["sse"]
                    ),
                )
            )
            for fraction in (0.01, 0.05):
                # Exactly ceil(fraction*n) largest absolute errors; ties use row order.
                count = int(np.ceil(len(error) * fraction))
                indices = np.argsort(-np.abs(error), kind="stable")[:count]
                tails.append(
                    dict(
                        model=model,
                        minutes=int(h * 5),
                        fraction=fraction,
                        n=count,
                        sse_share=float(np.square(error[indices]).sum() / full["sse"]),
                    )
                )
            for name, (labels, names) in forecast_partitions(
                ds.speed[origins], y, missing, age
            ).items():
                labels = np.broadcast_to(labels, y.shape)
                group_rows = []
                for j, label in enumerate(names):
                    mask = labels == j
                    result = residual_summary(p[mask], y[mask])
                    group_rows.append(result)
                    rows.append(
                        dict(
                            model=model,
                            minutes=int(h * 5),
                            partition=name,
                            group=label,
                            count_share=result["n"] / full["n"],
                            sse_share=result["sse"] / full["sse"],
                            **result,
                        )
                    )
                verify_partition(group_rows, full)
                np.testing.assert_allclose(
                    sum(r["sse"] for r in group_rows), full["sse"], rtol=1e-12
                )
                checked += 1
    out = OUT / ("diagnosis_baseline" if args.baseline_only else "diagnosis")
    out.mkdir(exist_ok=True)
    pd.DataFrame(aggregate).to_csv(out / "aggregate.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "partitions.csv", index=False)
    pd.DataFrame(tails).to_csv(out / "tails.csv", index=False)
    pd.DataFrame(ranges).to_csv(out / "prediction_ranges.csv", index=False)
    (out / "verification.json").write_text(
        json.dumps(
            dict(
                scope="validation only",
                partition_checks=checked,
                prediction_sha256={
                    k: hashlib.sha256(p.read_bytes()).hexdigest()
                    for k, p in paths.items()
                },
                note="transition groups use observed future targets for diagnosis only; counts are dependent forecast-target pairs",
            ),
            indent=2,
        )
        + "\n"
    )
    print(
        pd.DataFrame(aggregate)[
            ["model", "minutes", "mae", "rmse", "bias", "p99_abs"]
        ].to_string(index=False)
    )
    print(f"Verified {checked} partitions.")


if __name__ == "__main__":
    main()
