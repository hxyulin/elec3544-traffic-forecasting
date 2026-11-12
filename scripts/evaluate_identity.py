"""Score only the validation-selected MLP candidate; keep official metrics intact.

    uv run --extra neural scripts/evaluate_identity.py --seed 42

Run after both validation trials finish. This is an exploratory comparison on the
existing test period, which was examined before this investigation began.
"""

from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import pandas as pd
import torch

from traffic.data import IN_STEPS, REPORT_HORIZONS, ROOT, load_metr_la
from traffic.identity import IdentityData, IdentityMLP, predict_identity
from traffic.metrics import masked_errors


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    out = ROOT / "outputs/experiments/identity_mlp" / f"seed_{args.seed}"
    selections = pd.read_csv(out / "selection.csv")
    if set(selections.variant) != {"shared", "identity"} or len(selections) != 2:
        raise ValueError(
            "Complete both predeclared validation trials before test evaluation"
        )
    winner = selections.sort_values(["val_mae", "variant"]).iloc[0]
    variant = winner.variant
    path = out / variant / "model.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    contract = json.loads((out / "contract.json").read_text())
    ds = load_metr_la()
    if (
        contract["data_sha256"] != hashlib.sha256(ds.speed.tobytes()).hexdigest()
        or contract["sensor_ids"] != ds.sensor_ids
        or contract["horizons"] != list(REPORT_HORIZONS)
    ):
        raise ValueError("Experiment data, sensor order or horizons differ")
    if checkpoint["variant"] != variant or checkpoint["settings"]["seed"] != args.seed:
        raise ValueError("Checkpoint variant or seed differs")
    data = IdentityData(ds, args.device)
    np.testing.assert_allclose(
        [data.mean, data.std], contract["normalizer"], rtol=0, atol=1e-7
    )
    np.testing.assert_allclose(
        checkpoint["val_mae"], winner.val_mae, rtol=0, atol=1e-10
    )
    model = IdentityMLP(ds.n_sensors, use_identity=variant == "identity").to(
        args.device
    )
    model.load_state_dict(checkpoint["state_dict"])
    # Verify the restored model before accessing test targets for scoring.
    hs = np.array(REPORT_HORIZONS)
    val_starts = ds.splits["val"]
    val_true = ds.speed[val_starts[:, None] + IN_STEPS + hs[None, :] - 1]
    val_pred = predict_identity(model, data, "val")
    val_score = float(
        np.mean(
            [
                masked_errors(val_pred[:, i], val_true[:, i])["mae"]
                for i in range(len(hs))
            ]
        )
    )
    np.testing.assert_allclose(val_score, winner.val_mae, rtol=0, atol=1e-6)
    pred = predict_identity(model, data, "test")
    if not np.isfinite(pred).all():
        raise ValueError("Non-finite test predictions")
    starts = ds.splits["test"]
    true = ds.speed[starts[:, None] + IN_STEPS + hs[None, :] - 1]
    missing = ds.missing[starts[:, None] + np.arange(IN_STEPS)].any(1)
    rows = []
    for i, h in enumerate(hs):
        y, p = true[:, i], pred[:, i]
        masks = {
            "all_observed": y > 0,
            "below_40_mph": (y > 0) & (y < 40),
            "at_least_40_mph": y >= 40,
            "complete_input": (y > 0) & ~missing,
            "missing_input": (y > 0) & missing,
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
    pd.DataFrame(rows).to_csv(out / "test.csv", index=False)
    metadata = {
        "selected_variant": variant,
        "selection_uses": "mean validation MAE only",
        "selection_val_mae": float(winner.val_mae),
        "best_epoch": int(winner.best_epoch),
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "data_sha256": contract["data_sha256"],
        "device": args.device,
        "test_policy": "Only validation winner scored; no further tuning based on this result. Existing test period was examined before this investigation; this is exploratory evidence.",
    }
    (out / "test_evaluation.json").write_text(json.dumps(metadata, indent=2) + "\n")
    np.savez_compressed(
        ROOT / f"outputs/predictions/mlp_{variant}_seed{args.seed}_test.npz",
        pred=pred,
        starts=starts,
        horizons=hs,
    )
    print(pd.DataFrame(rows).query("condition=='all_observed'").to_string(index=False))


if __name__ == "__main__":
    main()
