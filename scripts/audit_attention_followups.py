"""Recompute subgroup metrics and provenance without test-period evaluation."""

import hashlib
import importlib
import json

import numpy as np
import pandas as pd

from traffic.data import ROOT
from traffic.metrics import masked_errors

for study in ("spatial", "temporal"):
    runner = importlib.import_module("try_attention_" + study)
    ds = runner.dataset()
    out = runner.OUT
    protocol = json.loads((out / "protocol.json").read_text())
    for name, expected in protocol["source"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
    for name, expected in protocol["data"].items():
        assert (
            hashlib.sha256(
                (ROOT / "data/raw/extracted" / name).read_bytes()
            ).hexdigest()
            == expected
        )
    starts = ds.splits["val"]
    truth = ds.speed[starts[:, None] + 11 + np.array([3, 6, 12])]
    origin = ds.speed[starts + 11]
    rows = []
    exposure = []
    for seed in (42, 43, 44):
        for variant in runner.VARIANTS:
            directory = out / f"seed{seed}" / variant
            p = np.load(directory / "best_prediction.npy")
            result = json.loads((directory / "selection.json").read_text())
            assert (
                hashlib.sha256(p.tobytes()).hexdigest() == result["prediction_sha256"]
            )
            assert all(
                masked_errors(p[:, i], truth[:, i]) == result["horizons"][str(h)]
                for i, h in enumerate([3, 6, 12])
            )
            for i, h in enumerate([3, 6, 12]):
                for group, mask in [
                    ("low_speed", (truth[:, i] > 0) & (truth[:, i] < 20)),
                    (
                        "drop60",
                        (truth[:, 2] > 0) & (origin > 0) & (origin - truth[:, 2] >= 10),
                    ),
                ]:
                    mask = mask & (truth[:, i] > 0)
                    error = (p[:, i] - truth[:, i])[mask].astype(float)
                    rows.append(
                        dict(
                            seed=seed,
                            variant=variant,
                            horizon=h,
                            group=group,
                            count=len(error),
                            mae=float(abs(error).mean()),
                            rmse=float(np.sqrt((error**2).mean())),
                        )
                    )
            history = pd.read_csv(directory / "history.csv")
            exposure.append(
                dict(
                    seed=seed,
                    variant=variant,
                    epochs=len(history),
                    rows=len(history) * 1_000_000,
                    best_epoch=result["best_epoch"],
                    epoch_cap=len(history) == 50,
                )
            )
    pd.DataFrame(rows).to_csv(out / "diagnostics.csv", index=False)
    pd.DataFrame(exposure).to_csv(out / "exposure.csv", index=False)
    (out / "source_data_verification.json").write_text(
        json.dumps(
            dict(
                source_hashes_match=True,
                data_hashes_match=True,
                stored_prediction_metrics_exact=True,
                original_test_excluded=True,
            ),
            indent=2,
        )
        + "\n"
    )
