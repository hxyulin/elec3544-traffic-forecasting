"""Summarize AI-assisted, validation-only replication for student inspection."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from traffic.data import ROOT

out = ROOT / "outputs/experiments/multihop_geometry_replication"
variants = ("local", "one", "two", "geometry")
rows, histories = [], {}
for seed in (42, 43, 44):
    for variant in variants:
        directory = out / f"seed{seed}" / variant
        selection = json.loads((directory / "selection.json").read_text())
        history = pd.read_csv(directory / "history.csv")
        histories[seed, variant] = history
        assert selection["restored_prediction_exact"]
        assert history.val_mae.min() == selection["val_mae"] or np.isclose(
            history.val_mae.min(), selection["val_mae"], atol=1e-14, rtol=0
        )
        metrics = selection["horizons"]
        for horizon in ("3", "6", "12", "macro"):
            values = (
                metrics[horizon]
                if horizon != "macro"
                else {
                    metric: np.mean([m[metric] for m in metrics.values()])
                    for metric in ("mae", "rmse")
                }
            )
            rows.append(
                dict(
                    seed=seed,
                    variant=variant,
                    horizon=horizon,
                    mae=values["mae"],
                    rmse=values["rmse"],
                    best_epoch=selection["best_epoch"],
                    epochs=len(history),
                    epoch_cap=len(history) == 50,
                )
            )
    for epoch in range(min(len(histories[seed, v]) for v in variants)):
        assert (
            len({histories[seed, v].iloc[epoch].sample_sha256 for v in variants}) == 1
        )
    # Also verify every observed epoch against the seeded sampling policy.
    protocol = json.loads((out / "protocol.json").read_text())
    starts_count = protocol["splits"]["train"][2]
    # METR-LA sensor count is recorded in the frozen speed array's loader contract.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "replication", ROOT / "scripts/try_multihop_replication.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ds = module.dataset()
    rng = np.random.default_rng(seed)
    import hashlib

    for epoch in range(max(len(histories[seed, v]) for v in variants)):
        selected = rng.choice(starts_count * ds.n_sensors, 1_000_000, replace=False)
        expected = hashlib.sha256(selected.tobytes()).hexdigest()
        for variant in variants:
            history = histories[seed, variant]
            if epoch < len(history):
                assert history.iloc[epoch].sample_sha256 == expected
frame = pd.DataFrame(rows)
frame.to_csv(out / "per_seed.csv", index=False)
frame.groupby(["variant", "horizon"])[["mae", "rmse"]].agg(["mean", "std"]).to_csv(
    out / "aggregate.csv"
)
deltas = []
for candidate, baseline in (("two", "one"), ("geometry", "two")):
    for seed in (42, 43, 44):
        a = frame[(frame.seed == seed) & (frame.variant == candidate)].set_index(
            "horizon"
        )
        b = frame[(frame.seed == seed) & (frame.variant == baseline)].set_index(
            "horizon"
        )
        for h in a.index:
            deltas.append(
                dict(
                    seed=seed,
                    comparison=f"{candidate}-{baseline}",
                    horizon=h,
                    mae=a.loc[h, "mae"] - b.loc[h, "mae"],
                    rmse=a.loc[h, "rmse"] - b.loc[h, "rmse"],
                )
            )
delta = pd.DataFrame(deltas)
delta.to_csv(out / "paired_deltas.csv", index=False)
delta.groupby(["comparison", "horizon"])[["mae", "rmse"]].agg(
    ["mean", "std", lambda x: int((x < 0).sum())]
).to_csv(out / "paired_aggregate.csv")
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, seed in zip(axes, (42, 43, 44)):
    for variant in variants:
        history = histories[seed, variant]
        ax.plot(history.epoch, history.val_mae, label=variant)
        best = history.loc[history.val_mae.idxmin()]
        ax.scatter(best.epoch, best.val_mae, s=15)
    ax.set(title=f"Seed {seed}", xlabel="Epoch", ylabel="Validation macro MAE (mph)")
axes[-1].legend()
fig.tight_layout()
fig.savefig(out / "curves.png", dpi=160)
(out / "verification.json").write_text(
    json.dumps(
        dict(
            all_12_completed=True,
            sample_hash_matches=True,
            restored_predictions_exact=True,
            assistance="AI-assisted exploration for student review",
        ),
        indent=2,
    )
    + "\n"
)
print(frame[frame.horizon == "macro"].to_string(index=False))
print(delta[delta.horizon == "macro"].to_string(index=False))
