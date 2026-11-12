"""Validation diagnostics, matched contrasts and sample audit; no model selection."""

import hashlib
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from try_capacity import OUT, dataset

from traffic.data import IN_STEPS, REPORT_HORIZONS

rows, diagnostics, histories = [], [], {}
ds = dataset()
truth = ds.speed[ds.splits["val"][:, None] + IN_STEPS + np.array(REPORT_HORIZONS) - 1]
origin = ds.speed[ds.splits["val"] + IN_STEPS - 1]
for seed in (42, 43, 44):
    for size in ("current", "wide"):
        for hop in ("one", "two"):
            directory = OUT / f"seed{seed}" / size / hop
            history = pd.read_csv(directory / "history.csv")
            histories[seed, size, hop] = history
            rng = np.random.default_rng(seed)
            for actual in history.sample_sha256:
                selected = rng.choice(
                    len(ds.splits["train"]) * ds.n_sensors, 1_000_000, replace=False
                )
                assert hashlib.sha256(selected.tobytes()).hexdigest() == actual
            for budget, filename, predfile in [
                (50, "prefix_selection.json", "prefix_prediction.npy"),
                (100, "selection.json", "best_prediction.npy"),
            ]:
                selection = json.loads((directory / filename).read_text())
                assert selection["restored_prediction_exact"]
                assert np.isclose(
                    selection["val_mae"],
                    history.loc[history.epoch <= budget, "val_mae"].min(),
                    atol=1e-14,
                    rtol=0,
                )
                for horizon in ("3", "6", "12", "macro"):
                    metrics = selection["horizons"]
                    values = (
                        metrics[horizon]
                        if horizon != "macro"
                        else {
                            k: np.mean([v[k] for v in metrics.values()])
                            for k in ("mae", "rmse")
                        }
                    )
                    rows.append(
                        dict(
                            seed=seed,
                            size=size,
                            hop=hop,
                            budget=budget,
                            horizon=horizon,
                            **values,
                            best_epoch=selection["best_epoch"],
                            epochs=len(history),
                            cap_reached=len(history) == 100,
                        )
                    )
                prediction = np.load(directory / predfile)
                for i, h in enumerate(REPORT_HORIZONS):
                    masks = {"below_40_mph": (truth[:, i] > 0) & (truth[:, i] < 40)}
                    if h == 12:
                        masks["drop_at_least_10_mph"] = (
                            (truth[:, i] > 0)
                            & (origin > 0)
                            & (truth[:, i] - origin <= -10)
                        )
                    for subgroup, mask in masks.items():
                        error = (prediction[:, i] - truth[:, i])[mask].astype(float)
                        diagnostics.append(
                            dict(
                                seed=seed,
                                size=size,
                                hop=hop,
                                budget=budget,
                                horizon=h,
                                subgroup=subgroup,
                                count=len(error),
                                mae=np.abs(error).mean(),
                                rmse=np.sqrt(np.square(error).mean()),
                            )
                        )
frame = pd.DataFrame(rows)
frame.to_csv(OUT / "per_seed.csv", index=False)
frame.groupby(["size", "hop", "budget", "horizon"])[["mae", "rmse"]].agg(
    ["mean", "std"]
).to_csv(OUT / "aggregate.csv")
pd.DataFrame(diagnostics).to_csv(OUT / "diagnostics.csv", index=False)
contrasts = []
for seed in (42, 43, 44):
    for budget in (50, 100):
        for h in ("3", "6", "12", "macro"):
            f = frame[
                (frame.seed == seed) & (frame.budget == budget) & (frame.horizon == h)
            ].set_index(["size", "hop"])
            for label, a, b in [
                ("capacity_one", ("wide", "one"), ("current", "one")),
                ("capacity_two", ("wide", "two"), ("current", "two")),
                ("two_hop_current", ("current", "two"), ("current", "one")),
                ("two_hop_wide", ("wide", "two"), ("wide", "one")),
            ]:
                contrasts.append(
                    dict(
                        seed=seed,
                        budget=budget,
                        horizon=h,
                        comparison=label,
                        **{k: f.loc[a, k] - f.loc[b, k] for k in ("mae", "rmse")},
                    )
                )
pd.DataFrame(contrasts).to_csv(OUT / "paired_deltas.csv", index=False)
pd.DataFrame(contrasts).groupby(["comparison", "budget", "horizon"])[
    ["mae", "rmse"]
].agg(["mean", "std"]).to_csv(OUT / "paired_aggregate.csv")
a = frame[frame.budget == 100].set_index(["seed", "size", "hop", "horizon"])
b = frame[frame.budget == 50].set_index(["seed", "size", "hop", "horizon"])
(a[["mae", "rmse"]] - b[["mae", "rmse"]]).to_csv(OUT / "budget_deltas.csv")
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, seed in zip(axes, (42, 43, 44)):
    for size in ("current", "wide"):
        for hop in ("one", "two"):
            history = histories[seed, size, hop]
            ax.plot(history.epoch, history.val_mae, label=f"{size}/{hop}")
    ax.axvline(50, color="gray", linestyle=":")
    ax.set(title=f"Seed {seed}", xlabel="Epoch", ylabel="Validation macro MAE (mph)")
axes[-1].legend()
fig.tight_layout()
fig.savefig(OUT / "curves.png", dpi=160)
(OUT / "verification.json").write_text(
    json.dumps(
        dict(
            all_12_completed=True,
            every_observed_sample_verified=True,
            all_24_restored_predictions_exact=True,
        ),
        indent=2,
    )
    + "\n"
)
print(frame[(frame.horizon == "macro")].to_string(index=False))
