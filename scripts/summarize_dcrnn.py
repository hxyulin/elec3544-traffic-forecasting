"""Audit completed reference fits and emit student-inspection tables."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from try_dcrnn import OUT, dataset, sha

rows, diagnostics, audits = [], [], []
ds = dataset()
truth = ds.speed[ds.splits["val"][:, None] + 12 + np.array([2, 5, 11])]
origin = ds.speed[ds.splits["val"] + 11]
protocol = json.loads((OUT / "protocol.json").read_text())
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
for seed, ax in zip((42, 43, 44), axes):
    histories = []
    for variant in ("enabled", "disabled"):
        d = OUT / f"seed{seed}" / variant
        r = json.loads((d / "result.json").read_text())
        assert r["completed"] and r["restored_exact"]
        assert sha((d / "model.pt").read_bytes()) == r["checkpoint_sha256"]
        pred = np.load(d / "prediction.npy")
        assert sha(pred.tobytes()) == r["prediction_sha256"]
        history = pd.read_csv(d / "history.csv")
        histories.append(history)
        rng = np.random.default_rng(seed)
        for actual in history.sample_sha256:
            selected = rng.choice(
                ds.splits["train"], protocol["windows_per_epoch"], replace=False
            )
            assert sha(selected.tobytes()) == actual
        for i, h in enumerate((3, 6, 12)):
            error = (pred[:, i] - truth[:, i])[truth[:, i] > 0].astype(float)
            metric = dict(
                mae=float(abs(error).mean()), rmse=float(np.sqrt((error**2).mean()))
            )
            assert metric == r["metrics"][i]
            rows.append(
                dict(
                    seed=seed,
                    variant=variant,
                    horizon=str(h),
                    **metric,
                    best_epoch=r["best_epoch"],
                    epochs=r["epochs"],
                )
            )
            masks = {"below_40_mph": (truth[:, i] > 0) & (truth[:, i] < 40)}
            if h == 12:
                masks["drop_at_least_10_mph"] = (
                    (truth[:, i] > 0) & (origin > 0) & (truth[:, i] - origin <= -10)
                )
            for name, mask in masks.items():
                e = (pred[:, i] - truth[:, i])[mask].astype(float)
                diagnostics.append(
                    dict(
                        seed=seed,
                        variant=variant,
                        horizon=h,
                        subgroup=name,
                        count=len(e),
                        mae=float(abs(e).mean()),
                        rmse=float(np.sqrt((e**2).mean())),
                    )
                )
        rows.append(
            dict(
                seed=seed,
                variant=variant,
                horizon="macro",
                **{
                    k: float(np.mean([m[k] for m in r["metrics"]]))
                    for k in ("mae", "rmse")
                },
                best_epoch=r["best_epoch"],
                epochs=r["epochs"],
            )
        )
        audits.append(dict(seed=seed, variant=variant, **r))
        ax.plot(history.epoch, history.val_mae, label=variant)
    n = min(map(len, histories))
    for key in ("sample_sha256", "sampling_before", "sampling_after"):
        assert list(histories[0][key][:n]) == list(histories[1][key][:n])
    ax.set(title=f"Seed {seed}", xlabel="Epoch", ylabel="Validation MAE (mph)")
axes[-1].legend()
fig.tight_layout()
fig.savefig(OUT / "curves.png", dpi=160)
f = pd.DataFrame(rows)
f.to_csv(OUT / "per_seed.csv", index=False)
f.groupby(["variant", "horizon"])[["mae", "rmse"]].agg(["mean", "std"]).to_csv(
    OUT / "aggregate.csv"
)
a = f[f.variant == "enabled"].set_index(["seed", "horizon"])
b = f[f.variant == "disabled"].set_index(["seed", "horizon"])
delta = a[["mae", "rmse"]] - b[["mae", "rmse"]]
delta.to_csv(OUT / "paired_deltas.csv")
delta.groupby("horizon").agg(
    ["mean", "std", lambda x: int((x < 0).sum()), lambda x: int((x > 0).sum())]
).to_csv(OUT / "paired_aggregate.csv")
pd.DataFrame(diagnostics).to_csv(OUT / "diagnostics.csv", index=False)
(OUT / "verification.json").write_text(
    json.dumps(
        dict(
            all_six_completed=True,
            all_restorations_exact=True,
            all_metrics_recomputed=True,
            all_window_hashes_verified=True,
            paired_sampling_rng_equal=True,
            runs=audits,
        ),
        indent=2,
    )
    + "\n"
)
print(f[f.horizon == "macro"].to_string(index=False))

# Earlier fitted models are context only, never an isolated recurrence contrast.
prior = OUT.parent / "capacity_budget"
if (prior / "protocol.json").exists():
    old = json.loads((prior / "protocol.json").read_text())
    for key in ("speed", "adjacency", "train", "val"):
        assert old["arrays"][key] == protocol["arrays"][key]
    old_rows = pd.read_csv(prior / "per_seed.csv")
    old_rows = old_rows[old_rows.budget == 50].copy()
    old_rows["family"] = "MLP, contextual comparison only"
    old_rows["variant"] = old_rows["size"] + "/" + old_rows["hop"]
    new_rows = f.copy()
    new_rows["family"] = "DCGRU encoder-decoder"
    pd.concat([new_rows, old_rows], ignore_index=True).to_csv(
        OUT / "contextual_comparison.csv", index=False
    )
    (OUT / "same_evaluation_targets.json").write_text(
        json.dumps(
            dict(
                speed_train_val_graph_hashes_match=True,
                confounds="features, row versus full-network sampling, 3 versus 12 target loss, batch size, initialization, compute and observed-target exposure",
            ),
            indent=2,
        )
        + "\n"
    )
