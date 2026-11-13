"""Recompute validation-only evidence for the frozen schedule factorial."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from try_teacher_schedule import OUT, dataset, sha

root = OUT
# Reuse the frozen audit, subgroup definitions, and plotting implementation.
source = Path(__file__).with_name("summarize_dcrnn.py").read_text()
source = source[: source.index("# Earlier fitted models")]
source = source.replace("from try_dcrnn import OUT, dataset, sha", "")
source = source.replace(
    "axes[-1].legend()",
    '[ax.axvline(30, color="gray", linestyle="--") for ax in axes]\naxes[-1].legend()',
)
for schedule in ("A", "B"):
    (root / schedule / "protocol.json").write_bytes(
        (root / "protocol.json").read_bytes()
    )
    exec(
        compile(source, "frozen_summary", "exec"),
        {"OUT": root / schedule, "dataset": dataset, "sha": sha},
    )
f = pd.concat(
    [pd.read_csv(root / s / "per_seed.csv").assign(schedule=s) for s in ("A", "B")]
)
f.to_csv(root / "per_seed.csv", index=False)
for name, keys, column, first, second in (
    ("schedule", ["variant", "seed", "horizon"], "schedule", "B", "A"),
    ("graph", ["schedule", "seed", "horizon"], "variant", "enabled", "disabled"),
):
    a = f[f[column] == first].set_index(keys)[["mae", "rmse"]]
    b = f[f[column] == second].set_index(keys)[["mae", "rmse"]]
    delta = a - b
    delta.to_csv(root / f"{name}_paired.csv")
    delta.groupby([k for k in keys if k != "seed"]).agg(
        ["mean", "std", lambda x: int((x < 0).sum()), lambda x: int((x > 0).sum())]
    ).to_csv(root / f"{name}_aggregate.csv")
phase_rows = []
for seed in (42, 43, 44):
    histories = []
    for schedule in ("A", "B"):
        for variant in ("enabled", "disabled"):
            d = root / schedule / f"seed{seed}" / variant
            h = pd.read_csv(d / "history.csv")
            assert len(h) == 50
            histories.append(h)
            if schedule == "B":
                assert (h.loc[h.epoch > 30, "probability_first"] == 0).all()
                assert h.loc[h.epoch > 30, "forced_transitions"].sum() == 0
            for phase in ("prefix", "post"):
                r = json.loads((d / f"{phase}.json").read_text())
                p = np.load(d / f"{phase}.npy")
                assert r["prediction_sha256"] == sha(p.tobytes())
                ds = dataset()
                truth = ds.speed[ds.splits["val"][:, None] + 12 + np.array([2, 5, 11])]
                for i in range(3):
                    e = (p[:, i] - truth[:, i])[truth[:, i] > 0].astype(float)
                    assert r["metrics"][i] == dict(
                        mae=float(abs(e).mean()), rmse=float(np.sqrt((e**2).mean()))
                    )
                phase_rows.append(
                    dict(
                        seed=seed,
                        schedule=schedule,
                        variant=variant,
                        phase=phase,
                        epoch=r["epoch"],
                        mae=np.mean([m["mae"] for m in r["metrics"]]),
                        rmse=np.mean([m["rmse"] for m in r["metrics"]]),
                    )
                )
    for h in histories[1:]:
        for key in (
            "sample_sha256",
            "sampling_before",
            "sampling_after",
            "observed_targets",
        ):
            assert list(h[key]) == list(histories[0][key])
pd.DataFrame(phase_rows).to_csv(root / "phase_selection.csv", index=False)
print(f[f.horizon == "macro"].to_string(index=False))

exposures = []
for schedule in ("A", "B"):
    for seed in (42, 43, 44):
        for variant in ("enabled", "disabled"):
            h = pd.read_csv(root / schedule / f"seed{seed}" / variant / "history.csv")
            for phase, mask in (
                ("all", h.epoch > 0),
                ("prefix", h.epoch <= 30),
                ("post", h.epoch > 30),
            ):
                v = h[mask]
                exposures.append(
                    dict(
                        schedule=schedule,
                        seed=seed,
                        variant=variant,
                        phase=phase,
                        probability_first=v.probability_first.iloc[0],
                        probability_last=v.probability_last.iloc[-1],
                        forced=int(v.forced_transitions.sum()),
                        eligible=int(v.eligible_transitions.sum()),
                        forced_observed=int(v.forced_observed.sum()),
                        observed_targets=int(v.observed_targets.sum()),
                        val_mae_first=v.val_mae.iloc[0],
                        val_mae_last=v.val_mae.iloc[-1],
                        val_mae_min=v.val_mae.min(),
                        seconds_last=v.seconds.iloc[-1],
                    )
                )
pd.DataFrame(exposures).to_csv(root / "exposure_and_phase.csv", index=False)
