"""Audit baseline compatibility, immutable artifacts, and all retained hashes."""

import json
from pathlib import Path

import pandas as pd
from try_teacher_schedule import OUT, sha

prior = Path(__import__("os").environ["PRIOR_STUDY"])
protocol = json.loads((OUT / "protocol.json").read_text())
old_protocol = json.loads(
    (prior / "outputs/experiments/dcrnn_reference/protocol.json").read_text()
)
report = {
    "source_matches": {
        k: v == protocol["source"][k]
        for k, v in old_protocol["source"].items()
        if k in protocol["source"]
    },
    "data_matches": protocol["arrays"] == old_protocol["arrays"],
    "initialization_matches": protocol["initialization"]
    == old_protocol["initialization"],
}
assert (
    all(report["source_matches"].values())
    and report["data_matches"]
    and report["initialization_matches"]
)
for seed in (42, 43, 44):
    for variant in ("enabled", "disabled"):
        a = pd.read_csv(
            prior
            / "outputs/experiments/dcrnn_reference"
            / f"seed{seed}"
            / variant
            / "history.csv"
        )
        b = pd.read_csv(OUT / "A" / f"seed{seed}" / variant / "history.csv")
        keys = (
            "loss",
            "val_mae",
            "sample_sha256",
            "sampling_before",
            "sampling_after",
            "observed_targets",
            "lr",
        )
        report[f"{seed}/{variant}"] = {
            k: list(a[k]) == list(b[k][: len(a)]) for k in keys
        }
        assert all(report[f"{seed}/{variant}"].values())
        old = json.loads(
            (
                prior
                / "outputs/experiments/dcrnn_reference"
                / f"seed{seed}"
                / variant
                / "result.json"
            ).read_text()
        )
        report[f"{seed}/{variant}"]["prior_selected_prediction_hash"] = old[
            "prediction_sha256"
        ]
manifest = {
    str(p.relative_to(OUT)): sha(p.read_bytes())
    for p in sorted(OUT.rglob("*"))
    if p.is_file() and p.name != "artifact_hashes.json"
}
(OUT / "baseline_compatibility.json").write_text(json.dumps(report, indent=2) + "\n")
(OUT / "artifact_hashes.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("All prior schedule A overlapping trajectories match exactly.")
