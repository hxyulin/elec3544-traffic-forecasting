# Multi-hop and available-geometry pilot

AI-assisted exploratory experiments. Inspect the implementation and evidence before drawing conclusions.

Source base: `0272ea47390ef3ac7ee89252d33da923fe79e690`. New implementation and runner hashes are frozen in `outputs/experiments/multihop_geometry_pilot/protocol.json`. Historical outputs were not reused or modified.

The raw archive contains only `metr_la.h5`, `sensor_ids_la.txt`, `sensor_locations_la.csv` (index, sensor_id, latitude, longitude), and `distances_la.csv` (from, to, cost). The existing downloader verified the archive SHA256 before extraction. No lane counts, limits, junction labels, or reliably matched external road metadata were available. No such labels were invented. The experiment uses supplied graph orientation; it does not establish traffic flow direction.

`multihop.py` preserves the local identity model and persistence anchor. Each orientation has separate one-hop and strict two-hop channels. A two-hop support squares the normalized support, removes self and direct neighbours, and renormalizes nonempty rows. Each support aggregates reliability-weighted normalized speed, observed fraction, reliability-weighted log gap age, and reliability mass. Empty rows stay zero. All histories end at the forecast origin.

Eight static geometry channels describe weighted east/north displacement, straight distance, and listed road distance in kilometres for each orientation. The coordinate conversion is a local spherical approximation. Geometry uses only sensor positions and static graph records. This is a small geometry comparison, not a detailed road inventory or edge-conditioned graph network.

All four variants contain 82,713 parameters with identical initial trainable values. A fixed mask disables all spatial slots, the two-hop and geometry slots, or only geometry slots for the local, one-hop, and two-hop controls. The geometry variant uses every slot. Disabled slots have zero input gradients. Geometry is repeated across the 12 history positions for this deliberately small extension; a separate static encoder remains untested.

The frozen pilot uses seed 42, three epochs, 100,000 sampled training rows per epoch, batch 2048, and the unchanged shared AdamW/masked-MAE training policy. Sample hashes match across all variants for each epoch. Training targets end before validation targets begin. Validation histories may contain earlier observed training-period speeds. Raw zero targets remain missing and are masked in loss and metrics. Normalization and initial missing-value fallback use only the purged training period. The runner discards the original test split and has no test evaluation path.

All four fits completed. Each selected epoch 3. See `summary.csv` and per-variant `selection.json` for horizon metrics. Mean validation MAE was 3.584731 mph (local), 3.552808 (one-hop), 3.532190 (two-hop), and 3.545329 (geometry). Two-hop improved by 0.020619 mph over one-hop at this pilot budget. Geometry increased MAE by 0.013139 mph relative to two-hop. These differences are not seed-replicated or independently evaluated. Every best checkpoint occurred at the budget limit, so convergence is not established. No full-budget study was run. The existing validation period has already informed earlier experiments. Historical baseline scores are confounded by the encoder size, purging, dependency versions, hardware, and training budget.

Execution used an isolated Linux workstation directory, four CPU threads and an RTX 5090 (32,607 MiB); the host reported 123 GiB RAM and 32 CPU threads. Python 3.12.14, PyTorch 2.15.0+cu132, CUDA 13.2, NumPy 2.5.4 and the complete package freeze are recorded per variant. The NVIDIA driver reported 615.78.08. Per-epoch runtime and per-fit runtime are recorded; per-fit runtime includes training and final validation, but excludes downloads, environment setup and feature construction. No total setup wall time was measured. Checkpoints remain remote and are hashed in selection records; raw data and checkpoints are excluded from this handoff.

The persistent session `traffic-spatial-pilot-20261112` ran the four fits sequentially with a 300-second timeout per variant and 1,200-second overall timeout. The session exited after completion; `pilot.log` records `PILOT_COMPLETED` and `EXIT_STATUS=0`. The private remote directory is recorded in the conversation handoff rather than in tracked documentation. Existing remote files were preserved by creating a new study directory.

To reproduce in a fresh project environment with the supplied source and lockfile:

```sh
uv sync --extra neural --group dev
uv run --extra neural pytest -q
uv run --extra neural scripts/try_multihop.py --prepare
CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run --extra neural scripts/try_multihop.py --variant local
CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run --extra neural scripts/try_multihop.py --variant one
CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run --extra neural scripts/try_multihop.py --variant two
CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run --extra neural scripts/try_multihop.py --variant geometry
```

Existing variant directories deliberately block another fit. The protocol rejects changed source/data hashes. Preserve the pilot; use a separately frozen study for any larger budget or further seeds. The next useful decision for student review is whether to replicate the two-hop comparison at the existing 50-epoch/one-million-row budget before interpreting a gain.

Validation actually run: Ruff format and lint on the three new Python files passed locally. The full remote pytest suite passed: 50 tests in 3.21 seconds. New tests cover raw horizon alignment, missing masks, future-input invariance, incoming/outgoing orientation, strict two-hop exclusions, geometry orientation, empty rows, disabled-input gradients, equal initialization, and purged target boundaries. No external APIs or metadata matching were used.

The frozen source manifest also captured macOS `._` tar metadata entries on the remote copy. They are not imported Python source and are not included in this handoff. All executable Python source hashes were checked against the delivered files and matched. Environment package records redact only the editable project's private absolute installation path. A fresh reproduction generates its own protocol before fitting; the archived pilot protocol preserves the observed run unchanged.
