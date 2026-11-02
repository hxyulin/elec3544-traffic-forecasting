"""METR-LA loading, sensor graph, chronological splits and sliding windows.

Conventions follow DCRNN (Li et al., 2018) so results compare with published numbers:
12 input steps (1 hour), 12 output steps, 70/10/20 chronological split over windows,
and a speed of 0 marks a missing reading.
"""

from __future__ import annotations

import hashlib
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
# Mirror used by the Torch Spatiotemporal library. Same content as the DCRNN Google Drive files.
RAW_URL = "https://drive.switch.ch/index.php/s/Z8cKHAVyiDqkzaG/download"
RAW_SHA256 = "cc249982e8ac42c8c0e21f2a1233a621eada95cee14131aaa3adce99e2efc30b"

IN_STEPS = 12
OUT_STEPS = 12
STEPS_PER_DAY = 288
STEPS_PER_WEEK = 7 * STEPS_PER_DAY
SPLIT_FRACTIONS = {"train": 0.7, "val": 0.1, "test": 0.2}
# Horizons reported in the proposal, as 1-based steps: 15, 30 and 60 minutes.
REPORT_HORIZONS = (3, 6, 12)


def download(raw_dir: Path = RAW_DIR) -> Path:
    """Fetch and unpack the raw files once. Returns the directory holding them."""
    out = raw_dir / "extracted"
    if (out / "metr_la.h5").exists():
        return out
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / "metr_la.zip"
    if not archive.exists():
        urllib.request.urlretrieve(RAW_URL, archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != RAW_SHA256:
        raise RuntimeError(f"{archive} has sha256 {digest}, expected {RAW_SHA256}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(out)
    return out


def build_adjacency(
    sensor_ids: list[str], distances: pd.DataFrame, threshold: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    """Directed road-distance graph as in DCRNN's gen_adj_mx.py.

    Weight is exp(-(d / std)^2), with std taken over all finite pairwise distances;
    weights below `threshold` are set to 0. Returns (adjacency, distance matrix in metres).
    """
    index = {sid: i for i, sid in enumerate(sensor_ids)}
    n = len(sensor_ids)
    dist = np.full((n, n), np.inf, dtype=np.float64)
    src = distances["from"].astype(str).map(index)
    dst = distances["to"].astype(str).map(index)
    keep = src.notna() & dst.notna()
    dist[src[keep].astype(int), dst[keep].astype(int)] = distances.loc[keep, "cost"]
    std = dist[np.isfinite(dist)].std()
    adj = np.exp(-np.square(dist / std))
    adj[adj < threshold] = 0.0
    return adj.astype(np.float32), dist


def split_starts(n_steps: int) -> dict[str, np.ndarray]:
    """Window start indices for each split, in time order.

    Window s uses steps [s, s+12) as input and [s+12, s+24) as target.
    """
    n_windows = n_steps - IN_STEPS - OUT_STEPS + 1
    n_test = round(n_windows * SPLIT_FRACTIONS["test"])
    n_train = round(n_windows * SPLIT_FRACTIONS["train"])
    n_val = n_windows - n_train - n_test
    starts = np.arange(n_windows)
    return {
        "train": starts[:n_train],
        "val": starts[n_train : n_train + n_val],
        "test": starts[n_train + n_val :],
    }


@dataclass
class MetrLA:
    timestamps: pd.DatetimeIndex
    sensor_ids: list[str]
    speed: np.ndarray  # (T, N) mph, 0 = missing. Use for targets and evaluation.
    filled: np.ndarray  # (T, N) missing values carried forward. Use for model inputs.
    adjacency: np.ndarray  # (N, N)
    distance: np.ndarray  # (N, N) metres, inf if no road path listed
    locations: pd.DataFrame  # sensor_id, latitude, longitude in sensor order
    splits: dict[str, np.ndarray]

    @property
    def missing(self) -> np.ndarray:
        return self.speed == 0

    @property
    def n_sensors(self) -> int:
        return self.speed.shape[1]

    @property
    def train_end(self) -> int:
        """First time step not covered by any training window (input or target)."""
        return int(self.splits["train"][-1]) + IN_STEPS + OUT_STEPS

    @property
    def time_of_week(self) -> np.ndarray:
        """5-minute slot within the week, 0 = Monday 00:00."""
        ts = self.timestamps
        return (
            ts.dayofweek * STEPS_PER_DAY + (ts.hour * 60 + ts.minute) // 5
        ).to_numpy()

    def normalizer(self) -> tuple[float, float]:
        """Mean and std of observed speeds in the training period, for z-scoring inputs."""
        train = self.speed[: self.train_end]
        observed = train[train > 0]
        return float(observed.mean()), float(observed.std())

    def windows(
        self, split: str, source: str = "filled"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Inputs (S, 12, N) from `source` and raw targets (S, 12, N) for one split."""
        starts = self.splits[split]
        x_src = getattr(self, source)
        x = np.lib.stride_tricks.sliding_window_view(x_src, IN_STEPS, axis=0)[starts]
        y = np.lib.stride_tricks.sliding_window_view(self.speed, OUT_STEPS, axis=0)[
            starts + IN_STEPS
        ]
        # sliding_window_view puts the window axis last: (S, N, 12) -> (S, 12, N).
        return x.transpose(0, 2, 1).copy(), y.transpose(0, 2, 1).copy()


def causal_fill(speed: np.ndarray, train_end: int) -> np.ndarray:
    """Carry the last observed value forward; uses only past values, so no leakage.

    Gaps at the very start of the series have no past value. They get the sensor's
    mean observed speed over the training period.
    """
    frame = pd.DataFrame(np.where(speed > 0, speed, np.nan))
    frame = frame.ffill()
    train = speed[:train_end]
    sensor_mean = np.nanmean(np.where(train > 0, train, np.nan), axis=0)
    frame = frame.fillna(pd.Series(sensor_mean))
    return frame.to_numpy(dtype=np.float32)


def load_metr_la() -> MetrLA:
    raw = download()
    speed_df = pd.read_hdf(raw / "metr_la.h5")
    sensor_ids = [str(c) for c in speed_df.columns]
    listed = (raw / "sensor_ids_la.txt").read_text().strip().split(",")
    if listed != sensor_ids:
        raise RuntimeError(
            "sensor order in metr_la.h5 does not match sensor_ids_la.txt"
        )

    speed = speed_df.to_numpy(dtype=np.float32)
    splits = split_starts(len(speed))
    train_end = int(splits["train"][-1]) + IN_STEPS + OUT_STEPS
    adjacency, distance = build_adjacency(
        sensor_ids, pd.read_csv(raw / "distances_la.csv")
    )
    locations = pd.read_csv(raw / "sensor_locations_la.csv", dtype={"sensor_id": str})
    locations = locations.set_index("sensor_id").loc[sensor_ids].reset_index()

    return MetrLA(
        timestamps=pd.DatetimeIndex(speed_df.index),
        sensor_ids=sensor_ids,
        speed=speed,
        filled=causal_fill(speed, train_end),
        adjacency=adjacency,
        distance=distance,
        locations=locations[["sensor_id", "latitude", "longitude"]],
        splits=splits,
    )
