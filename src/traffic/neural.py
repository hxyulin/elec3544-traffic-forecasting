"""Temporal and graph GRUs with indexed batches instead of materialised windows."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .data import IN_STEPS, REPORT_HORIZONS, STEPS_PER_DAY, STEPS_PER_WEEK, MetrLA


class SensorGRU(nn.Module):
    """Each sensor uses only its own speed history, observation flags, and calendar.

    Parameters are shared across sensors. The three outputs are residuals around
    persistence, predicted directly rather than by feeding future speeds back in.
    """

    def __init__(self, hidden_size: int = 32):
        super().__init__()
        self.gru = nn.GRU(7, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, len(REPORT_HORIZONS))
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(x)
        return x[:, -1, 0, None] + self.head(hidden[-1])


def masked_mae(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """Mean absolute error over positive targets; an empty batch has zero gradient."""
    mask = true > 0
    return torch.where(mask, (pred - true).abs(), 0).sum() / mask.sum().clamp_min(1)


class NeuralData:
    """Gather (window, sensor) rows in window-major order on the selected device."""

    def __init__(self, ds: MetrLA, device: str, graph_mode: str = "none"):
        self.mean, self.std = ds.normalizer()
        self.n_sensors = ds.n_sensors
        self.splits = ds.splits
        self.device = device
        tow = ds.time_of_week
        day = 2 * np.pi * (tow % STEPS_PER_DAY) / STEPS_PER_DAY
        week = 2 * np.pi * tow / STEPS_PER_WEEK
        calendar = np.stack(
            [
                np.sin(day),
                np.cos(day),
                np.sin(week),
                np.cos(week),
                tow // STEPS_PER_DAY >= 5,
            ],
            axis=1,
        )
        features = np.concatenate(
            [
                ((ds.filled - self.mean) / self.std)[:, :, None],
                (ds.speed > 0)[:, :, None],
                np.broadcast_to(calendar[:, None], (*ds.speed.shape, 5)),
            ],
            axis=2,
        ).astype(np.float32)
        if graph_mode not in ("none", "road", "identity"):
            raise ValueError(f"Unknown graph mode: {graph_mode}")
        if graph_mode != "none":
            local = features[:, :, :2]
            signals = []
            if graph_mode == "road":
                for support in graph_supports(ds.adjacency):
                    aggregate = support @ local.transpose(1, 0, 2).reshape(
                        ds.n_sensors, -1
                    )
                    aggregate = aggregate.reshape(
                        ds.n_sensors, len(local), 2
                    ).transpose(1, 0, 2)
                    signals.append(aggregate - local)
            else:
                signals = [np.zeros_like(local), np.zeros_like(local)]
            features = np.concatenate([features, *signals], axis=2)
        self.features = torch.tensor(features, device=device)
        self.speed = torch.tensor(ds.speed, device=device)
        self.offsets = torch.arange(IN_STEPS, device=device)
        self.target_offsets = (
            torch.tensor(REPORT_HORIZONS, device=device) + IN_STEPS - 1
        )
        self.starts = {k: torch.tensor(v, device=device) for k, v in ds.splits.items()}

    def size(self, split: str) -> int:
        return len(self.splits[split]) * self.n_sensors

    def batch(self, split: str, rows: np.ndarray):
        ids = torch.as_tensor(rows, device=self.device)
        sensor = ids % self.n_sensors
        starts = self.starts[split][ids // self.n_sensors]
        x = self.features[starts[:, None] + self.offsets, sensor[:, None]]
        y = self.speed[starts[:, None] + self.target_offsets, sensor[:, None]]
        return x, y


@torch.no_grad()
def predict(
    model: SensorGRU, data: NeuralData, split: str, batch_size: int = 8192
) -> np.ndarray:
    model.eval()
    output = np.empty((data.size(split), len(REPORT_HORIZONS)), dtype=np.float32)
    for start in range(0, len(output), batch_size):
        end = min(start + batch_size, len(output))
        x, _ = data.batch(split, np.arange(start, end))
        output[start:end] = (model(x) * data.std + data.mean).cpu().numpy()
    return output.reshape(-1, data.n_sensors, len(REPORT_HORIZONS)).transpose(0, 2, 1)


def graph_supports(adjacency: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row-normalized incoming and outgoing road weights, excluding self-loops.

    A[i,j] is a road link from i to j. Empty rows fall back to the identity,
    so their neighbour-minus-local signal is zero. The local path is separate.
    """
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("Adjacency must be square")
    if not np.isfinite(adjacency).all() or (adjacency < 0).any():
        raise ValueError("Adjacency must contain finite nonnegative weights")
    weights = adjacency.astype(np.float32).copy()
    np.fill_diagonal(weights, 0)
    supports = []
    for a in (weights.T, weights):
        total = a.sum(1)
        support = a / np.where(total > 0, total, 1)[:, None]
        isolated = np.flatnonzero(total == 0)
        support[isolated, isolated] = 1
        supports.append(support)
    return tuple(supports)


class GraphSensorGRU(SensorGRU):
    """Learn one-hop incoming/outgoing filtering of speed and observation flags.

    Packed inputs hold seven local channels, then two incoming differences and
    two outgoing differences. The persistence anchor always uses the local speed.
    Zero graph signals reduce the model to the temporal GRU, including gradients.
    """

    def __init__(self, hidden_size: int = 32):
        super().__init__(hidden_size)
        self.incoming = nn.Linear(2, 2, bias=False)
        self.outgoing = nn.Linear(2, 2, bias=False)
        nn.init.zeros_(self.incoming.weight)
        nn.init.zeros_(self.outgoing.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial = (
            x[:, :, :2] + self.incoming(x[:, :, 7:9]) + self.outgoing(x[:, :, 9:11])
        )
        filtered = torch.cat([spatial, x[:, :, 2:7]], dim=2)
        _, hidden = self.gru(filtered)
        return x[:, -1, 0, None] + self.head(hidden[-1])
