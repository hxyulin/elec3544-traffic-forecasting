"""Shared per-sensor GRU and indexed batches without materialising every window."""

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

    def __init__(self, ds: MetrLA, device: str):
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
