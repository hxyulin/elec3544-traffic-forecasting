"""Validation candidate with sensor/time identities and causal gap age.

Inspired by STID (Shao et al., 2022), not a reproduction of its architecture.
A shared horizon-conditioned residual MLP predicts changes from persistence.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .data import IN_STEPS, REPORT_HORIZONS, STEPS_PER_DAY, MetrLA
from .neural import NeuralData


def gap_age(speed: np.ndarray) -> np.ndarray:
    """Consecutive missing steps, including gaps before any sensor observation."""
    time = np.arange(len(speed))[:, None]
    last = np.maximum.accumulate(np.where(speed > 0, time, -1), axis=0)
    return (time - last).astype(np.float32)


class IdentityData(NeuralData):
    """Reuse the benchmark windows; add only known identities and past gap length."""

    def __init__(self, ds: MetrLA, device: str):
        super().__init__(ds, device)
        self.tow = torch.tensor(ds.time_of_week, device=device)
        self.age = torch.tensor(np.log1p(gap_age(ds.speed)), device=device)
        self.horizons = torch.tensor(REPORT_HORIZONS, device=device)

    def batch(self, split: str, rows: np.ndarray):
        x, y = super().batch(split, rows)
        ids = torch.as_tensor(rows, device=self.device)
        sensor = ids % self.n_sensors
        origin = self.starts[split][ids // self.n_sensors] + IN_STEPS - 1
        target_tow = self.tow[origin[:, None] + self.horizons]
        age = self.age[origin, sensor]
        return (x, sensor, target_tow, age), y


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)


class IdentityMLP(nn.Module):
    """History, target-calendar identities and optional sensor identity.

    Both variants have identical parameter counts. Disabling identity supplies
    zero sensor vectors, giving their embedding weights zero gradient.
    """

    def __init__(self, n_sensors: int, use_identity: bool = True, dropout: float = 0.1):
        super().__init__()
        self.use_identity = use_identity
        self.history = nn.Sequential(nn.Linear(IN_STEPS * 2 + 1, 64), nn.ReLU())
        self.sensor = nn.Embedding(n_sensors, 16)
        self.time_of_day = nn.Embedding(STEPS_PER_DAY, 8)
        self.day_of_week = nn.Embedding(7, 4)
        self.horizon = nn.Embedding(len(REPORT_HORIZONS), 4)
        self.blocks = nn.Sequential(
            ResidualBlock(96, dropout), ResidualBlock(96, dropout)
        )
        self.head = nn.Linear(96, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        x: torch.Tensor,
        sensor: torch.Tensor,
        target_tow: torch.Tensor,
        age: torch.Tensor,
    ) -> torch.Tensor:
        h = len(REPORT_HORIZONS)
        history = self.history(
            torch.cat([x[:, :, :2].reshape(len(x), -1), age[:, None]], dim=1)
        )
        identity = self.sensor(sensor)
        if not self.use_identity:
            identity = identity * 0
        horizon = self.horizon(torch.arange(h, device=x.device))[None].expand(
            len(x), -1, -1
        )
        features = torch.cat(
            [
                history[:, None].expand(-1, h, -1),
                identity[:, None].expand(-1, h, -1),
                self.time_of_day(target_tow % STEPS_PER_DAY),
                self.day_of_week(target_tow // STEPS_PER_DAY),
                horizon,
            ],
            dim=2,
        )
        return x[:, -1, 0, None] + self.head(self.blocks(features)).squeeze(-1)


@torch.no_grad()
def predict_identity(
    model: IdentityMLP, data: IdentityData, split: str, batch_size: int = 8192
) -> np.ndarray:
    model.eval()
    output = np.empty((data.size(split), len(REPORT_HORIZONS)), dtype=np.float32)
    for start in range(0, len(output), batch_size):
        end = min(start + batch_size, len(output))
        inputs, _ = data.batch(split, np.arange(start, end))
        output[start:end] = (model(*inputs) * data.std + data.mean).cpu().numpy()
    return output.reshape(-1, data.n_sensors, len(REPORT_HORIZONS)).transpose(0, 2, 1)
