"""Separate road-neighbour history channels for the identity forecasting baseline."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import IN_STEPS, MetrLA
from .identity import IdentityData, IdentityMLP, ResidualBlock, gap_age
from .neural import graph_supports


def neighbour_features(ds: MetrLA) -> np.ndarray:
    """Causal incoming/outgoing messages, with stale readings downweighted.

    For each direction and timestamp: reliability-weighted normalized speed,
    observed weight share, reliability-weighted log gap age, and reliability mass.
    Reliability is 1/(1+consecutive missing steps). Empty road rows are zero.
    All aggregation uses the same timestamp, never a forecast target timestamp.
    """
    mean, std = ds.normalizer()
    age = gap_age(ds.speed)
    reliability = 1 / (1 + age)
    observed = (ds.speed > 0).astype(np.float32)
    sources = np.stack(
        [
            reliability * (ds.filled - mean) / std,
            observed,
            reliability * np.log1p(age),
            reliability,
        ],
        axis=-1,
    )
    signals = []
    for support in graph_supports(ds.adjacency):
        # graph_supports uses identity fallback for isolated nodes; remove it here.
        support = support.copy()
        np.fill_diagonal(support, 0)
        aggregate = support @ sources.transpose(1, 0, 2).reshape(ds.n_sensors, -1)
        aggregate = aggregate.reshape(ds.n_sensors, len(age), 4).transpose(1, 0, 2)
        mass = aggregate[:, :, 3]
        for channel in (0, 2):
            np.divide(
                aggregate[:, :, channel],
                mass,
                out=aggregate[:, :, channel],
                where=mass > 0,
            )
        signals.append(aggregate)
    return np.concatenate(signals, axis=2).astype(np.float32)


class NeighbourIdentityData(IdentityData):
    def __init__(self, ds: MetrLA, device: str):
        super().__init__(ds, device)
        spatial = torch.tensor(neighbour_features(ds), device=device)
        self.features = torch.cat([self.features, spatial], dim=2)


class NeighbourIdentityMLP(IdentityMLP):
    """Keep 64 local and 32 spatial hidden channels separate before prediction.

    The control zeros messages before their bias-free encoder. Both models have
    identical parameters and initialization; the local persistence anchor stays.
    """

    def __init__(self, n_sensors: int, use_neighbours: bool = True):
        super().__init__(n_sensors)
        self.use_neighbours = use_neighbours
        self.spatial = nn.Sequential(nn.Linear(IN_STEPS * 8, 32, bias=False), nn.ReLU())
        self.blocks = nn.Sequential(ResidualBlock(128, 0.1), ResidualBlock(128, 0.1))
        self.head = nn.Linear(128, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def history_features(
        self, x: torch.Tensor, age: torch.Tensor, *context
    ) -> torch.Tensor:
        spatial = x[:, :, 7:]
        if not self.use_neighbours:
            spatial = spatial * 0
        return torch.cat(
            [
                super().history_features(x, age),
                self.spatial(spatial.reshape(len(x), -1)),
            ],
            dim=1,
        )


def forecast_loss(
    pred: torch.Tensor, true: torch.Tensor, std: float, objective: str
) -> torch.Tensor:
    """Masked losses in normalized units, with fixed settings selected in advance.

    Huber uses smooth L1 with beta=5 mph. Mixed uses .8 MAE + .2 MSE/std.
    Dividing all objectives by training std preserves the baseline MAE scale.
    """
    error = pred - true
    if objective == "mae":
        values = error.abs()
    elif objective == "huber":
        values = F.smooth_l1_loss(pred, true, reduction="none", beta=5.0)
    elif objective == "mixed":
        values = 0.8 * error.abs() + 0.2 * error.square() / std
    else:
        raise ValueError(f"Unknown objective: {objective}")
    mask = true > 0
    return torch.where(mask, values, 0).sum() / mask.sum().clamp_min(1) / std
