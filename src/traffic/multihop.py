"""AI-assisted exploratory graph histories and available static geometry."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .data import IN_STEPS
from .identity import IdentityData, gap_age
from .neural import graph_supports
from .spatial_identity import NeighbourIdentityMLP


def hop_supports(adjacency):
    """Incoming/outgoing one-hop and strict two-hop supports.

    Two-hop walks follow the same supplied orientation. Remove self and direct
    neighbours, then renormalize. No identity fallback or verified flow claim.
    """
    one, two = [], []
    for support in graph_supports(adjacency):
        a = support.copy()
        np.fill_diagonal(a, 0)
        b = a @ a
        b[a > 0] = 0
        np.fill_diagonal(b, 0)
        b /= np.maximum(b.sum(1, keepdims=True), 1e-12)
        one.append(a)
        two.append(b)
    return (*one, *two)


def geometry_features(ds):
    """Weighted displacement east/north, straight and listed road distance, km.

    Coordinates are aligned by the loader. Incoming road lengths use D.T.
    Empty supports yield zeros. No statistics are fitted to speed targets.
    """
    xy = ds.locations[["longitude", "latitude"]].to_numpy(float)
    if not np.isfinite(xy).all():
        raise ValueError("Geometry requires finite sensor coordinates")
    radians = np.deg2rad(xy)
    delta = radians[None, :, :] - radians[:, None, :]
    east = delta[:, :, 0] * np.cos(radians[:, 1, None]) * 6371
    north = delta[:, :, 1] * 6371
    straight = np.hypot(east, north)
    result = []
    for support, road in zip(
        hop_supports(ds.adjacency)[:2], (ds.distance.T, ds.distance)
    ):
        if not np.isfinite(road[support > 0]).all():
            raise ValueError("Active edge lacks listed distance")
        road = np.where(np.isfinite(road), road / 1000, 0)
        result.extend(
            [(support * value).sum(1) for value in (east, north, straight, road)]
        )
    return np.stack(result, axis=1).astype(np.float32)


def multihop_features(ds):
    mean, std = ds.normalizer()
    age = gap_age(ds.speed)
    reliability = 1 / (1 + age)
    sources = np.stack(
        [
            reliability * (ds.filled - mean) / std,
            ds.speed > 0,
            reliability * np.log1p(age),
            reliability,
        ],
        axis=-1,
    )
    signals = []
    for support in hop_supports(ds.adjacency):
        aggregate = np.einsum("ij,tjc->tic", support, sources, optimize=True)
        mass = aggregate[:, :, 3]
        for channel in (0, 2):
            np.divide(
                aggregate[:, :, channel],
                mass,
                out=aggregate[:, :, channel],
                where=mass > 0,
            )
        signals.append(aggregate)
    geometry = np.broadcast_to(geometry_features(ds), (*ds.speed.shape, 8))
    return np.concatenate([*signals, geometry], axis=2).astype(np.float32)


class MultihopData(IdentityData):
    def __init__(self, ds, device="cpu"):
        super().__init__(ds, device)
        self.features = torch.cat(
            [self.features, torch.tensor(multihop_features(ds), device=device)], dim=2
        )


class MultihopMLP(NeighbourIdentityMLP):
    """Same 24-slot encoder and parameters across all four input ablations."""

    def __init__(self, n_sensors, variant="geometry"):
        super().__init__(n_sensors)
        keep = {"local": 0, "one": 8, "two": 16, "geometry": 24}[variant]
        self.register_buffer("input_mask", (torch.arange(24) < keep).float())
        self.spatial = nn.Sequential(
            nn.Linear(IN_STEPS * 24, 32, bias=False), nn.ReLU()
        )

    def history_features(self, x, age, *context):
        masked = torch.cat([x[:, :, :7], x[:, :, 7:] * self.input_mask], dim=2)
        return super().history_features(masked, age)
