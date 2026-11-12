"""A small Transformer over local and road-neighbour history tokens.

One pre-normalized encoder block uses two heads, width 16 and FFN width 32.
The uniform-attention intervention keeps every parameter and input unchanged.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import IN_STEPS, OUT_STEPS, STEPS_PER_DAY, MetrLA, causal_fill
from .identity import ResidualBlock
from .spatial_identity import NeighbourIdentityData, NeighbourIdentityMLP


def neighbour_tokens(adjacency):
    n = len(adjacency)
    strength = np.maximum(adjacency, adjacency.T).copy()
    np.fill_diagonal(strength, 0)
    ids = np.repeat(np.arange(n)[:, None], 4, axis=1)
    count = min(3, n - 1)
    ids[:, 1 : 1 + count] = np.argsort(-strength, axis=1, kind="stable")[:, :count]
    local = np.arange(n)[:, None]
    valid = strength[local, ids] > 0
    valid[:, 0] = True
    edges = np.stack(
        [
            adjacency[ids, local],
            adjacency[local, ids],
            np.zeros_like(ids, dtype=np.float32),
        ],
        axis=-1,
    ).astype(np.float32)
    edges[:, 0] = [0, 0, 1]
    edges[~valid] = 0
    return ids, edges, valid


class AttentionData(NeighbourIdentityData):
    def __init__(self, ds: MetrLA, device="cpu"):
        super().__init__(ds, device)
        ids, edges, valid = neighbour_tokens(ds.adjacency)
        self.node_ids = torch.tensor(ids, device=device)
        self.edges = torch.tensor(edges, device=device)
        self.valid = torch.tensor(valid, device=device)

    def batch(self, split, rows):
        inputs, y = super().batch(split, rows)
        sensor = inputs[1]
        ids = torch.as_tensor(rows, device=self.device)
        origin = self.starts[split][ids // self.n_sensors] + IN_STEPS - 1
        nodes = self.node_ids[sensor]
        steps = origin[:, None, None] - IN_STEPS + 1 + self.offsets[None, None]
        histories = self.features[:, :, :2][steps, nodes[:, :, None]].reshape(
            len(rows), 4, -1
        )
        histories = torch.cat(
            [histories, self.age[origin[:, None], nodes, None]], dim=2
        )
        return (*inputs, histories, nodes, self.edges[sensor], self.valid[sensor]), y


class SensorAttention(nn.Module):
    def __init__(self, uniform=False):
        super().__init__()
        self.uniform = uniform
        self.norm1 = nn.LayerNorm(16)
        self.qkv = nn.Linear(16, 48)
        self.output = nn.Linear(16, 16)
        self.dropout = nn.Dropout(0.1)
        self.norm2 = nn.LayerNorm(16)
        self.ffn = nn.Sequential(
            nn.Linear(16, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 16)
        )

    def forward(self, tokens, valid):
        b, k, _ = tokens.shape
        q, key, value = (
            self.qkv(self.norm1(tokens))
            .reshape(b, k, 3, 2, 8)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        if self.uniform:
            q, key = q * 0, key * 0
        attended = F.scaled_dot_product_attention(
            q, key, value, attn_mask=valid[:, None, None, :], dropout_p=0.0
        )
        tokens = tokens + self.dropout(
            self.output(attended.transpose(1, 2).reshape(b, k, 16))
        )
        return tokens + self.dropout(self.ffn(self.norm2(tokens)))


class AttentionMLP(NeighbourIdentityMLP):
    def __init__(self, n_sensors, uniform=False):
        super().__init__(n_sensors)
        self.token_projection = nn.Linear(25 + 16 + 3, 16)
        self.encoder = SensorAttention(uniform)
        self.blocks = nn.Sequential(ResidualBlock(144, 0.1), ResidualBlock(144, 0.1))
        self.head = nn.Linear(144, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def history_features(self, x, age, histories, nodes, edges, valid):
        tokens = self.token_projection(
            torch.cat([histories, self.sensor(nodes), edges], dim=2)
        )
        encoded = self.encoder(tokens, valid)[:, 0]
        return torch.cat([super().history_features(x, age), encoded], dim=1)


def purged_period(ds: MetrLA, evaluation_start: int, days: int = 7) -> MetrLA:
    """Disjoint full target periods: expanding train, tuning week, evaluation week.

    Histories may use preceding observations, as in rolling forecasting. No target
    timestamp crosses a split, and preprocessing is refitted to this train period.
    """
    length = days * STEPS_PER_DAY
    tune_start, end = evaluation_start - length, evaluation_start + length
    if tune_start < IN_STEPS + OUT_STEPS or end > len(ds.speed):
        raise ValueError("Insufficient data for the purged periods")
    splits = {
        "train": np.arange(tune_start - IN_STEPS - OUT_STEPS + 1),
        "val": np.arange(
            tune_start - IN_STEPS, evaluation_start - IN_STEPS - OUT_STEPS + 1
        ),
        "evaluation": np.arange(
            evaluation_start - IN_STEPS, end - IN_STEPS - OUT_STEPS + 1
        ),
    }
    return replace(ds, splits=splits, filled=causal_fill(ds.speed, tune_start))
