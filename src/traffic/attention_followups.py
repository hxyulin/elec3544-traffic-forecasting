"""AI-assisted controlled adaptations, not Graphormer or RoFormer reproductions."""

import numpy as np
import torch
from torch import nn

from .attention import AttentionData, AttentionMLP
from .data import IN_STEPS
from .identity import ResidualBlock
from .spatial_identity import NeighbourIdentityData, NeighbourIdentityMLP


def graph_tokens(adjacency):
    """Seven slots: self, top three undirected direct, top three strict two-hop.

    Selection strength is max(A,A.T); two-hop strength is its matrix square.
    Pair relation is (forward,reverse), each 0=self, 1=direct, 2=two, 3=other.
    A[i,j] declares i to j only; this does not assert traffic flow direction.
    """
    a = adjacency.copy()
    np.fill_diagonal(a, 0)
    s = np.maximum(a, a.T)
    two = s @ s
    two[s > 0] = 0
    np.fill_diagonal(two, 0)
    n = len(a)
    ids = np.repeat(np.arange(n)[:, None], 7, axis=1)
    valid = np.zeros((n, 7), bool)
    valid[:, 0] = True
    for i in range(n):
        for offset, strength in ((1, s), (4, two)):
            candidates = np.argsort(-strength[i], kind="stable")[:3]
            ids[i, offset : offset + len(candidates)] = candidates
            valid[i, offset : offset + len(candidates)] = strength[i, candidates] > 0
    local = np.arange(n)[:, None]
    edges = np.stack([a[ids, local], a[local, ids], ids == local], -1).astype(
        np.float32
    )
    edges[~valid] = 0
    reach = (a > 0).astype(np.int32)
    distance = np.full((n, n), 3, dtype=np.int64)
    distance[reach @ reach > 0] = 2
    distance[a > 0] = 1
    np.fill_diagonal(distance, 0)
    relation = distance * 4 + distance.T
    return ids, edges, valid, relation[ids[:, :, None], ids[:, None, :]]


class SpatialData(AttentionData):
    def __init__(self, ds, device="cpu"):
        NeighbourIdentityData.__init__(self, ds, device)
        ids, edges, valid, relation = graph_tokens(ds.adjacency)
        self.node_ids = torch.tensor(ids, device=device)
        self.edges = torch.tensor(edges, device=device)
        self.valid = torch.tensor(valid, device=device)
        self.relation = torch.tensor(relation, device=device)

    def batch(self, split, rows):
        inputs, y = NeighbourIdentityData.batch(self, split, rows)
        sensor = inputs[1]
        rows = torch.as_tensor(rows, device=self.device)
        origin = self.starts[split][rows // self.n_sensors] + IN_STEPS - 1
        nodes = self.node_ids[sensor]
        steps = origin[:, None, None] - IN_STEPS + 1 + self.offsets[None, None]
        histories = self.features[:, :, :2][steps, nodes[:, :, None]].reshape(
            len(rows), 7, -1
        )
        histories = torch.cat([histories, self.age[origin[:, None], nodes, None]], 2)
        return (
            *inputs,
            histories,
            nodes,
            self.edges[sensor],
            self.valid[sensor],
            self.relation[sensor],
        ), y


def sinusoidal(length, width):
    angle = torch.arange(length)[:, None] * 10000 ** (
        -torch.arange(0, width, 2).float() / width
    )
    return torch.stack([angle.sin(), angle.cos()], -1).flatten(-2)


def rotate(x, positions=None):
    if positions is None:
        positions = torch.arange(x.shape[-2], device=x.device)
    angle = positions[:, None] * 10000 ** (
        -torch.arange(0, x.shape[-1], 2, device=x.device).float() / x.shape[-1]
    )
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.stack(
        [a * angle.cos() - b * angle.sin(), a * angle.sin() + b * angle.cos()], -1
    ).flatten(-2)


class Encoder(nn.Module):
    def __init__(self, width, heads, content=True, bias=False, rope=False):
        super().__init__()
        self.width, self.heads = width, heads
        self.content, self.bias, self.rope = content, bias, rope
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, width * 3)
        self.relations = nn.Embedding(16, heads)
        self.output = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, width * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(width * 2, width),
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, tokens, valid, relation=None):
        b, k, w = tokens.shape
        q, key, value = (
            self.qkv(self.norm1(tokens))
            .reshape(b, k, 3, self.heads, w // self.heads)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        if self.rope:
            q, key = rotate(q), rotate(key)
        scores = (q @ key.transpose(-1, -2)) / (w // self.heads) ** 0.5
        scores = scores * int(self.content)
        if relation is not None:
            scores = scores + self.relations(relation).permute(0, 3, 1, 2) * int(
                self.bias
            )
        scores = scores.masked_fill(~valid[:, None, None, :], -torch.inf)
        attended = scores.softmax(-1) @ value
        tokens = tokens + self.dropout(
            self.output(attended.transpose(1, 2).reshape(b, k, w))
        )
        return tokens + self.dropout(self.ffn(self.norm2(tokens)))


class SpatialMLP(AttentionMLP):
    def __init__(self, n_sensors, variant):
        super().__init__(n_sensors)
        self.encoder = Encoder(
            16, 2, variant in ("content", "combined"), variant in ("bias", "combined")
        )

    def history_features(self, x, age, histories, nodes, edges, valid, relation):
        tokens = self.token_projection(
            torch.cat([histories, self.sensor(nodes), edges], 2)
        )
        encoded = self.encoder(tokens, valid, relation)[:, 0]
        return torch.cat(
            [NeighbourIdentityMLP.history_features(self, x, age), encoded], 1
        )


class TemporalMLP(NeighbourIdentityMLP):
    def __init__(self, n_sensors, variant):
        super().__init__(n_sensors)
        self.variant = variant
        self.temporal_projection = nn.Linear(7, 32)
        self.encoder = Encoder(32, 4, rope=variant == "rope")
        self.register_buffer("positions", sinusoidal(12, 32))
        self.blocks = nn.Sequential(ResidualBlock(160, 0.1), ResidualBlock(160, 0.1))
        self.head = nn.Linear(160, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def history_features(self, x, age, *context):
        tokens = self.temporal_projection(x[:, :, :7])
        if self.variant == "sinusoidal":
            tokens = tokens + self.positions
        encoded = self.encoder(
            tokens, torch.ones(tokens.shape[:2], dtype=torch.bool, device=x.device)
        )[:, -1]
        return torch.cat([super().history_features(x, age), encoded], 1)
