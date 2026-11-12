"""AI-assisted DCRNN-style reference. See third_party/dcrnn for attribution.

A[i,j] denotes a supplied directed i-to-j edge, not verified physical flow.
Use transposed row random walks, as upstream, with independent Chebyshev
recurrences per direction. Upstream mutates x0 between directions; we reset it.
"""

import numpy as np
import torch
from torch import nn


def supports(adjacency):
    a = torch.as_tensor(adjacency, dtype=torch.float32)
    walks = []
    for matrix in (a, a.T):
        degree = matrix.sum(1, keepdim=True)
        walks.append((matrix / degree.clamp_min(1e-12)).T)
    return torch.stack(walks)


def diffuse(x, walks, enabled=True):
    slots = [x]
    for walk in walks:
        first = torch.einsum("nm,bmf->bnf", walk, x)
        second = 2 * torch.einsum("nm,bmf->bnf", walk, first) - x
        slots.extend([first, second] if enabled else [torch.zeros_like(x)] * 2)
    return torch.stack(slots, -1).flatten(-2)


class Cell(nn.Module):
    def __init__(self, features, hidden):
        super().__init__()
        self.gates = nn.Linear((features + hidden) * 5, 2 * hidden, bias=False)
        self.candidate = nn.Linear((features + hidden) * 5, hidden, bias=False)
        self.gate_bias = nn.Parameter(torch.ones(2 * hidden))
        self.candidate_bias = nn.Parameter(torch.zeros(hidden))
        for layer in (self.gates, self.candidate):
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, x, h, walks, enabled):
        r, u = torch.sigmoid(
            self.gates(diffuse(torch.cat([x, h], -1), walks, enabled)) + self.gate_bias
        ).chunk(2, -1)
        c = torch.tanh(
            self.candidate(diffuse(torch.cat([x, r * h], -1), walks, enabled))
            + self.candidate_bias
        )
        return u * h + (1 - u) * c


class DCRNN(nn.Module):
    def __init__(self, adjacency, enabled=True, hidden=32):
        super().__init__()
        self.register_buffer("walks", supports(adjacency))
        self.enabled = enabled
        self.hidden = hidden
        self.encoder = Cell(2, hidden)
        self.decoder = Cell(1, hidden)
        self.readout = nn.Linear(hidden, 1)

    def forward(self, history, targets=None, probability=0.0, generator=None):
        h = history.new_zeros(history.shape[0], history.shape[2], self.hidden)
        for x in history.unbind(1):
            h = self.encoder(x, h, self.walks, self.enabled)
        previous = history.new_zeros(history.shape[0], history.shape[2], 1)
        outputs = []
        for step in range(12):
            h = self.decoder(previous, h, self.walks, self.enabled)
            previous = self.readout(h)
            outputs.append(previous[..., 0])
            if self.training and targets is not None and step < 11:
                # NaN denotes missing normalized targets. Missing targets always
                # fall back to the prediction, never normalized physical zero.
                draw = torch.rand((), generator=generator).item()
                if draw < probability:
                    y = targets[:, step, :, None]
                    previous = torch.where(torch.isfinite(y), y, previous)
        return torch.stack(outputs, 1)


def masked_mae(prediction, target):
    mask = torch.isfinite(target)
    return (prediction - torch.nan_to_num(target)).abs().masked_select(
        mask
    ).sum() / mask.sum().clamp_min(1)


def batches(ds, starts, device):
    mean, std = ds.normalizer()
    ix = starts[:, None] + np.arange(12)
    x = np.stack(
        [(ds.filled[ix] - mean) / std, (ds.speed[ix] > 0).astype(np.float32)], -1
    )
    raw = ds.speed[ix + 12]
    y = np.where(raw > 0, (raw - mean) / std, np.nan)
    return torch.tensor(x, device=device), torch.tensor(y, device=device)
