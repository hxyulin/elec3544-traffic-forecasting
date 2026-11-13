"""Training-only finite-value/runtime pilot, no validation selection."""

import json
import time

import numpy as np
import torch
from try_attention_spatial import dataset

from traffic.attention_followups import SpatialData, SpatialMLP, TemporalMLP
from traffic.spatial_identity import NeighbourIdentityData, forecast_loss

torch.set_num_threads(4)
ds = dataset()
results = []
for cls, data_cls, variant in (
    (SpatialMLP, SpatialData, "combined"),
    (TemporalMLP, NeighbourIdentityData, "rope"),
):
    data = data_cls(ds, "cuda")
    model = cls(ds.n_sensors, variant).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for i in range(500):
        inputs, y = data.batch(
            "train",
            np.random.default_rng(i).choice(data.size("train"), 2048, replace=False),
        )
        opt.zero_grad()
        loss = forecast_loss(model(*inputs) * data.std + data.mean, y, data.std, "mae")
        assert torch.isfinite(loss)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    results.append(
        dict(
            variant=variant,
            batches=500,
            seconds=time.perf_counter() - started,
            peak_bytes=torch.cuda.max_memory_allocated(),
        )
    )
print(json.dumps(results, indent=2))
