"""Graph orientation, causal inputs, raw horizons, masks and matched controls."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from traffic.data import MetrLA, causal_fill, split_starts  # noqa: E402
from traffic.multihop import MultihopData, MultihopMLP, geometry_features, hop_supports  # noqa: E402


@pytest.fixture
def ds():
    speed = (np.arange(360).reshape(120, 3) % 60 + 10).astype(np.float32)
    speed[5:8, 1] = 0
    splits = split_starts(120)
    a = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=np.float32)
    return MetrLA(
        pd.date_range("2012-03-05", periods=120, freq="5min"),
        ["a", "b", "c"],
        speed,
        causal_fill(speed, splits["train"][-1] + 24),
        a,
        np.where(a > 0, 1000.0, np.inf),
        pd.DataFrame(
            dict(longitude=[-118.0, -117.99, -117.98], latitude=[34.0, 34.01, 34.02])
        ),
        splits,
    )


def test_direction_and_strict_hops(ds):
    incoming, outgoing, two_in, two_out = hop_supports(ds.adjacency)
    assert incoming[1, 0] == outgoing[0, 1] == 1
    assert two_out[0, 2] == two_in[2, 0] == 1
    assert two_out.sum() == two_in.sum() == 1
    assert not two_out[0, 1] and not two_out.diagonal().any()
    assert not outgoing[2].any()
    reverse = hop_supports(ds.adjacency.T)
    np.testing.assert_array_equal(reverse[2], two_out)
    g = geometry_features(ds)
    assert g.shape == (3, 8) and np.isfinite(g).all()
    assert g[0, 4] > 0 and g[1, 0] < 0
    assert g[0, 7] == 1 and not g[0, :4].any()


def test_alignment_masks_and_future_invariance(ds):
    data = MultihopData(ds)
    inputs, y = data.batch("val", np.arange(3))
    s = ds.splits["val"][0]
    np.testing.assert_array_equal(y, ds.speed[s + np.array([14, 17, 23])].T)
    train_inputs, _ = data.batch("train", np.array([1]))
    np.testing.assert_array_equal(train_inputs[0][0, 5:8, 1], 0)
    boundary = s + 12
    speed, filled = ds.speed.copy(), ds.filled.copy()
    speed[boundary:] = 999
    filled[boundary:] = 999
    # Normalization frozen to original training observations for this causality test.
    changed = replace(ds, speed=speed, filled=filled)
    changed.normalizer = ds.normalizer
    changed_inputs, _ = MultihopData(changed).batch("val", np.arange(3))
    for a, b in zip(inputs, changed_inputs):
        torch.testing.assert_close(a, b)
    assert np.isfinite(data.features.numpy()).all()


def test_disabled_inputs_and_matched_parameters(ds):
    inputs, _ = MultihopData(ds).batch("val", np.arange(3))
    models = []
    for variant, keep in [("local", 0), ("one", 8), ("two", 16), ("geometry", 24)]:
        torch.manual_seed(42)
        model = MultihopMLP(3, variant).eval()
        torch.nn.init.normal_(model.head.weight)
        changed = inputs[0].clone()
        changed[:, :, 7 + keep :] += 100
        torch.testing.assert_close(model(*inputs), model(changed, *inputs[1:]))
        x = inputs[0].clone().requires_grad_()
        model(x, *inputs[1:]).sum().backward()
        assert not torch.count_nonzero(x.grad[:, :, 7 + keep :])
        models.append(model)
    for other in models[1:]:
        for a, b in zip(models[0].parameters(), other.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_study_purges_targets(ds, monkeypatch):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "try_multihop", Path(__file__).parents[1] / "scripts/try_multihop.py"
    )
    try_multihop = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(try_multihop)

    monkeypatch.setattr(try_multihop, "load_metr_la", lambda: ds)
    purged = try_multihop.dataset()
    assert set(purged.splits) == {"train", "val"}
    assert purged.train_end <= purged.splits["val"][0] + 12
    np.testing.assert_array_equal(purged.speed, ds.speed)
    np.testing.assert_allclose(
        purged.normalizer(),
        (
            ds.speed[: purged.train_end][ds.speed[: purged.train_end] > 0].mean(),
            ds.speed[: purged.train_end][ds.speed[: purged.train_end] > 0].std(),
        ),
    )
