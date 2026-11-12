import numpy as np
import torch

from traffic.dcrnn import DCRNN, diffuse, masked_mae, supports


def test_operator():
    a = np.array([[0, 2, 1], [0, 0, 3], [4, 0, 0]], dtype=np.float32)
    x = torch.arange(6.0).reshape(1, 3, 2)
    walks = supports(a)
    expected = [x.numpy()]
    for m in (a, a.T):
        p = (m / m.sum(1, keepdims=True)).T
        expected.extend([p @ x.numpy(), (2 * p @ p - np.eye(3)) @ x.numpy()])
    np.testing.assert_allclose(
        diffuse(x, walks).numpy(), np.stack(expected, -1).reshape(1, 3, 10), atol=1e-6
    )


def test_disabled_and_evaluation():
    torch.manual_seed(42)
    on = DCRNN(np.zeros((3, 3)))
    off = DCRNN(np.zeros((3, 3)), False)
    off.load_state_dict(on.state_dict())
    x = torch.randn(2, 12, 3, 2)
    # Zero walk has a -x second polynomial; disabled explicitly zeros slots.
    assert torch.count_nonzero(diffuse(x[:, 0], off.walks, False)[..., 1::5]) == 0
    assert sum(p.numel() for p in on.parameters()) == sum(
        p.numel() for p in off.parameters()
    )
    on.eval()
    a = on(x)
    assert torch.equal(a, on(x, torch.randn(2, 12, 3), 1))
    clone = DCRNN(np.zeros((3, 3)))
    clone.load_state_dict(on.state_dict())
    clone.eval()
    assert torch.equal(a, clone(x))
    on.train()
    missing = torch.full((2, 12, 3), float("nan"))
    assert torch.equal(on(x), on(x, missing, 1))
    loss = masked_mae(on(x), torch.zeros(2, 12, 3))
    loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in on.parameters()
    )
    p = torch.ones(2, requires_grad=True)
    masked_mae(p, torch.full((2,), float("nan"))).backward()
    assert torch.equal(p.grad, torch.zeros(2))


def test_windows_and_training_policy():
    from types import SimpleNamespace

    from traffic.dcrnn import batches

    raw = np.arange(40 * 3, dtype=np.float32).reshape(40, 3) + 1
    ds = SimpleNamespace(speed=raw, filled=raw.copy(), normalizer=lambda: (10.0, 2.0))
    x, y = batches(ds, np.array([0, 4]), "cpu")
    np.testing.assert_array_equal(x[1, :, :, 0], (raw[4:16] - 10) / 2)
    np.testing.assert_array_equal(y[1], (raw[16:28] - 10) / 2)
    model = DCRNN(np.ones((3, 3)))
    model.train()
    zero = model(x, y, 0, torch.Generator().manual_seed(7))
    forced = model(x, y, 1, torch.Generator().manual_seed(7))
    assert not torch.equal(zero[:, 1:], forced[:, 1:])
    assert torch.equal(zero[:, 0], forced[:, 0])
    model.eval()
    assert torch.equal(model(x, y, 1), model(x, y + 900, 1))
    assert torch.equal(model(x, y, 1), model(x, torch.full_like(y, float("nan")), 1))


def test_purge_and_training_statistics():
    from dataclasses import replace

    import pandas as pd

    from traffic.data import MetrLA, causal_fill, split_starts

    speed = np.arange(100 * 3, dtype=np.float32).reshape(100, 3) + 1
    split = split_starts(100)
    boundary = int(split["val"][0]) + 12
    split = dict(
        train=split["train"][split["train"] + 24 <= boundary], val=split["val"]
    )
    ds = MetrLA(
        pd.date_range("2020", periods=100, freq="5min"),
        ["a", "b", "c"],
        speed,
        speed.copy(),
        np.eye(3),
        np.eye(3),
        pd.DataFrame(),
        split,
    )
    changed = speed.copy()
    changed[boundary:] = 9000
    assert ds.train_end <= boundary
    assert ds.normalizer() == replace(ds, speed=changed).normalizer()
    np.testing.assert_array_equal(
        causal_fill(speed, boundary)[:boundary],
        causal_fill(changed, boundary)[:boundary],
    )


def test_disabled_self_path_and_projection_slots():
    torch.manual_seed(9)
    model = DCRNN(np.ones((3, 3)), False).eval()
    x = torch.randn(2, 12, 3, 2)
    changed = x.clone()
    changed[:, :, 1:] += 100
    assert torch.equal(model(x)[:, :, 0], model(changed)[:, :, 0])
    z = torch.randn(2, 3, 7)
    slots = diffuse(z, model.walks, False).reshape(2, 3, 7, 5)
    assert torch.equal(slots[..., 0], z)
    assert torch.count_nonzero(slots[..., 1:]) == 0
    layer = torch.nn.Linear(35, 4, bias=False)
    direct = torch.nn.functional.linear(z, layer.weight.reshape(4, 7, 5)[..., 0])
    torch.testing.assert_close(layer(slots.flatten(-2)), direct)
