import numpy as np
import torch

from traffic.attention_followups import (
    Encoder,
    SpatialMLP,
    TemporalMLP,
    graph_tokens,
    rotate,
    sinusoidal,
)


def test_graph_reference():
    a = np.zeros((8, 8), np.float32)
    a[0, 1] = 2
    a[1, 2] = 1
    ids, edges, valid, relation = graph_tokens(a)
    assert ids[0, 1] == 1 and ids[0, 4] == 2
    assert valid[0].sum() == 3
    assert relation[0, 0, 1] == 7  # forward direct, reverse unreachable
    assert relation[0, 0, 4] == 11
    assert not edges[~valid].any()


def test_interventions_permutation_padding():
    torch.manual_seed(3)
    m = Encoder(16, 2, False, False).eval()
    x = torch.randn(2, 7, 16)
    valid = torch.tensor([[True] * 5 + [False] * 2] * 2)
    rel = torch.randint(16, (2, 7, 7))
    y = m(x, valid, rel)
    y.square().sum().backward()
    assert not m.relations.weight.grad.any()
    assert not m.qkv.weight.grad[:32].any()
    p = torch.tensor([0, 3, 2, 1, 4, 6, 5])
    torch.testing.assert_close(m(x[:, p], valid[:, p], rel[:, p][:, :, p]), y[:, p])
    changed = x.clone()
    changed[:, 5:] = 1000
    torch.testing.assert_close(m(changed, valid, rel)[:, :5], y[:, :5])
    m.content = True
    with torch.no_grad():
        m.relations.weight.zero_()
    a = m(x, valid, rel)
    m.bias = True
    torch.testing.assert_close(a, m(x, valid, rel))


def test_rotation_reference():
    x = torch.randn(2, 4, 12, 8, requires_grad=True)
    y = rotate(x)
    torch.testing.assert_close(x.square().sum(-1), y.square().sum(-1))
    angle = torch.tensor(1.0).cos()
    torch.testing.assert_close(
        y[..., 1, 0], x[..., 1, 0] * angle - x[..., 1, 1] * torch.tensor(1.0).sin()
    )
    shifted = rotate(x, torch.arange(12) + 3)
    torch.testing.assert_close(
        y @ y.transpose(-1, -2),
        shifted @ shifted.transpose(-1, -2),
        atol=1e-5,
        rtol=1e-5,
    )
    y.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert sinusoidal(12, 32).shape == (12, 32)


def test_equal_initial_parameters():
    for cls, variants in (
        (SpatialMLP, ["uniform", "content", "bias", "combined"]),
        (TemporalMLP, ["sinusoidal", "rope"]),
    ):
        states = []
        for variant in variants:
            torch.manual_seed(42)
            states.append(dict(cls(8, variant).named_parameters()))
        for state in states[1:]:
            assert state.keys() == states[0].keys()
            for k in state:
                assert torch.equal(state[k], states[0][k])


def test_causal_horizons_train_stats():
    from dataclasses import replace

    from test_multihop import ds as fixture

    from traffic.attention_followups import SpatialData
    from traffic.spatial_identity import NeighbourIdentityData, forecast_loss

    ds = fixture.__wrapped__()
    origin = ds.splits["val"][0] + 11
    for cls in (SpatialData, NeighbourIdentityData):
        data = cls(ds, "cpu")
        inputs, y = data.batch("val", np.arange(3))
        np.testing.assert_array_equal(y, ds.speed[origin + np.array([3, 6, 12])].T)
        changed_speed, changed_fill = ds.speed.copy(), ds.filled.copy()
        changed_speed[origin + 1 :] = 999
        changed_fill[origin + 1 :] = 999
        changed = replace(ds, speed=changed_speed, filled=changed_fill)
        changed.normalizer = ds.normalizer
        other, _ = cls(changed, "cpu").batch("val", np.arange(3))
        for a, b in zip(inputs, other):
            torch.testing.assert_close(a, b)
        assert data.mean == ds.normalizer()[0]
    pred = torch.tensor([[4.0, 100.0, 6.0]], requires_grad=True)
    true = torch.tensor([[2.0, 0.0, 8.0]])
    assert forecast_loss(pred, true, 1.0, "mae") == 2
    forecast_loss(pred, true, 1.0, "mae").backward()
    assert pred.grad[0, 1] == 0


def test_zero_scores_equal_valid_value_mean():
    model = Encoder(16, 2, content=False, bias=False).eval()
    with torch.no_grad():
        model.output.weight.copy_(torch.eye(16))
        model.output.bias.zero_()
        for parameter in model.ffn.parameters():
            parameter.zero_()
    tokens = torch.randn(2, 7, 16)
    valid = torch.tensor([[True] * 4 + [False] * 3] * 2)
    values = model.qkv(model.norm1(tokens))[..., 32:]
    reference = tokens + values[:, :4].mean(1)[:, None]
    relations = torch.randint(16, (2, 7, 7))
    torch.testing.assert_close(model(tokens, valid, relations), reference)
