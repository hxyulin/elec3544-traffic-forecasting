"""Attention masking, controls, input alignment and purged time boundaries."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from traffic.attention import (
    AttentionData,
    AttentionMLP,
    SensorAttention,
    neighbour_tokens,
    purged_period,
)
from traffic.data import IN_STEPS, MetrLA, causal_fill, split_starts
from traffic.identity import predict_identity


@pytest.fixture
def ds():
    speed = (np.arange(2700).reshape(900, 3) % 60 + 10).astype(np.float32)
    speed[10:15, 1] = 0
    speed[:3, 0] = 0
    splits = split_starts(900)
    return MetrLA(
        pd.date_range("2012-03-05", periods=900, freq="5min"),
        ["a", "b", "c"],
        speed,
        causal_fill(speed, splits["train"][-1] + 24),
        np.array([[0, 0.8, 0.3], [0, 0, 0], [0, 0.4, 0]], dtype=np.float32),
        np.eye(3),
        pd.DataFrame(),
        splits,
    )


def test_neighbour_tokens_have_local_fallback_and_directed_edges():
    adjacency = np.array([[0, 0.8, 0.3], [0, 0, 0], [0, 0.4, 0]], dtype=np.float32)
    ids, edges, valid = neighbour_tokens(adjacency)
    np.testing.assert_array_equal(ids[0, :3], [0, 1, 2])
    np.testing.assert_array_equal(edges[0, 0], [0, 0, 1])
    np.testing.assert_allclose(edges[0, 1], [0, 0.8, 0])
    np.testing.assert_allclose(edges[1, 1], [0.8, 0, 0])
    assert valid[:, 0].all() and not valid[:, 3].any()
    _, empty_edges, empty_valid = neighbour_tokens(np.zeros((1, 1)))
    np.testing.assert_array_equal(empty_valid, [[True, False, False, False]])
    assert np.isfinite(empty_edges).all()


def test_attention_padding_and_neighbour_permutation():
    torch.manual_seed(42)
    tokens = torch.randn(2, 4, 16)
    valid = torch.tensor([[True, True, True, False], [True, False, False, False]])
    for uniform in (False, True):
        block = SensorAttention(uniform).eval()
        expected = block(tokens, valid)[:, 0]
        changed = tokens.clone()
        changed[~valid] = 1000
        torch.testing.assert_close(block(changed, valid)[:, 0], expected)
        order = [0, 2, 3, 1]
        torch.testing.assert_close(
            block(tokens[:, order], valid[:, order])[:, 0], expected
        )
        assert torch.isfinite(expected).all()


def test_uniform_control_has_identical_parameters_but_zero_query_key_gradients():
    torch.manual_seed(42)
    learned = SensorAttention().eval()
    torch.manual_seed(42)
    uniform = SensorAttention(True).eval()
    for a, b in zip(learned.parameters(), uniform.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    tokens = torch.randn(2, 4, 16)
    valid = torch.ones(2, 4, dtype=torch.bool)
    assert not torch.allclose(
        learned(tokens, valid)[:, 0], uniform(tokens, valid)[:, 0]
    )
    uniform(tokens, valid)[:, 0].square().sum().backward()
    assert torch.count_nonzero(uniform.qkv.weight.grad[:32]) == 0
    assert torch.count_nonzero(uniform.qkv.weight.grad[32:]) > 0


def test_token_batch_alignment_and_initial_persistence(ds, tmp_path):
    data = AttentionData(ds)
    inputs, y = data.batch("val", np.arange(6))
    x, sensor, _, age, history, nodes, edges, valid = inputs
    torch.testing.assert_close(history[:, 0, :24], x[:, :, :2].reshape(6, 24))
    torch.testing.assert_close(history[:, 0, 24], age)
    torch.testing.assert_close(nodes[:, 0], sensor)
    assert valid[:, 0].all()
    expected = (
        ds.windows("val")[1][:2, np.array([2, 5, 11])].transpose(0, 2, 1).reshape(6, 3)
    )
    np.testing.assert_array_equal(y, expected)
    for uniform in (True, False):
        model = AttentionMLP(ds.n_sensors, uniform)
        p = predict_identity(model, data, "val", batch_size=19)
        np.testing.assert_allclose(
            p,
            np.repeat(ds.filled[ds.splits["val"] + IN_STEPS - 1, None], 3, axis=1),
            atol=1e-5,
        )
        torch.nn.init.normal_(model.head.weight)
        model.eval()
        path = tmp_path / "attention.pt"
        torch.save(model.state_dict(), path)
        restored = AttentionMLP(ds.n_sensors, uniform).eval()
        restored.load_state_dict(torch.load(path, weights_only=True))
        torch.testing.assert_close(model(*inputs), restored(*inputs))


def test_attention_inputs_ignore_future_speeds(ds):
    original = AttentionData(ds)
    first_target = ds.splits["test"][0] + IN_STEPS
    speed, filled = ds.speed.copy(), ds.filled.copy()
    speed[first_target:] = 1000
    filled[first_target:] = 1000
    changed = AttentionData(replace(ds, speed=speed, filled=filled))
    for a, b in zip(
        original.batch("test", np.arange(3))[0], changed.batch("test", np.arange(3))[0]
    ):
        torch.testing.assert_close(a, b)


def test_purged_periods_refit_preprocessing_and_separate_full_target_ranges(ds):
    fold = purged_period(ds, 550, days=1)
    train, val, evaluation = [fold.splits[k] for k in ["train", "val", "evaluation"]]
    assert fold.train_end == 262
    assert train[-1] + 23 < val[0] + 12
    assert val[-1] + 23 < evaluation[0] + 12
    assert evaluation[0] + 12 == 550 and evaluation[-1] + 23 == 837
    speed = ds.speed.copy()
    speed[fold.train_end :] = 1000
    changed = purged_period(replace(ds, speed=speed), 550, days=1)
    assert changed.normalizer() == fold.normalizer()
    np.testing.assert_array_equal(
        changed.filled[: fold.train_end], fold.filled[: fold.train_end]
    )
    with pytest.raises(ValueError):
        purged_period(ds, 100, days=1)


def test_training_and_checkpoint_selection_ignore_outer_targets(ds, tmp_path):
    from traffic.training import fit_forecaster

    fold = purged_period(ds, 550, days=1)
    speed = ds.speed.copy()
    speed[550:] = 999
    changed = purged_period(replace(ds, speed=speed), 550, days=1)
    saved = []
    prior_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        for i, dataset in enumerate([fold, changed]):
            torch.manual_seed(99)
            data = AttentionData(dataset)
            model = AttentionMLP(dataset.n_sensors)
            truth = dataset.windows("val")[1][:, np.array([2, 5, 11])]
            directory = tmp_path / str(i)
            directory.mkdir()
            fit_forecaster(
                model,
                data,
                truth,
                seed=99,
                epochs=2,
                variant="attention",
                directory=directory,
                rows_per_epoch=64,
                batch_size=32,
            )
            saved.append(torch.load(directory / "model.pt", weights_only=True))
        assert saved[0]["val_mae"] == saved[1]["val_mae"]
        assert saved[0]["best_epoch"] == saved[1]["best_epoch"]
        for name, value in saved[0]["state_dict"].items():
            torch.testing.assert_close(
                value, saved[1]["state_dict"][name], rtol=0, atol=0
            )
    finally:
        torch.set_num_threads(prior_threads)
