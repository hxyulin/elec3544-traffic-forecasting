"""Neural alignment, masking, sensor independence, and checkpoint invariants."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from traffic.data import IN_STEPS, REPORT_HORIZONS, MetrLA, split_starts
from traffic.neural import NeuralData, SensorGRU, masked_mae, predict


@pytest.fixture
def ds():
    speed = (np.arange(120 * 3).reshape(120, 3) % 60 + 10).astype(np.float32)
    speed[30, 1] = 0
    return MetrLA(
        timestamps=pd.date_range("2012-03-05", periods=120, freq="5min"),
        sensor_ids=["a", "b", "c"],
        speed=speed,
        filled=np.where(speed > 0, speed, 40),
        adjacency=np.eye(3),
        distance=np.eye(3),
        locations=pd.DataFrame(),
        splits=split_starts(120),
    )


def test_batches_align_with_existing_windows_and_masks(ds):
    data = NeuralData(ds, "cpu")
    for split in ds.splits:
        x, y = data.batch(split, np.arange(data.size(split)))
        existing_x, existing_y = ds.windows(split)
        expected = (
            existing_y[:, np.array(REPORT_HORIZONS) - 1, :]
            .transpose(0, 2, 1)
            .reshape(-1, 3)
        )
        np.testing.assert_array_equal(y.numpy(), expected)
        recovered = x[:, :, 0].numpy() * data.std + data.mean
        np.testing.assert_allclose(
            recovered, existing_x.transpose(0, 2, 1).reshape(-1, IN_STEPS), atol=1e-5
        )
        starts = ds.splits[split]
        mask = ds.speed[starts[:, None] + np.arange(IN_STEPS)] > 0
        np.testing.assert_array_equal(
            x[:, :, 1].numpy(), mask.transpose(0, 2, 1).reshape(-1, IN_STEPS)
        )


def test_missing_targets_have_no_loss_or_gradient():
    p = torch.tensor([[7.0, 500.0, 11.0]], requires_grad=True)
    loss = masked_mae(p, torch.tensor([[5.0, 0.0, 10.0]]))
    assert loss.item() == pytest.approx(1.5)
    loss.backward()
    np.testing.assert_allclose(p.grad.numpy(), [[0.5, 0.0, 0.5]])
    empty = torch.ones(2, 3, requires_grad=True)
    zero = masked_mae(empty, torch.zeros(2, 3))
    zero.backward()
    assert zero.item() == 0 and torch.count_nonzero(empty.grad) == 0


def test_zero_head_equals_persistence_and_output_order(ds):
    data = NeuralData(ds, "cpu")
    p = predict(SensorGRU(8), data, "test", batch_size=7)
    last = ds.filled[ds.splits["test"] + IN_STEPS - 1]
    np.testing.assert_allclose(p, np.repeat(last[:, None, :], 3, axis=1), atol=1e-5)


def test_test_future_values_do_not_change_input(ds):
    data = NeuralData(ds, "cpu")
    # The first test target overlaps the benchmark training period. Use the next
    # origin so perturbations stay outside the training normalizer as well.
    first_target = ds.splits["test"][1] + IN_STEPS
    speed, filled = ds.speed.copy(), ds.filled.copy()
    speed[first_target:] = 1000
    filled[first_target:] = 1000
    changed = NeuralData(replace(ds, speed=speed, filled=filled), "cpu")
    rows = ds.n_sensors + np.arange(ds.n_sensors)
    torch.testing.assert_close(
        data.batch("test", rows)[0], changed.batch("test", rows)[0]
    )


def test_sensor_sequences_are_independent_and_checkpoint_reloads(tmp_path):
    torch.manual_seed(1)
    model = SensorGRU(8)
    torch.nn.init.normal_(model.head.weight)
    x = torch.randn(3, 12, 7)
    before = model(x)
    changed = x.clone()
    changed[1:] = 100
    torch.testing.assert_close(before[0], model(changed)[0])
    checkpoint = tmp_path / "gru.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = SensorGRU(8)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    torch.testing.assert_close(before, restored(x))
