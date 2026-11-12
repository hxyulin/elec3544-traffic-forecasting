"""Identity inputs must preserve alignment, causality and the sensor ablation."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from traffic.data import IN_STEPS, REPORT_HORIZONS, MetrLA, split_starts
from traffic.identity import IdentityData, IdentityMLP, gap_age, predict_identity


@pytest.fixture
def dataset():
    speed = (np.arange(360).reshape(120, 3) % 60 + 10).astype(np.float32)
    speed[20:25, 0] = 0
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


def test_gap_age_handles_initial_gaps_and_ignores_future():
    speed = np.array([[0, 20], [0, 0], [30, 0], [0, 40], [50, 0]], dtype=np.float32)
    np.testing.assert_array_equal(
        gap_age(speed), [[1, 0], [2, 1], [0, 2], [1, 0], [0, 1]]
    )
    changed = speed.copy()
    changed[3:] = 0
    np.testing.assert_array_equal(gap_age(speed)[:3], gap_age(changed)[:3])


def test_identity_batch_and_prediction_order(dataset):
    data = IdentityData(dataset, "cpu")
    inputs, y = data.batch("val", np.arange(data.size("val")))
    x, sensor, tow, age = inputs
    np.testing.assert_array_equal(
        sensor, np.tile(np.arange(3), len(dataset.splits["val"]))
    )
    o = np.repeat(dataset.splits["val"] + IN_STEPS - 1, 3)
    expected_tow = dataset.time_of_week[o[:, None] + np.array(REPORT_HORIZONS)]
    np.testing.assert_array_equal(tow, expected_tow)
    np.testing.assert_allclose(
        age, np.log1p(gap_age(dataset.speed))[o, sensor], atol=1e-6
    )
    expected_y = (
        dataset.windows("val")[1][:, np.array(REPORT_HORIZONS) - 1, :]
        .transpose(0, 2, 1)
        .reshape(-1, 3)
    )
    np.testing.assert_array_equal(y, expected_y)
    for identity in (True, False):
        p = predict_identity(
            IdentityMLP(3, use_identity=identity), data, "val", batch_size=7
        )
        persistence = dataset.filled[dataset.splits["val"] + IN_STEPS - 1]
        np.testing.assert_allclose(
            p, np.repeat(persistence[:, None, :], 3, axis=1), atol=1e-5
        )


def test_sensor_identity_is_active_only_when_enabled(dataset, tmp_path):
    torch.manual_seed(1)
    model = IdentityMLP(3, dropout=0).eval()
    torch.nn.init.normal_(model.head.weight)
    inputs, _ = IdentityData(dataset, "cpu").batch("val", np.arange(3))
    x, sensor, tow, age = inputs
    # Hold histories/calendar/age identical, varying only the sensor identity.
    same = (x[:1].expand(3, -1, -1), sensor, tow[:1].expand(3, -1), age[:1].expand(3))
    enabled = model(*same)
    assert not torch.allclose(enabled[0], enabled[1])
    model.use_identity = False
    disabled = model(*same)
    torch.testing.assert_close(disabled[0], disabled[1])
    disabled.sum().backward()
    assert torch.count_nonzero(model.sensor.weight.grad) == 0
    model.use_identity = True
    model.zero_grad()
    model(*same).sum().backward()
    assert torch.count_nonzero(model.sensor.weight.grad) > 0
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)
    restored = IdentityMLP(3, dropout=0).eval()
    restored.load_state_dict(torch.load(path, weights_only=True))
    torch.testing.assert_close(enabled, restored(*same))


def test_future_speeds_do_not_change_identity_inputs(dataset):
    data = IdentityData(dataset, "cpu")
    first_target = dataset.splits["test"][1] + IN_STEPS
    speed, filled = dataset.speed.copy(), dataset.filled.copy()
    speed[first_target:] = 1000
    filled[first_target:] = 1000
    changed = IdentityData(replace(dataset, speed=speed, filled=filled), "cpu")
    rows = np.arange(3) + 3
    for original, altered in zip(
        data.batch("test", rows)[0], changed.batch("test", rows)[0]
    ):
        torch.testing.assert_close(original, altered)
