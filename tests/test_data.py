import numpy as np
import pandas as pd
import pytest

from traffic.data import (
    IN_STEPS,
    OUT_STEPS,
    causal_fill,
    load_metr_la,
    prepare_metr_la,
    split_starts,
)
from traffic.metrics import masked_errors
from traffic.models import feature_matrix, targets, weekly_profile


@pytest.fixture(scope="module")
def ds():
    return load_metr_la()


def test_split_sizes_match_dcrnn():
    s = split_starts(34272)
    assert [len(s[k]) for k in ("train", "val", "test")] == [23974, 3425, 6850]


def test_splits_are_chronological(ds):
    tr, va, te = ds.splits["train"], ds.splits["val"], ds.splits["test"]
    assert tr[-1] < va[0] and va[-1] < te[0]
    assert te[-1] + IN_STEPS + OUT_STEPS == len(ds.speed)


def test_windows_align_with_series(ds):
    x, y = ds.windows("test", source="speed")
    s = ds.splits["test"][5]
    np.testing.assert_array_equal(x[5], ds.speed[s : s + IN_STEPS])
    np.testing.assert_array_equal(
        y[5], ds.speed[s + IN_STEPS : s + IN_STEPS + OUT_STEPS]
    )


def test_causal_fill_uses_only_the_past():
    speed = np.array([[0], [50], [0], [0], [60]], dtype=np.float32)
    filled = causal_fill(speed, train_end=5)
    np.testing.assert_array_equal(filled[:, 0], [55, 50, 50, 50, 60])


def test_fill_fallback_ignores_values_after_training_and_preserves_targets():
    speed = np.array([[0], [50], [0], [60], [0], [80]], dtype=np.float32)
    original = speed.copy()
    filled = causal_fill(speed, train_end=4)
    np.testing.assert_array_equal(filled[:, 0], [55, 50, 50, 60, 60, 80])
    changed = speed.copy()
    changed[5] = 100
    np.testing.assert_array_equal(causal_fill(changed, train_end=4)[:5], filled[:5])
    np.testing.assert_array_equal(speed, original)


def test_prepare_tables_aligns_sensors_without_files():
    timestamps = pd.date_range("2012-03-01", periods=50, freq="5min")
    speed = pd.DataFrame(
        {"B": np.arange(50) + 20, "A": np.arange(50) + 40}, index=timestamps
    )
    speed.iloc[2, 0] = 0
    distances = pd.DataFrame({"from": ["A", "B"], "to": ["B", "A"], "cost": [100, 200]})
    locations = pd.DataFrame(
        {"sensor_id": ["A", "B"], "latitude": [1, 2], "longitude": [3, 4]}
    )

    ds = prepare_metr_la(speed, distances, locations)

    assert ds.sensor_ids == ["B", "A"]
    assert ds.locations["latitude"].tolist() == [2, 1]
    assert ds.distance[0, 1] == 200
    assert ds.distance[1, 0] == 100
    assert ds.speed[2, 0] == 0
    assert ds.filled[2, 0] == 21
    x, y = ds.windows("train")
    np.testing.assert_array_equal(x[0], ds.filled[:12])
    np.testing.assert_array_equal(y[0], ds.speed[12:24])


def test_masked_errors_skip_zeros():
    out = masked_errors(np.array([10.0, 99.0, 30.0]), np.array([12.0, 0.0, 30.0]))
    assert out["mae"] == pytest.approx(1.0) and out["n"] == 2


def test_features_line_up_with_targets(ds):
    profile = weekly_profile(ds)
    x = feature_matrix(ds, "val", 3, profile, ("lags",))
    y = targets(ds, "val", 3)
    assert x.shape == (len(ds.splits["val"]) * ds.n_sensors, IN_STEPS + 1)
    # Row (window 7, sensor 11): last lag is the filled speed at the origin, target 3 steps later.
    s, n = 7, 11
    row = s * ds.n_sensors + n
    origin = ds.splits["val"][s] + IN_STEPS - 1
    assert x[row, IN_STEPS - 1] == ds.filled[origin, n]
    assert y[row] == ds.speed[origin + 3, n]


def test_profile_uses_training_period_only(ds):
    changed = ds.speed.copy()
    changed[ds.train_end :] = 1.0
    ds2 = type(ds)(**{**ds.__dict__, "speed": changed})
    np.testing.assert_array_equal(weekly_profile(ds), weekly_profile(ds2))
