import numpy as np
import pytest

from traffic.data import IN_STEPS, OUT_STEPS, causal_fill, load_metr_la, split_starts
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
