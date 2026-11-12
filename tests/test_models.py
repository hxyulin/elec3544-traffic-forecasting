"""Checks for the leak fix and neighbour features on a small synthetic series."""

import numpy as np
import pandas as pd
import pytest

from traffic.data import IN_STEPS, STEPS_PER_WEEK, MetrLA, causal_fill, split_starts
from traffic.models import (
    feature_matrix,
    n_features,
    neighbour_index,
    ridge,
    targets,
    weekly_profile,
    weekly_profile_loo,
)

N = 6


@pytest.fixture(scope="module")
def ds():
    rng = np.random.default_rng(0)
    t = 4 * STEPS_PER_WEEK
    speed = rng.uniform(20, 70, size=(t, N)).astype(np.float32)
    speed[rng.random((t, N)) < 0.05] = 0
    splits = split_starts(t)
    train_end = int(splits["train"][-1]) + 24
    adj = np.zeros((N, N), dtype=np.float32)
    adj[0, 1], adj[1, 0], adj[0, 2], adj[3, 0] = 0.9, 0.5, 0.4, 0.8  # sensor 0: 3 links
    adj[4, 5] = 0.7  # sensors 4 and 5: one link each
    return MetrLA(
        timestamps=pd.date_range("2012-03-05", periods=t, freq="5min"),  # a Monday
        sensor_ids=[str(i) for i in range(N)],
        speed=speed,
        filled=causal_fill(speed, train_end),
        adjacency=adj,
        distance=np.full((N, N), np.inf),
        locations=pd.DataFrame(),
        splits=splits,
    )


def test_loo_profile_excludes_own_reading(ds):
    full, loo = weekly_profile(ds), weekly_profile_loo(ds)
    t, n = 100, 2
    tow = ds.time_of_week[t]
    same_slot = np.flatnonzero(ds.time_of_week[: ds.train_end] == tow)
    others = [s for s in same_slot if s != t and ds.speed[s, n] > 0]
    assert loo[t, n] == pytest.approx(np.mean(ds.speed[others, n]), rel=1e-5)
    assert loo[t, n] != pytest.approx(full[tow, n], rel=1e-6)


def test_train_rows_use_loo_but_val_and_test_keep_full_profile(ds):
    full, loo = weekly_profile(ds), weekly_profile_loo(ds)
    x_fix = feature_matrix(ds, "train", 3, full, ("profile",), profile_loo=loo)
    x_old = feature_matrix(ds, "train", 3, full, ("profile",))
    assert not np.allclose(x_fix, x_old)
    for split in ("val", "test"):
        a = feature_matrix(ds, split, 3, full, ("profile",), profile_loo=loo)
        b = feature_matrix(ds, split, 3, full, ("profile",))
        np.testing.assert_array_equal(a, b)


def test_fixed_profile_does_not_encode_the_target(ds):
    """With the leak, the profile at the target time correlates with the target more strongly."""
    full, loo = weekly_profile(ds), weekly_profile_loo(ds)
    y = targets(ds, "train", 3)
    keep = y > 0
    old = feature_matrix(ds, "train", 3, full, ("profile",))[keep, 0]
    new = feature_matrix(ds, "train", 3, full, ("profile",), profile_loo=loo)[keep, 0]
    assert np.corrcoef(old, y[keep])[0, 1] > np.corrcoef(new, y[keep])[0, 1]


def test_neighbour_index_picks_strongest_links(ds):
    idx, w = neighbour_index(ds)
    assert list(idx[0]) == [1, 3, 2]  # strengths 0.9, 0.8, 0.4
    assert w[4, 0] == pytest.approx(0.7) and w[4, 1] == 0  # only one real neighbour


def test_neighbour_features_shape_and_fallback(ds):
    full = weekly_profile(ds)
    x = feature_matrix(ds, "val", 3, full, ("neighbours",))
    assert x.shape == (len(ds.splits["val"]) * N, 7)
    assert n_features(("lags", "calendar", "profile", "neighbours", "sensor")) == 28
    o = ds.splits["val"][0] + IN_STEPS - 1
    row = lambda sensor: x[sensor]
    assert row(0)[0] == ds.filled[o, 1]  # strongest neighbour of sensor 0 is sensor 1
    assert (
        row(4)[1] == ds.filled[o, 4]
    )  # no second neighbour: falls back to its own speed
    assert row(0)[3] == ds.filled[o, 1] - ds.filled[o - 3, 1]


def test_ridge_trains_with_every_feature_group(ds):
    m = ridge("r", ("lags", "calendar", "profile", "neighbours"))
    m.fit(ds, (3,))
    pred = m.predict(ds, "test", (3,))
    assert pred.shape == (len(ds.splits["test"]), 1, N) and np.isfinite(pred).all()


def test_boosting_treats_sensor_as_last_column(ds):
    from traffic.models import gradient_boosting

    groups = ("lags", "calendar", "profile", "neighbours", "sensor")
    m = gradient_boosting(groups=groups, params={"max_leaf_nodes": 15})
    m.fit(ds, (3,))
    x = feature_matrix(ds, "val", 3, m.profile, groups)
    assert x.shape[1] == n_features(groups)
    assert set(np.unique(x[:, -1])) == set(range(N))
    assert np.isfinite(m.predict(ds, "val", (3,))).all()
