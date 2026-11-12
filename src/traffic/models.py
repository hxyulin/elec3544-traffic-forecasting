"""Baselines and classical models. Each model predicts (S, len(horizons), N) speeds."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import IN_STEPS, STEPS_PER_DAY, STEPS_PER_WEEK, MetrLA


def origins(ds: MetrLA, split: str) -> np.ndarray:
    """Index of the last input step of each window: the time the forecast is made."""
    return ds.splits[split] + IN_STEPS - 1


class Persistence:
    """Every future speed equals the last observed speed."""

    name = "persistence"

    def fit(self, ds: MetrLA) -> Persistence:
        return self

    def predict(self, ds: MetrLA, split: str, horizons: tuple[int, ...]) -> np.ndarray:
        last = ds.filled[origins(ds, split)]
        return np.repeat(last[:, None, :], len(horizons), axis=1)


def _profile_sums(ds: MetrLA) -> tuple[np.ndarray, np.ndarray]:
    """Sum and count of observed training speeds per time-of-week slot and sensor."""
    speed = ds.speed[: ds.train_end]
    tow = ds.time_of_week[: ds.train_end]
    observed = speed > 0
    sums = np.zeros((STEPS_PER_WEEK, ds.n_sensors))
    counts = np.zeros((STEPS_PER_WEEK, ds.n_sensors))
    np.add.at(sums, tow, np.where(observed, speed, 0))
    np.add.at(counts, tow, observed)
    return sums, counts


def weekly_profile(ds: MetrLA) -> np.ndarray:
    """(2016, N) mean observed training speed per sensor and 5-minute slot of the week."""
    sums, counts = _profile_sums(ds)
    sensor_mean = sums.sum(0) / counts.sum(0)
    with np.errstate(invalid="ignore"):
        profile = sums / counts
    # A slot with no observations for a sensor falls back to that sensor's overall mean.
    return np.where(counts > 0, profile, sensor_mean).astype(np.float32)


def weekly_profile_loo(ds: MetrLA) -> np.ndarray:
    """(train_end, N) leave-own-week-out profile for each training time step.

    Each slot of the week occurs once per week, so removing a step's own reading from
    its slot average is the same as averaging over the other weeks. Training rows use
    this so the feature never contains the row's own target. Validation and test rows
    keep the full profile from weekly_profile.
    """
    sums, counts = _profile_sums(ds)
    speed = ds.speed[: ds.train_end]
    tow = ds.time_of_week[: ds.train_end]
    observed = speed > 0
    rest_sum = sums[tow] - np.where(observed, speed, 0)
    rest_count = counts[tow] - observed
    sensor_mean = sums.sum(0) / counts.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        profile = rest_sum / rest_count
    return np.where(rest_count > 0, profile, sensor_mean).astype(np.float32)


def neighbour_index(ds: MetrLA, k: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """For each sensor, its k most strongly connected neighbours and their weights.

    The graph is directed, so a pair's strength is the larger weight over both directions.
    Uses only ds.adjacency (road distances), so it carries no information from the speeds.
    """
    strength = np.maximum(ds.adjacency, ds.adjacency.T).astype(np.float64)
    np.fill_diagonal(strength, 0.0)
    idx = np.argsort(-strength, axis=1, kind="stable")[:, :k]
    return idx, np.take_along_axis(strength, idx, axis=1)


class HistoricalAverage:
    """Mean training speed for the same sensor at the same time of week."""

    name = "historical_average"

    def fit(self, ds: MetrLA) -> HistoricalAverage:
        self.profile = weekly_profile(ds)
        return self

    def predict(self, ds: MetrLA, split: str, horizons: tuple[int, ...]) -> np.ndarray:
        o = origins(ds, split)
        return np.stack(
            [self.profile[ds.time_of_week[o + h]] for h in horizons], axis=1
        )


N_FEATURES = {"lags": 13, "calendar": 5, "profile": 2, "neighbours": 7, "sensor": 1}
GROUP_ORDER = ("lags", "calendar", "profile", "neighbours", "sensor")


def n_features(groups: tuple[str, ...]) -> int:
    return sum(N_FEATURES[g] for g in GROUP_ORDER if g in groups)


def feature_matrix(
    ds: MetrLA,
    split: str,
    horizon: int,
    profile: np.ndarray,
    groups: tuple[str, ...],
    *,
    profile_loo: np.ndarray | None = None,
) -> np.ndarray:
    """One row per (window, sensor), ordered window-major to match y[:, h-1].ravel().

    groups (columns always appear in this order):
      lags        12 most recent speeds (filled) and the share of them that were missing
      calendar    time of day and day of week as sin/cos, weekend flag
      profile     weekly-profile speed at the target time and at the origin. For the train
                  split with profile_loo given, the leave-own-week-out profile is used.
      neighbours  latest speed and 15-minute change of the 3 most strongly connected
                  sensors, plus the weight-averaged latest speed of all neighbours
      sensor      sensor index, for tree models to treat as a category
    """
    o = origins(ds, split)
    s, n = len(o), ds.n_sensors
    steps = o[:, None] - np.arange(IN_STEPS - 1, -1, -1)  # (S, 12), oldest first
    cols: list[np.ndarray] = []

    def per_window(v: np.ndarray) -> np.ndarray:
        return np.broadcast_to(v[:, None, None], (s, n, 1))

    if "lags" in groups:
        cols.append(ds.filled[steps].transpose(0, 2, 1))
        cols.append(ds.missing[steps].mean(axis=1)[:, :, None])
    if "calendar" in groups:
        tow = ds.time_of_week[o + horizon]
        day_angle = 2 * np.pi * (tow % STEPS_PER_DAY) / STEPS_PER_DAY
        week_angle = 2 * np.pi * tow / STEPS_PER_WEEK
        weekend = (tow // STEPS_PER_DAY >= 5).astype(np.float32)
        for v in (
            np.sin(day_angle),
            np.cos(day_angle),
            np.sin(week_angle),
            np.cos(week_angle),
            weekend,
        ):
            cols.append(per_window(v))
    if "profile" in groups:
        if split == "train" and profile_loo is not None:
            at_target, at_origin = profile_loo[o + horizon], profile_loo[o]
        else:
            at_target = profile[ds.time_of_week[o + horizon]]
            at_origin = profile[ds.time_of_week[o]]
        cols.append(at_target[:, :, None])
        cols.append(at_origin[:, :, None])
    if "neighbours" in groups:
        idx, w = neighbour_index(ds)
        last, earlier = ds.filled[o], ds.filled[o - 3]  # (S, N)
        has = (w > 0)[
            None
        ]  # sensors with fewer than 3 neighbours fall back to themselves
        nb_last = np.where(has, last[:, idx], last[:, :, None])
        nb_change = nb_last - np.where(has, earlier[:, idx], earlier[:, :, None])
        strength = np.maximum(ds.adjacency, ds.adjacency.T).astype(np.float64)
        np.fill_diagonal(strength, 0.0)
        total = strength.sum(1)
        weights = strength / np.where(total > 0, total, 1.0)[:, None]
        avg = np.where(total > 0, last @ weights.T, last)
        cols.extend([nb_last, nb_change, avg[:, :, None]])
    if "sensor" in groups:
        cols.append(np.broadcast_to(np.arange(n)[None, :, None], (s, n, 1)))
    return np.concatenate(cols, axis=2, dtype=np.float32).reshape(s * n, -1)


def targets(ds: MetrLA, split: str, horizon: int) -> np.ndarray:
    return ds.speed[origins(ds, split) + horizon].ravel()


class FeatureModel:
    """One regressor per horizon, trained on all sensors pooled. Missing targets are dropped."""

    def __init__(
        self,
        name: str,
        groups: tuple[str, ...],
        make,
        max_train_rows: int | None = None,
        fix_leak: bool = True,
    ):
        self.name = name
        self.groups = groups
        self.make = make
        self.max_train_rows = max_train_rows
        self.fix_leak = fix_leak  # False reproduces the original leaky profile feature
        self.models: dict[int, object] = {}
        self.profile_loo = None

    def _xy(
        self, ds: MetrLA, split: str, horizon: int, max_rows: int | None, seed: int
    ):
        x = feature_matrix(
            ds, split, horizon, self.profile, self.groups, profile_loo=self.profile_loo
        )
        y = targets(ds, split, horizon)
        keep = np.flatnonzero(y > 0)
        if max_rows is not None and len(keep) > max_rows:
            keep = np.sort(
                np.random.default_rng(seed).choice(keep, max_rows, replace=False)
            )
        return x[keep], y[keep]

    def fit(self, ds: MetrLA, horizons: tuple[int, ...]) -> FeatureModel:
        self.profile = weekly_profile(ds)
        self.profile_loo = weekly_profile_loo(ds) if self.fix_leak else None
        for h in horizons:
            x, y = self._xy(ds, "train", h, self.max_train_rows, seed=h)
            model = self.make()
            if isinstance(model, HistGradientBoostingRegressor):
                # Early stopping on the chronological validation split, not a random holdout.
                xv, yv = self._xy(ds, "val", h, 200_000, seed=100 + h)
                model.fit(x, y, X_val=xv, y_val=yv)
            else:
                model.fit(x, y)
            self.models[h] = model
        return self

    def predict(self, ds: MetrLA, split: str, horizons: tuple[int, ...]) -> np.ndarray:
        preds = []
        for h in horizons:
            x = feature_matrix(ds, split, h, self.profile, self.groups)
            preds.append(self.models[h].predict(x).reshape(-1, ds.n_sensors))
        return np.stack(preds, axis=1).astype(np.float32)


# Starting settings from the first version, used until evaluate.py --tune picks better ones.
RIDGE_ALPHA = 1.0
GB_PARAMS = {"max_leaf_nodes": 63, "learning_rate": 0.1, "min_samples_leaf": 100}


def ridge(
    name: str,
    groups: tuple[str, ...],
    alpha: float = RIDGE_ALPHA,
    fix_leak: bool = True,
) -> FeatureModel:
    return FeatureModel(
        name,
        groups,
        lambda: make_pipeline(StandardScaler(), Ridge(alpha=alpha)),
        fix_leak=fix_leak,
    )


def gradient_boosting(
    name: str = "gradient_boosting",
    groups: tuple[str, ...] = ("lags", "calendar", "profile", "sensor"),
    params: dict | None = None,
    fix_leak: bool = True,
) -> FeatureModel:
    """Sensor is always the last column, so its index is the feature count minus one."""
    p = {**GB_PARAMS, **(params or {})}
    sensor_col = n_features(groups) - 1
    return FeatureModel(
        name,
        groups,
        lambda: HistGradientBoostingRegressor(
            max_iter=500,
            categorical_features=[sensor_col],
            early_stopping=True,
            n_iter_no_change=20,
            random_state=0,
            **p,
        ),
        max_train_rows=1_000_000,
        fix_leak=fix_leak,
    )
