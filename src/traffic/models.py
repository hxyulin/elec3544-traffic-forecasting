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


def weekly_profile(ds: MetrLA) -> np.ndarray:
    """(2016, N) mean observed training speed per sensor and 5-minute slot of the week."""
    speed = ds.speed[: ds.train_end]
    tow = ds.time_of_week[: ds.train_end]
    observed = speed > 0
    sums = np.zeros((STEPS_PER_WEEK, ds.n_sensors))
    counts = np.zeros((STEPS_PER_WEEK, ds.n_sensors))
    np.add.at(sums, tow, np.where(observed, speed, 0))
    np.add.at(counts, tow, observed)
    sensor_mean = sums.sum(0) / counts.sum(0)
    with np.errstate(invalid="ignore"):
        profile = sums / counts
    # A slot with no observations for a sensor falls back to that sensor's overall mean.
    return np.where(counts > 0, profile, sensor_mean).astype(np.float32)


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


def feature_matrix(
    ds: MetrLA, split: str, horizon: int, profile: np.ndarray, groups: tuple[str, ...]
) -> np.ndarray:
    """One row per (window, sensor), ordered window-major to match y[:, h-1].ravel().

    groups:
      lags     12 most recent speeds (filled) and the share of them that were missing
      calendar time of day and day of week as sin/cos, weekend flag
      profile  weekly-profile speed at the target time and at the origin
      sensor   sensor index, for tree models to treat as a category
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
        cols.append(profile[ds.time_of_week[o + horizon]][:, :, None])
        cols.append(profile[ds.time_of_week[o]][:, :, None])
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
    ):
        self.name = name
        self.groups = groups
        self.make = make
        self.max_train_rows = max_train_rows
        self.models: dict[int, object] = {}

    def _xy(
        self, ds: MetrLA, split: str, horizon: int, max_rows: int | None, seed: int
    ):
        x = feature_matrix(ds, split, horizon, self.profile, self.groups)
        y = targets(ds, split, horizon)
        keep = np.flatnonzero(y > 0)
        if max_rows is not None and len(keep) > max_rows:
            keep = np.sort(
                np.random.default_rng(seed).choice(keep, max_rows, replace=False)
            )
        return x[keep], y[keep]

    def fit(self, ds: MetrLA, horizons: tuple[int, ...]) -> FeatureModel:
        self.profile = weekly_profile(ds)
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


def ridge(name: str, groups: tuple[str, ...]) -> FeatureModel:
    return FeatureModel(
        name, groups, lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    )


def gradient_boosting() -> FeatureModel:
    groups = ("lags", "calendar", "profile", "sensor")
    sensor_col = 12 + 1 + 5 + 2  # position of the sensor column in feature_matrix
    return FeatureModel(
        "gradient_boosting",
        groups,
        lambda: HistGradientBoostingRegressor(
            max_iter=500,
            learning_rate=0.1,
            max_leaf_nodes=63,
            min_samples_leaf=100,
            categorical_features=[sensor_col],
            early_stopping=True,
            n_iter_no_change=20,
            random_state=0,
        ),
        max_train_rows=1_000_000,
    )
