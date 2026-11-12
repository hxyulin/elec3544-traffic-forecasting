"""Exploratory analysis of METR-LA. Writes figures and a summary to outputs/eda/.

Every statistic that could inform a modelling choice uses the training period only.

    uv run scripts/eda.py
"""

from __future__ import annotations

import json

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from eda_details import analyse
from matplotlib.colors import LinearSegmentedColormap

from traffic.data import ROOT, STEPS_PER_DAY, load_metr_la
from traffic.models import weekly_profile

OUT = ROOT / "outputs" / "eda"
BLUE, GRAY, MUTED = "#2a78d6", "#a8a7a2", "#52514e"
BLUE_RAMP = LinearSegmentedColormap.from_list(
    "blue", ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"]
)
CONGESTED_MPH = 40

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 9,
        "axes.edgecolor": GRAY,
        "axes.labelcolor": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e6e5e0",
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "lines.linewidth": 1.6,
    }
)


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def longest_run(mask: np.ndarray) -> int:
    best = run = 0
    for m in mask:
        run = run + 1 if m else 0
        best = max(best, run)
    return best


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ds = load_metr_la()
    speed, missing, ts = ds.speed, ds.missing, ds.timestamps
    train = slice(0, ds.train_end)
    observed_train = speed[train][~missing[train]]

    # Missing data: per sensor, and over time.
    per_sensor = missing.mean(axis=0)
    daily = pd.Series(missing.mean(axis=1), index=ts).resample("D").mean()
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3), width_ratios=[1, 1.6])
    a.bar(
        np.arange(ds.n_sensors), np.sort(per_sensor)[::-1] * 100, width=1.0, color=BLUE
    )
    a.set(
        xlabel="Sensor (sorted)",
        ylabel="Readings missing (%)",
        title="Missing share per sensor",
    )
    b.plot(daily.index, daily * 100, color=BLUE)
    for split in ("val", "test"):
        b.axvline(
            ts[ds.splits[split][0] + 11], color=MUTED, linewidth=0.8, linestyle="--"
        )
        b.annotate(
            split,
            (ts[ds.splits[split][0] + 11], 1),
            xycoords=("data", "axes fraction"),
            xytext=(3, -10),
            textcoords="offset points",
            color=MUTED,
        )
    b.set(ylabel="Readings missing (%)", title="Missing share per day")
    b.tick_params(axis="x", labelrotation=30)
    save(fig, "missing")

    # Weekly cycle: median over sensors of the training-period weekly profile.
    profile = weekly_profile(ds)
    hours = np.arange(len(profile)) / 12
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.fill_between(
        hours,
        *np.percentile(profile, [25, 75], axis=1),
        color=BLUE,
        alpha=0.18,
        linewidth=0,
    )
    ax.plot(hours, np.median(profile, axis=1), color=BLUE)
    ax.set_xticks(
        np.arange(0, 168, 24) + 12, ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    )
    ax.set_xticks(np.arange(0, 169, 24), minor=True)
    ax.grid(False, axis="x")
    ax.grid(True, axis="x", which="minor")
    ax.tick_params(axis="x", which="major", length=0)
    ax.set(
        ylabel="Speed (mph)",
        title="Mean speed by time of week: median sensor, band = middle 50% of sensors",
    )
    save(fig, "weekly_profile")

    # Speed distribution.
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(
        observed_train,
        bins=np.arange(0, 72, 2),
        color=BLUE,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.axvline(CONGESTED_MPH, color=MUTED, linewidth=0.8, linestyle="--")
    ax.set(
        xlabel="Speed (mph)",
        ylabel="Readings",
        title="Observed speeds, training period",
    )
    save(fig, "speed_distribution")

    # Temporal autocorrelation of the filled series, median over sensors, up to one day.
    filled = ds.filled[train]
    z = (filled - filled.mean(0)) / filled.std(0)
    lags = np.arange(1, STEPS_PER_DAY + 1)
    acf = np.array([np.median((z[:-k] * z[k:]).mean(0)) for k in lags])
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(lags * 5 / 60, acf, color=BLUE)
    ax.axvline(1, color=MUTED, linewidth=0.8, linestyle="--")
    ax.set(
        xlabel="Lag (hours)",
        ylabel="Autocorrelation",
        title="Speed autocorrelation, median sensor",
    )
    save(fig, "autocorrelation")

    # Spatial dependence: correlation of residuals from the weekly cycle vs road distance.
    resid = filled - profile[ds.time_of_week[train]]
    corr = np.corrcoef(resid.T)
    off_diag = ~np.eye(ds.n_sensors, dtype=bool)
    linked = off_diag & np.isfinite(ds.distance)
    d_km = ds.distance[linked] / 1000
    c_linked = corr[linked]
    edges = np.array([0, 1, 2, 3, 5, 8, 12, 20, 30, 50])
    which = np.digitize(d_km, edges) - 1
    centres, medians = [], []
    for i in range(len(edges) - 1):
        sel = which == i
        if sel.sum() >= 20:
            centres.append(np.median(d_km[sel]))
            medians.append(np.median(c_linked[sel]))
    unlinked = corr[off_diag & ~np.isfinite(ds.distance)]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.scatter(d_km, c_linked, s=3, color=GRAY, alpha=0.25, linewidths=0)
    ax.plot(centres, medians, color=BLUE, marker="o", markersize=4)
    ax.axhline(np.median(unlinked), color=MUTED, linewidth=0.8, linestyle="--")
    ax.annotate(
        "median, pairs with no listed road path",
        (0, np.median(unlinked)),
        xytext=(2, -11),
        textcoords="offset points",
        color=MUTED,
    )
    ax.set(
        xlabel="Road distance (km)",
        ylabel="Correlation of residual speed",
        title="Nearby sensors deviate from their weekly pattern together",
    )
    save(fig, "spatial_correlation")

    # Sensor map with graph edges.
    lon, lat = ds.locations["longitude"].to_numpy(), ds.locations["latitude"].to_numpy()
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, j in zip(*np.nonzero(ds.adjacency * off_diag)):
        ax.plot(
            [lon[i], lon[j]],
            [lat[i], lat[j]],
            color=GRAY,
            linewidth=0.4,
            alpha=0.6,
            zorder=1,
        )
    sensor_mean = np.nanmean(np.where(missing[train], np.nan, speed[train]), axis=0)
    pts = ax.scatter(
        lon,
        lat,
        c=sensor_mean,
        cmap=BLUE_RAMP.reversed(),
        s=18,
        edgecolors="white",
        linewidths=0.5,
        zorder=2,
    )
    fig.colorbar(pts, ax=ax, label="Mean training speed (mph)", shrink=0.6)
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.1))
    ax.set(xlabel="Longitude", ylabel="Latitude", title="Sensors and graph edges")
    ax.set_aspect(1 / np.cos(np.deg2rad(lat.mean())))
    ax.grid(False)
    save(fig, "sensor_map")

    sensor_outage = [longest_run(missing[:, n]) for n in range(ds.n_sensors)]
    summary = {
        "time_steps": int(speed.shape[0]),
        "sensors": ds.n_sensors,
        "first_timestamp": str(ts[0]),
        "last_timestamp": str(ts[-1]),
        "split_windows": {k: len(v) for k, v in ds.splits.items()},
        "split_first_target_time": {
            k: str(ts[v[0] + 12]) for k, v in ds.splits.items()
        },
        "missing_share_all": round(float(missing.mean()), 4),
        "missing_share_by_split_period": {
            k: round(float(missing[v[0] + 12 : v[-1] + 24].mean()), 4)
            for k, v in ds.splits.items()
        },
        "sensors_over_20pct_missing": int((per_sensor > 0.2).sum()),
        "steps_with_all_sensors_missing": int(missing.all(axis=1).sum()),
        "longest_outage_hours_median_sensor": round(
            float(np.median(sensor_outage)) * 5 / 60, 1
        ),
        "longest_outage_hours_max": round(float(max(sensor_outage)) * 5 / 60, 1),
        "graph_edges_excluding_self": int(((ds.adjacency > 0) & off_diag).sum()),
        "train_mean_speed_mph": round(float(observed_train.mean()), 2),
        "train_std_speed_mph": round(float(observed_train.std()), 2),
        f"train_share_below_{CONGESTED_MPH}_mph": round(
            float((observed_train < CONGESTED_MPH).mean()), 4
        ),
        "autocorrelation_15_30_60_min": [
            round(float(acf[k - 1]), 3) for k in (3, 6, 12)
        ],
        "residual_corr_median_within_2km": round(
            float(np.median(c_linked[d_km < 2])), 3
        ),
        "residual_corr_median_no_road_path": round(float(np.median(unlinked)), 3),
    }
    summary.update(analyse(ds, OUT))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
