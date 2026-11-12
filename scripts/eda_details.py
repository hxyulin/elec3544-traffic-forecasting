"""Training-only congestion and lead/lag analysis, plus descriptive outage audits."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from traffic.data import IN_STEPS, OUT_STEPS
from traffic.models import weekly_profile


def runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def analyse(ds, out):
    train = ds.speed[: ds.train_end]
    observed = train > 0
    free = np.nanpercentile(np.where(observed, train, np.nan), 85, axis=0)
    thresholds = {
        "fixed_40_mph": np.full(ds.n_sensors, 40.0),
        "relative_60pct_p85": 0.6 * free,
    }
    rows, per_sensor = (
        [],
        pd.DataFrame({"sensor_id": ds.sensor_ids, "free_flow_p85_mph": free}),
    )
    vmax = 100 * max(
        ((observed & (train < t)).sum(0) / observed.sum(0)).max()
        for t in thresholds.values()
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    for ax, (rule, threshold) in zip(axes, thresholds.items()):
        flag = observed & (train < threshold)
        share = flag.sum(0) / observed.sum(0)
        per_sensor[rule + "_threshold_mph"] = threshold
        per_sensor[rule + "_share"] = share
        rows.append(
            {
                "rule": rule,
                "flagged_readings": int(flag.sum()),
                "observed_readings": int(observed.sum()),
                "share": float(flag.sum() / observed.sum()),
                "sensors_flagged": int(flag.any(0).sum()),
                "median_sensor_share": float(np.median(share)),
                "threshold_min_mph": float(threshold.min()),
                "threshold_max_mph": float(threshold.max()),
            }
        )
        pts = ax.scatter(
            ds.locations.longitude,
            ds.locations.latitude,
            c=share * 100,
            vmin=0,
            vmax=vmax,
            cmap="viridis",
            s=18,
        )
        ax.set(title=rule.replace("_", " "), xlabel="Longitude", ylabel="Latitude")
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.set_aspect(1 / np.cos(np.deg2rad(ds.locations.latitude.mean())))
        fig.colorbar(
            pts, ax=ax, label="Observed training readings flagged (%)", shrink=0.7
        )
    fig.tight_layout()
    fig.savefig(out / "congestion_rules.png", dpi=160)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(out / "congestion_rules.csv", index=False)
    per_sensor.to_csv(out / "congestion_by_sensor.csv", index=False)
    fixed = observed & (train < 40)
    relative = observed & (train < 0.6 * free)
    congestion = {
        "comparison": rows,
        "disagreement_share": float(
            np.logical_xor(fixed, relative).sum() / observed.sum()
        ),
        "selected_rule": "fixed_40_mph",
        "threshold_mph": 40,
        "selection_uses": "training data only",
        "reason": "Use a common absolute low-speed definition for interpretable cross-sensor errors and continuity with baseline analysis. The relative rule is retained as a sensitivity comparison; neither rule is ground-truth congestion.",
    }

    all_missing = ds.missing.all(1)
    outage_rows = [
        {
            "start": str(ds.timestamps[a]),
            "end_inclusive": str(ds.timestamps[b - 1]),
            "steps": b - a,
            "hours": (b - a) / 12,
        }
        for a, b in runs(all_missing)
    ]
    pd.DataFrame(outage_rows).to_csv(out / "network_outages.csv", index=False)
    sensor_rows = []
    for n, sensor in enumerate(ds.sensor_ids):
        intervals = runs(ds.missing[:, n])
        a, b = max(intervals, key=lambda p: p[1] - p[0]) if intervals else (0, 0)
        sensor_rows.append(
            {
                "sensor_id": sensor,
                "missing_share": float(ds.missing[:, n].mean()),
                "longest_steps": b - a,
                "longest_hours": (b - a) / 12,
                "start": str(ds.timestamps[a]),
                "end_inclusive": str(ds.timestamps[max(a, b - 1)]),
                "network_offline_steps_in_run": int(all_missing[a:b].sum()),
            }
        )
    pd.DataFrame(sensor_rows).sort_values("longest_steps", ascending=False).to_csv(
        out / "sensor_outages.csv", index=False
    )
    periods = []
    for split, starts in ds.splits.items():
        a, b = int(starts[0]) + IN_STEPS, int(starts[-1]) + IN_STEPS + OUT_STEPS
        missing = ds.missing[a:b]
        offline = all_missing[a:b]
        periods.append(
            {
                "split": split,
                "steps": b - a,
                "missing_share": float(missing.mean()),
                "network_offline_steps": int(offline.sum()),
                "network_offline_share": float(offline.mean()),
                "partial_missing_contribution": float(
                    (missing & ~offline[:, None]).mean()
                ),
                "missing_share_when_network_not_offline": float(
                    missing[~offline].mean()
                ),
            }
        )
    pd.DataFrame(periods).to_csv(out / "missing_decomposition.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(
        [r["split"] for r in periods],
        [100 * r["network_offline_share"] for r in periods],
        label="All sensors offline",
    )
    ax.bar(
        [r["split"] for r in periods],
        [100 * r["partial_missing_contribution"] for r in periods],
        bottom=[100 * r["network_offline_share"] for r in periods],
        label="Other missing readings",
    )
    ax.set(
        ylabel="Readings missing (%)",
        title="Missingness within each split's target period",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "missing_decomposition.png", dpi=160)
    plt.close(fig)

    # One strongest outgoing neighbour per sensor. Positive lag means source leads destination.
    strength = ds.adjacency.copy()
    np.fill_diagonal(strength, 0)
    residual = train - weekly_profile(ds)[ds.time_of_week[: ds.train_end]]
    lagrows = []
    for i in range(ds.n_sensors):
        j = int(strength[i].argmax())
        if strength[i, j] <= 0:
            continue
        scores = []
        for lag in range(-12, 13):
            a = slice(max(0, -lag), min(len(train), len(train) - lag))
            b = slice(max(0, lag), min(len(train), len(train) + lag))
            keep = observed[a, i] & observed[b, j]
            x, y = residual[a, i][keep], residual[b, j][keep]
            c = (
                float(np.corrcoef(x, y)[0, 1])
                if len(x) >= 100 and x.std() > 0 and y.std() > 0
                else np.nan
            )
            scores.append((lag, c, int(keep.sum())))
        valid = [r for r in scores if np.isfinite(r[1])]
        if not valid:
            continue
        # Prefer smaller absolute lag on an exact tie.
        lag, c, n = max(valid, key=lambda r: (r[1], -abs(r[0])))
        zero = scores[12][1]
        lagrows.append(
            {
                "source": ds.sensor_ids[i],
                "destination": ds.sensor_ids[j],
                "weight": float(strength[i, j]),
                "road_distance_m": float(ds.distance[i, j]),
                "best_lag_minutes": lag * 5,
                "best_correlation": c,
                "zero_lag_correlation": zero,
                "gain_over_zero": c - zero,
                "paired_observations": n,
            }
        )
    table = pd.DataFrame(lagrows)
    table.to_csv(out / "neighbour_lags.csv", index=False)
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3))
    a.hist(table.best_lag_minutes, bins=np.arange(-62.5, 65, 5))
    a.set(
        xlabel="Best lag (minutes; positive = source leads)",
        ylabel="Directed pairs",
        title="Strongest outgoing neighbour per sensor",
    )
    b.scatter(table.zero_lag_correlation, table.gain_over_zero, s=12, alpha=0.6)
    b.set(
        xlabel="Correlation at zero lag",
        ylabel="Gain at selected lag",
        title="Exploratory lag search, training observations",
    )
    fig.tight_layout()
    fig.savefig(out / "neighbour_lags.png", dpi=160)
    plt.close(fig)
    split_audit = []
    splits = list(ds.splits)
    for left, right in zip(splits, splits[1:]):
        left_end = int(ds.splits[left][-1]) + IN_STEPS + OUT_STEPS
        right_start = int(ds.splits[right][0]) + IN_STEPS
        split_audit.append(
            {
                "left": left,
                "right": right,
                "shared_target_timestamps": max(0, left_end - right_start),
                "shared_minutes": 5 * max(0, left_end - right_start),
            }
        )
    return {
        "congestion": congestion,
        "missing_decomposition": periods,
        "outage_intervals": len(outage_rows),
        "longest_network_outage_hours": max(r["hours"] for r in outage_rows),
        "lag_analysis": {
            "pairs": len(table),
            "positive_lag_pairs": int((table.best_lag_minutes > 0).sum()),
            "zero_lag_pairs": int((table.best_lag_minutes == 0).sum()),
            "negative_lag_pairs": int((table.best_lag_minutes < 0).sum()),
            "median_gain_over_zero": float(table.gain_over_zero.median()),
            "method": "Training-only weekly residuals; pairwise observed readings; strongest outgoing edge per sensor; Pearson correlation at -60 to +60 minutes in 5-minute steps.",
            "limitations": "Graph direction is source-to-destination road distance, not independently verified upstream/downstream flow. Lag search is descriptive and has no significance test. Pairs can be reciprocal and are not independent. Missing values are excluded, not forward-filled.",
        },
        "split_boundary_audit": split_audit,
    }
