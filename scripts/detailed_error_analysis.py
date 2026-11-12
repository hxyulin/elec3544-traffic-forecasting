"""Audit saved test errors by sensor, target time, missing inputs, and speed changes.

    uv run scripts/detailed_error_analysis.py

No models are fitted or selected. Representative sensors use training congestion
ranks; failure cases are explicitly selected after observing test errors.
"""

from __future__ import annotations

import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from traffic.analysis import grouped_errors, load_predictions, verify_partition
from traffic.data import IN_STEPS, REPORT_HORIZONS, ROOT, load_metr_la
from traffic.metrics import masked_errors
from summarize_results import markdown

OUT = ROOT / "outputs/error_analysis/detailed"
FOCUS = ["persistence", "gradient_boosting_neighbours", "gru", "graph_gru"]
LABELS = {
    "persistence": "Persistence",
    "gradient_boosting_neighbours": "Neighbour boosting",
    "gru": "Temporal GRU",
    "graph_gru": "Graph GRU",
}
TRACE = ["persistence", "gradient_boosting_neighbours", "gru", "graph_gru"]
COLORS = {
    "persistence": "#888888",
    "gradient_boosting_neighbours": "#2677b8",
    "gru": "#26864b",
    "graph_gru": "#c83c35",
}
INPUT_NAMES = ["0 missing", "1–3 missing", "4–11 missing", "12 missing"]
CHANGE_NAMES = ["within 10 mph", "drop ≥10 mph", "rise ≥10 mph", "origin missing"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ds = load_metr_la()
    starts = ds.splits["test"]
    hs = np.array(REPORT_HORIZONS)
    steps = starts[:, None] + IN_STEPS + hs[None, :] - 1
    truth = ds.speed[steps]
    origin_steps = starts + IN_STEPS - 1
    origin = ds.speed[origin_steps]
    missing_count = ds.missing[starts[:, None] + np.arange(IN_STEPS)].sum(1)
    input_codes = np.select(
        [missing_count == 0, missing_count <= 3, missing_count < 12],
        [0, 1, 2],
        default=3,
    )
    recorded = pd.read_csv(ROOT / "outputs/results/metrics.csv")
    tables = {
        k: [] for k in ("sensor", "hour", "day_type", "input_missing", "speed_change")
    }
    predictions = {}
    checks = 0
    for name in recorded.model.unique():
        pred = load_predictions(
            ROOT / f"outputs/predictions/{name}.npz", starts, hs, truth.shape
        )
        if name in FOCUS:
            predictions[name] = pred
        for i, h in enumerate(hs):
            y, p = truth[:, i], pred[:, i]
            aggregate = masked_errors(p, y)
            expected = recorded[
                (recorded.model == name)
                & (recorded.split == "test")
                & (recorded.horizon == h)
            ]
            if len(expected) != 1:
                raise ValueError(f"{name}: missing or duplicate aggregate")
            np.testing.assert_allclose(
                [aggregate[k] for k in ("mae", "rmse", "mape", "n")],
                expected.iloc[0][["mae", "rmse", "mape", "n"]].astype(float),
                atol=0.000051,
                rtol=0,
            )
            times = ds.timestamps[steps[:, i]]
            change = np.where(
                origin == 0,
                3,
                np.where(y - origin <= -10, 1, np.where(y - origin >= 10, 2, 0)),
            )
            partitions = {
                "sensor": (np.arange(ds.n_sensors)[None, :], ds.sensor_ids),
                "hour": (times.hour.to_numpy()[:, None], list(range(24))),
                "day_type": (
                    np.asarray(times.dayofweek >= 5, dtype=np.int64)[:, None],
                    ["weekday", "weekend"],
                ),
                "input_missing": (input_codes, INPUT_NAMES),
                "speed_change": (change, CHANGE_NAMES),
            }
            for key, (codes, labels) in partitions.items():
                rows = grouped_errors(p, y, np.asarray(codes), len(labels))
                verify_partition(rows, aggregate)
                checks += 1
                tables[key].extend(
                    {
                        "model": name,
                        "minutes": int(h * 5),
                        key: labels[r["group"]],
                        **{k: v for k, v in r.items() if k != "group"},
                    }
                    for r in rows
                )
    frames = {k: pd.DataFrame(v) for k, v in tables.items()}
    for key, frame in frames.items():
        frame.to_csv(OUT / f"{key}_metrics.csv", index=False, float_format="%.8f")

    # Time-of-day axes use target timestamps, not forecast-origin timestamps.
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    for ax, minutes in zip(axes, (15, 30, 60)):
        for name in FOCUS:
            part = frames["hour"][
                (frames["hour"].model == name) & (frames["hour"].minutes == minutes)
            ]
            ax.plot(part.hour, part.mae, label=LABELS[name])
        ax.set(
            title=f"{minutes}-minute horizon",
            xlabel="Target hour (dataset clock)",
            xticks=[0, 6, 12, 18, 23],
        )
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("MAE (mph)")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "hour_mae.png", dpi=160)
    plt.close(fig)

    sensor = frames["sensor"]
    sensor60 = sensor[sensor.minutes == 60]
    spatial = sensor60.pivot(index="sensor", columns="model", values="mae").reindex(
        ds.sensor_ids
    )
    gain = spatial.gru - spatial.graph_gru
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    values = [spatial.gradient_boosting_neighbours, spatial.graph_gru, gain]
    vmax = float(max(values[0].max(), values[1].max()))
    gmax = max(float(np.abs(gain).max()), 0.001)
    for ax, v, title in zip(
        axes,
        values,
        ["Neighbour boosting MAE", "Graph GRU MAE", "Temporal minus graph MAE"],
    ):
        difference = title.startswith("Temporal")
        pts = ax.scatter(
            ds.locations.longitude,
            ds.locations.latitude,
            c=v,
            cmap="RdBu" if difference else "viridis",
            s=20,
            vmin=-gmax if difference else 0,
            vmax=gmax if difference else vmax,
        )
        ax.set(title=title, xlabel="Longitude")
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.set_aspect(1 / np.cos(np.deg2rad(ds.locations.latitude.mean())))
        fig.colorbar(pts, ax=ax, label="mph", shrink=0.7)
    axes[0].set_ylabel("Latitude")
    fig.suptitle("60-minute forecasts; positive difference favours graph")
    fig.tight_layout()
    fig.savefig(OUT / "sensor_mae.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, title, labels in zip(
        axes,
        ["input_missing", "speed_change"],
        [
            "Missing readings in the 12-step input",
            "Change from observed origin to target",
        ],
        [INPUT_NAMES, CHANGE_NAMES],
    ):
        frame = frames[key]
        frame = frame[frame.minutes == 60]
        for offset, name in enumerate(FOCUS):
            part = frame[frame.model == name].set_index(key).reindex(labels)
            ax.bar(
                np.arange(len(labels)) + (offset - 1.5) * 0.2,
                part.mae,
                width=0.2,
                label=LABELS[name],
            )
        ax.set(
            title=title,
            xticks=np.arange(len(labels)),
            xticklabels=labels,
            ylabel="60-minute MAE (mph)",
        )
        ax.tick_params(axis="x", rotation=20)
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "failure_regimes.png", dpi=160)
    plt.close(fig)

    # These sensor ranks are fixed from training data, before inspecting test errors.
    train = ds.speed[: ds.train_end]
    train_low = ((train > 0) & (train < 40)).sum(0) / (train > 0).sum(0)
    order = np.argsort(train_low, kind="stable")
    cases = []
    dates = pd.DatetimeIndex(ds.timestamps[steps[:, 2]].normalize().unique())
    for quantile in (0.5, 0.9):
        n = int(order[int((ds.n_sensors - 1) * quantile)])
        chosen = None
        for day in dates:
            ix = np.flatnonzero(ds.timestamps[steps[:, 2]].normalize() == day)
            if (
                day.dayofweek < 5
                and len(ix) == 288
                and (truth[ix, 2, n] > 0).mean() >= 0.8
            ):
                chosen = (day, ix)
                break
        if chosen is None:
            raise ValueError(
                f"No full weekday with 80% observed targets for {ds.sensor_ids[n]}"
            )
        cases.append(
            {
                "kind": "representative",
                "selection": f"training congestion rank {quantile:.0%}; earliest full test weekday with ≥80% observed targets",
                "sensor": ds.sensor_ids[n],
                "sensor_index": n,
                "training_low_speed_share": float(train_low[n]),
                "day": str(chosen[0].date()),
                "window_indices": chosen[1].tolist(),
            }
        )

    trace_rows = []

    def draw_trace(ax, case, hindex, indices):
        n = case["sensor_index"]
        indices = np.asarray(indices)
        time = ds.timestamps[steps[indices, hindex]]
        y = truth[indices, hindex, n]
        observed = y > 0
        ax.plot(
            time,
            np.where(observed, y, np.nan),
            color="black",
            linewidth=1.5,
            label="Observed",
        )
        for name in TRACE:
            p = predictions[name][indices, hindex, n]
            ax.plot(
                time,
                np.where(observed, p, np.nan),
                color=COLORS[name],
                linewidth=1,
                label=LABELS[name],
            )
        ax.fill_between(
            time,
            0,
            1,
            where=~observed,
            transform=ax.get_xaxis_transform(),
            color="#dddddd",
            alpha=0.5,
            step="mid",
            label="Missing target",
        )
        ax.set_xlim(time[0], time[-1])
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.set(ylabel="Speed (mph)", xlabel="Target time (dataset clock)")
        ax.grid(alpha=0.2)
        for row, idx in enumerate(indices):
            trace_rows.append(
                {
                    "case_id": case["case_id"],
                    "sensor": ds.sensor_ids[n],
                    "minutes": int(hs[hindex] * 5),
                    "origin_time": str(ds.timestamps[origin_steps[idx]]),
                    "target_time": str(time[row]),
                    "true_speed": float(y[row]) if observed[row] else np.nan,
                    "observed": bool(observed[row]),
                    "input_missing_count": int(missing_count[idx, n]),
                    **{
                        name: float(predictions[name][idx, hindex, n]) for name in FOCUS
                    },
                }
            )

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
    for row, case in enumerate(cases):
        case["case_id"] = f"representative_{row + 1}"
        for col, hindex in enumerate((0, 2)):
            day = pd.Timestamp(case["day"])
            ix = np.flatnonzero(ds.timestamps[steps[:, hindex]].normalize() == day)
            draw_trace(axes[row, col], case, hindex, ix)
            axes[row, col].set_title(
                f"Sensor {case['sensor']}, {case['day']}, {hs[hindex] * 5} min"
            )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    fig.tight_layout(rect=[0, 0.075, 1, 1])
    fig.savefig(OUT / "representative_traces.png", dpi=160)
    plt.close(fig)

    # Post-hoc test cases illustrate failures; they are not a representative sample.
    error = {
        name: np.abs(p[:, 2].astype(float) - truth[:, 2])
        for name, p in predictions.items()
    }
    scores = [
        ("largest_graph_error", error["graph_gru"]),
        (
            "largest_gap_to_boosting",
            error["graph_gru"] - error["gradient_boosting_neighbours"],
        ),
        ("largest_gap_to_temporal", error["graph_gru"] - error["gru"]),
    ]
    allowed = truth[:, 2] > 0
    failures = []
    fig, axes = plt.subplots(3, 1, figsize=(11, 8))
    for ax, (kind, score) in zip(axes, scores):
        ranked = np.where(allowed, score, -np.inf)
        flat = int(ranked.argmax())
        w, n = np.unravel_index(flat, ranked.shape)
        if not np.isfinite(ranked[w, n]):
            raise ValueError("No eligible failure case")
        center = int(steps[w, 2])
        ix = np.flatnonzero(np.abs(steps[:, 2] - center) <= 36)
        case = {
            "case_id": kind,
            "kind": "post_hoc_failure",
            "selection": "maximum remaining test score in this category; exclude target times within two hours of earlier cases across all sensors",
            "score_definition": kind,
            "score_mph": float(score[w, n]),
            "sensor": ds.sensor_ids[n],
            "sensor_index": int(n),
            "window_index": int(w),
            "target_step": center,
            "origin_time": str(ds.timestamps[origin_steps[w]]),
            "target_time": str(ds.timestamps[center]),
            "true_speed": float(truth[w, 2, n]),
            "origin_speed": float(origin[w, n]),
            "input_missing_count": int(missing_count[w, n]),
            "prediction": {name: float(p[w, 2, n]) for name, p in predictions.items()},
        }
        failures.append(case)
        draw_trace(ax, case, 2, ix)
        ax.axvline(
            ds.timestamps[center], color="#666666", linestyle="--", linewidth=0.8
        )
        ax.scatter(
            [ds.timestamps[center]],
            [truth[w, 2, n]],
            s=35,
            facecolors="none",
            edgecolors="black",
            zorder=5,
        )
        ax.set_title(
            f"{kind.replace('_', ' ')}: sensor {ds.sensor_ids[n]}, {ds.timestamps[center]} (60 min)"
        )
        allowed[np.abs(steps[:, 2] - center) <= 24] = False
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    fig.tight_layout(rect=[0, 0.065, 1, 1])
    fig.savefig(OUT / "failure_traces.png", dpi=160)
    plt.close(fig)
    pd.DataFrame(trace_rows).to_csv(
        OUT / "trace_values.csv", index=False, float_format="%.8f"
    )
    manifest = {
        "representative_cases": cases,
        "failure_cases": failures,
        "trace_policy": "Predictions are issued at the recorded origin and plotted at their target times; missing ground truth appears as gaps. Only observed targets are scored. Trace CSV retains predictions even when ground truth is missing.",
        "limitations": [
            "Representative sensors are chosen by training rank; dates use test coverage, not forecast error.",
            "Failure cases are chosen using test errors, for illustration only; no models or settings are selected.",
            "The two-hour failure exclusion reduces duplicate event displays but does not establish independence.",
        ],
    }
    (OUT / "cases.json").write_text(json.dumps(manifest, indent=2) + "\n")
    verification = {
        "models": list(recorded.model.unique()),
        "horizons_minutes": (hs * 5).tolist(),
        "aggregate_checks": len(recorded.model.unique()) * len(hs),
        "partition_checks": checks,
        "partitions": list(frames),
        "counts": "observed forecast-target pairs",
        "time_grouping": "target timestamp as provided by dataset; no timezone conversion",
        "speed_change_definition": "Target minus observed raw origin speed; drop ≤−10, rise ≥10 mph. Missing origins form a separate group.",
        "empty_groups": "n=0 and NaN metrics",
        "csv_precision": "8 decimal places; partition checks use unrounded scores",
        "recomposition": "Counts, count-weighted MAE/MAPE and RMSE reconstructed from count-weighted squared subgroup RMSE match pooled results within 1e-9 mph/percentage points.",
    }
    (OUT / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    write_report(frames, manifest, verification)
    print(json.dumps(verification, indent=2))


def write_report(frames, manifest, verification):
    hour = frames["hour"]
    sensor = frames["sensor"]
    s60 = sensor[sensor.minutes == 60].pivot(
        index="sensor", columns="model", values="mae"
    )
    hour_graph = hour[(hour.model == "graph_gru") & (hour.minutes == 60)]
    peak = hour_graph.loc[hour_graph.mae.idxmax()]
    trough = hour_graph.loc[hour_graph.mae.idxmin()]
    gains = s60.gru - s60.graph_gru
    overview = []
    for name in FOCUS:
        v = s60[name]
        overview.append(
            {
                "Model": LABELS[name],
                "Sensor MAE median": round(float(v.median()), 3),
                "Sensor MAE 90th percentile": round(float(v.quantile(0.9)), 3),
                "Sensor MAE maximum": round(float(v.max()), 3),
            }
        )
    missing = frames["input_missing"]
    changes = frames["speed_change"]

    def table(frame, key):
        t = (
            frame[(frame.minutes == 60) & frame.model.isin(FOCUS)]
            .pivot(index=key, columns="model", values="mae")
            .reindex(
                index=INPUT_NAMES if key == "input_missing" else CHANGE_NAMES,
                columns=FOCUS,
            )
            .round(3)
        )
        t.columns = [LABELS[n] for n in t.columns]
        return markdown(t.reset_index())

    availability = missing[
        (missing.model == "graph_gru") & (missing.minutes == 60)
    ].set_index("input_missing")
    input_temporal = missing[
        (missing.model == "gru") & (missing.minutes == 60)
    ].set_index("input_missing")
    full_share = float(availability.loc["0 missing", "n"] / availability.n.sum())
    full_gain = float(
        input_temporal.loc["0 missing", "mae"] - availability.loc["0 missing", "mae"]
    )
    missing_deltas = availability.mae - input_temporal.mae
    counts = changes[(changes.model == "graph_gru") & (changes.minutes == 60)][
        ["speed_change", "n"]
    ].copy()
    failure_rows = []
    for c in manifest["failure_cases"]:
        failure_rows.append(
            {
                "Case": c["case_id"],
                "Sensor": c["sensor"],
                "Target": c["target_time"],
                "Observed mph": round(c["true_speed"], 2),
                "Missing inputs": c["input_missing_count"],
                **{LABELS[n]: round(c["prediction"][n], 2) for n in TRACE},
            }
        )
    report = f"""# Final forecast error analysis

We analyse the saved test predictions for all ten models without fitting or selecting models. Every prediction file is checked against the test starts, horizons, expected shape, and finite values. All {verification["aggregate_checks"]} model/horizon aggregates reproduce `metrics.csv`. Each of {verification["partition_checks"]} subgroup partitions reproduces the pooled count, MAE, RMSE, and MAPE from its observed targets. Partition checks use unrounded scores; CSV tables round to eight decimal places. Full tables are in `outputs/error_analysis/detailed/`.

The main figures compare persistence, neighbour boosting, the temporal GRU, and the graph GRU. Other classical models and the separately trained disabled graph remain in the CSV tables. The disabled graph duplicates the temporal GRU, so it is not drawn as another curve. Horizon labels are lead times from the final input to the target. Hours and dates use the dataset timestamps as supplied, with no timezone conversion. Zero targets are excluded from every score.

## Error by time of day

The graph model's 60-minute MAE is highest at target hour {int(peak.hour):02d}:00 ({peak.mae:.3f} mph) and lowest at {int(trough.hour):02d}:00 ({trough.mae:.3f} mph). These hours describe this test period and are not selected deployment schedules. `hour_metrics.csv` includes every model, horizon, hour, and observed count. `day_type_metrics.csv` provides weekday/weekend scores. Hours are grouped by target time, so a 60-minute forecast in the 08:00 group was issued one hour earlier.

![Errors by target hour](../outputs/error_analysis/detailed/hour_mae.png)

## Error across sensors

At 60 minutes, graph filtering improves sensor MAE relative to the temporal GRU for {int((gains > 0).sum())} of {len(gains)} sensors and worsens it for {int((gains < 0).sum())}. The median sensor gain is {gains.median():.4f} mph. These counts give each sensor equal weight, whereas aggregate metrics weight observed target pairs. They do not establish statistical significance or independent replications. Sensor percentiles below are unweighted across sensors and are not pooled errors.

{markdown(pd.DataFrame(overview))}

The maps align sensors by ID with their dataset coordinates. The first two maps share a colour scale. The difference map has a symmetric scale; positive values favour the graph model. Per-sensor observed counts are available in `sensor_metrics.csv`, so errors on sensors with little data can be identified.

![Errors and graph gains across sensors](../outputs/error_analysis/detailed/sensor_mae.png)

## Missing inputs and rapid speed changes

Input availability counts missing readings in the sensor's own twelve-step history. All four groups partition observed targets: no missing inputs, 1–3 missing inputs, 4–11 missing inputs, or all twelve missing. These groups describe input availability; they do not identify outage causes. Correlations between missingness and error are not causal effects.

60-minute MAE by input availability:

{table(missing, "input_missing")}

Fully observed inputs account for {full_share:.2%} of observed 60-minute targets. In that group, graph filtering reduces temporal-GRU MAE by {full_gain:.3f} mph. For histories with 1–3, 4–11, and all twelve readings missing, graph MAE increases by {", ".join(f"{missing_deltas.loc[label]:.3f}" for label in INPUT_NAMES[1:])} mph respectively. The small aggregate gain therefore coexists with worse performance in each missing-input group. These groups have different traffic and sensor compositions, so this does not isolate a causal effect of missing data.

Speed-change groups compare the observed raw origin speed with the observed target. Drops of at least 10 mph and rises of at least 10 mph are separated from changes strictly within 10 mph. A missing raw origin forms its own group, even though models can use filled inputs. The drop/rise definitions mechanically imply at least 10 mph persistence error when the origin is observed. The groups use future observed targets and therefore describe forecast errors retrospectively; they are not available categories at forecast time and do not assess a detector.

{table(changes, "speed_change")}

Observed 60-minute target counts, common to all models:

{markdown(counts)}

![Missing inputs and speed-change errors](../outputs/error_analysis/detailed/failure_regimes.png)

## Representative traces

We fix two sensors using their training-only low-speed shares: ranks 50% and 90% in the sorted sensor list. For each sensor, the date is the first complete test weekday with at least 80% observed targets at 60 minutes. This coverage rule does not use model errors. The chosen sensors are {manifest["representative_cases"][0]["sensor"]} and {manifest["representative_cases"][1]["sensor"]}. They illustrate two training-defined traffic profiles and are not a random or exhaustive sample of sensors.

The plots show 15- and 60-minute forecasts at their target timestamps. Every plotted value was forecast at the recorded earlier origin. Missing ground truth produces shaded gaps in both observations and displayed predictions; the CSV retains all prediction values. `trace_values.csv` contains the exact origins, targets, input missing counts, observations, and four model predictions.

![Training-rank-selected sensors on their first eligible weekdays](../outputs/error_analysis/detailed/representative_traces.png)

## Failure cases selected after observing test errors

We deliberately select three 60-minute cases: the largest graph-model absolute error; the largest remaining excess absolute error over neighbour boosting; and the largest remaining excess error over the temporal GRU. After each choice, target times within two hours are excluded across all sensors for later choices. This limits repeated displays of one event but does not make cases independent. Selections use observed test errors for illustration only. They do not support estimates of how often these failures occur and do not change any model or setting.

{markdown(pd.DataFrame(failure_rows))}

The largest graph-error case and the largest gap to the temporal model both have all twelve input readings missing. Their missing raw origins are not treated as zero speeds by the models: the pipeline carries earlier observations forward. This is a reason to report input availability alongside aggregate errors. It does not identify whether the observed target reflects an incident, recovery, or measurement problem.

Each panel spans up to three hours either side of the selected target, limited by test coverage. The vertical line and circle mark the selected target. The temporal GRU is shown explicitly because the final case selects a failure relative to it. Grey shading marks missing target readings. A sharp change or missing history is an observed association, not evidence of an incident or operational cause.

![Post-hoc failure examples with surrounding observations](../outputs/error_analysis/detailed/failure_traces.png)

## Interpretation and limits

Aggregate scores, traffic-condition errors, hourly profiles, and sensor distributions answer different questions. The graph-versus-temporal difference is small in the main experiment, while neighbour boosting remains the strongest measured system overall. The low-speed and rapid-change subsets help explain why an overall improvement need not translate into better forecasts during congestion. Use the measured subgroup tables and traces when describing where a model helps and where it fails.

Target periods share eleven timestamps at each benchmark split boundary. Sliding windows, linked sensors, repeated targets across horizons, and shared events produce dependent errors. Group counts are forecast-target pairs, not independent samples; no confidence interval or significance claim is made. Findings apply to this METR-LA test period, one neural seed, and the recorded training budgets. Sensors, hours, and worst-case examples are not used to retune models. Repeated seeds, independent periods, and a purged-split sensitivity study would be needed for stronger generalization claims.

## Reproduction

```sh
uv run scripts/error_analysis.py
uv run scripts/detailed_error_analysis.py
```

Generate each model's saved predictions first using the evaluation and checkpoint commands in the README. `cases.json` records trace-selection rules and all selected errors; `verification.json` records aggregate and partition checks. Empty groups retain a zero count and missing metrics rather than fabricated zero error.
"""
    (ROOT / "docs/error_analysis.md").write_text(report)


if __name__ == "__main__":
    main()
