"""Check baseline stability and write the classical-model report from saved results."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from traffic.data import ROOT
from traffic.reporting import markdown_table, model_label

OUT = ROOT / "outputs" / "results"


def main():
    metrics = pd.read_csv(OUT / "metrics.csv")
    metrics = metrics[
        metrics.model.isin(
            [
                "persistence",
                "historical_average",
                "ridge_lags",
                "ridge",
                "ridge_neighbours",
                "gradient_boosting",
                "gradient_boosting_neighbours",
            ]
        )
    ]
    previous = pd.read_csv(OUT / "previous_metrics.csv")
    keys = ["model", "split", "horizon"]
    baseline = ["persistence", "historical_average"]
    old = previous[previous.model.isin(baseline)].set_index(keys).sort_index()
    new = metrics[metrics.model.isin(baseline)].set_index(keys).sort_index()
    pd.testing.assert_index_equal(old.index, new.index)
    np.testing.assert_allclose(
        old[["mae", "rmse", "mape", "n"]],
        new[["mae", "rmse", "mape", "n"]],
        rtol=0,
        atol=0.000051,
    )
    (OUT / "baseline_verification.json").write_text(
        json.dumps(
            {
                "models": baseline,
                "splits": ["val", "test"],
                "rows_checked": len(new),
                "metrics": ["mae", "rmse", "mape", "n"],
                "atol": 0.000051,
                "result": "unchanged at CSV precision",
            },
            indent=2,
        )
        + "\n"
    )
    test = metrics[metrics.split == "test"]
    table = test.pivot(index="model", columns="minutes", values=["mae", "rmse"])
    table.columns = [
        f"{metric.upper()} {minutes} min" for metric, minutes in table.columns
    ]
    table = table.round(3).reset_index()
    abl = pd.read_csv(OUT / "ablation.csv")
    atest = abl[abl.split == "test"]
    abtable = atest.pivot(
        index=["model", "feature_set"], columns="minutes", values=["mae", "rmse"]
    )
    abtable.columns = [
        f"{metric.upper()} {minutes} min" for metric, minutes in abtable.columns
    ]
    abtable = abtable.round(3).reset_index()
    abtable["feature_set"] = pd.Categorical(
        abtable["feature_set"],
        categories=["lags", "+calendar", "+profile", "+neighbours", "+profile (leaky)"],
        ordered=True,
    )
    abtable = abtable.sort_values(["model", "feature_set"])
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, group in test.groupby("model", sort=False):
        ax.plot(group.minutes, group.mae, marker="o", label=model_label(name))
    ax.set(
        xlabel="Forecast horizon (minutes)",
        ylabel="Test MAE (mph)",
        xticks=[15, 30, 60],
    )
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "classical_mae.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    order = ["lags", "+calendar", "+profile", "+neighbours"]
    for ax, model in zip(axes, ["ridge", "gradient_boosting"]):
        for minutes in (15, 30, 60):
            g = (
                atest[(atest.model == model) & (atest.minutes == minutes)]
                .set_index("feature_set")
                .reindex(order)
            )
            ax.plot(range(4), g.mae, marker="o", label=f"{minutes} min")
        ax.set(
            title=model_label(model),
            xticks=range(4),
            xticklabels=order,
            ylabel="Test MAE (mph)",
        )
        ax.tick_params(axis="x", labelrotation=20)
        ax.grid(alpha=0.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "ablation_mae.png", dpi=160)
    plt.close(fig)
    params = json.loads((OUT / "best_params.json").read_text())
    paragraphs = []
    for model in ("ridge", "gradient_boosting"):
        g = atest[atest.model == model].set_index(["feature_set", "minutes"])
        gains = [
            float(g.loc[("+profile", h), "mae"] - g.loc[("+neighbours", h), "mae"])
            for h in (15, 30, 60)
        ]
        leak = [
            float(g.loc[("+profile (leaky)", h), "mae"] - g.loc[("+profile", h), "mae"])
            for h in (15, 30, 60)
        ]
        paragraphs.append(
            f"For {model.replace('_', ' ')}, adding neighbours reduces test MAE by {', '.join(f'{v:.3f}' for v in gains)} mph at 15, 30, and 60 minutes respectively. Negative values mean the added features worsen error. Replacing the leaky profile with the corrected profile reduces MAE by {', '.join(f'{v:.3f}' for v in leak)} mph at the same horizons. These are measured differences for these settings, not significance tests."
        )
    text = (
        """# Baselines and classical models

We compare persistence, historical average, ridge regression, and histogram gradient boosting on the existing METR-LA benchmark windows. Inputs cover one hour; reported horizons are 15, 30, and 60 minutes. Models share the same splits, observed-target masks, and scoring functions. MAE and RMSE are in mph; zero targets are excluded.

## Features and profile correction

The feature sequence adds calendar variables, weekly-profile speeds, and neighbour measurements to twelve lagged speeds and the input-window missing share. Calendar features encode target time of day and week with sine/cosine terms plus a weekend indicator. Profile features give the weekly mean at the origin and target slots. Each training row excludes the target's entire Monday-to-Sunday calendar week from both profile estimates and fallback means, including forecasts crossing a week boundary. Empty slots use the sensor mean outside that week, then the global observed mean outside that week. Validation and test use the full training profile.

Neighbour features use the three strongest connections after taking the maximum of both directed weights. They contain latest speeds, 15-minute changes, and a weighted average of all connected sensors. Absent neighbours fall back to the sensor's own measurements. Boosting also receives categorical sensor ID in every ablation; ridge is pooled across sensors without sensor ID. Therefore comparisons within each model isolate feature additions, while cross-model comparisons also reflect different feature representations.

## Validation selection

Ridge alpha is selected separately for each feature set from 0.1, 1, 10, 100, and 1000. Boosting searches max_leaf_nodes in 31, 63, 127; learning_rate in 0.05, 0.1; and min_samples_leaf in 100, 400. Selection minimizes mean validation MAE over the three horizons. Boosting settings are selected using lags, calendar, profile, and sensor ID, then held fixed across ablations. This measures feature additions under a shared boosting configuration, rather than each feature set's separately optimized performance.

Boosting samples at most one million observed training rows per horizon, with the horizon step as the sampling seed. Early stopping uses at most 200,000 chronological-validation rows sampled with seed 100 plus the horizon. The maximum iteration count is 500 and patience is 20; boosting random_state is 0. The same validation period supplies stopping and selection, so its error is a selection statistic. Test scores are used for reporting only.

Selected settings:

```json
"""
        + json.dumps(params, indent=2)
        + "\n```\n\n## Test results\n\n"
        + markdown_table(table)
        + "\n\n![Classical model errors](../outputs/results/classical_mae.png)\n\nPersistence and historical-average results match the saved pre-integration scores on validation and test at all three horizons, for MAE, RMSE, MAPE, and counts, within CSV rounding precision. The check is recorded in `baseline_verification.json`.\n\n## Feature ablations\n\n"
        + markdown_table(abtable)
        + "\n\n![Feature additions](../outputs/results/ablation_mae.png)\n\n"
        + "\n\n".join(paragraphs)
        + """

The leaky-profile runs use the corrected profile model's selected settings. They isolate the feature construction difference at those settings; they are not a rerun of the entire old selection procedure. Original metrics are retained separately in `previous_metrics.csv`.

## Evaluation limits

Adjacent benchmark split target periods overlap by eleven timestamps, or 55 minutes. We preserve this protocol for comparability and do not claim fully disjoint target periods. Training profiles may contain training-period observations later than a particular training row; cross-fitting removes that row's calendar week but is not a rolling-origin simulation. Neighbour features use only information available at the forecast origin.

Overlapping windows, shared sensors, and reciprocal graph links make errors dependent. No significance claim follows from small MAE differences alone. The comparison establishes classical reference results for the planned temporal and graph neural models. It does not yet answer whether graph neural modelling improves on a matched temporal architecture. Traffic-condition results are saved separately in `outputs/error_analysis/`.

Reproduce with `uv run scripts/evaluate.py --tune`, `uv run scripts/evaluate.py`, `uv run scripts/evaluate.py --ablation`, `uv run scripts/error_analysis.py`, and `uv run scripts/summarize_results.py`. Package versions and selected settings are recorded in `outputs/results/run_metadata.json`.
"""
    )
    (ROOT / "docs" / "classical_models.md").write_text(text)
    print(table.to_string(index=False))
    print("Baseline stability verified; report and figures written.")


if __name__ == "__main__":
    main()
