"""Compare saved test predictions during low-speed and other observed traffic.

Run ``uv run scripts/error_analysis.py`` after ``scripts/evaluate.py``.
The threshold is selected in the training-only EDA. Conditions are
defined by observed target speed, so this describes errors retrospectively;
it does not evaluate a congestion detector. No models are fitted or selected.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from traffic.data import IN_STEPS, REPORT_HORIZONS, ROOT, load_metr_la
from traffic.metrics import masked_errors

OUT = ROOT / "outputs" / "error_analysis"
THRESHOLD_MPH = 40
MODELS = {
    "persistence": "Persistence",
    "historical_average": "Historical average",
    "ridge_lags": "Ridge, lags only",
    "ridge": "Ridge",
    "gradient_boosting": "Gradient boosting",
    "ridge_neighbours": "Ridge + neighbours",
    "gradient_boosting_neighbours": "Gradient boosting + neighbours",
}


def main() -> None:
    selection = json.loads((ROOT / "outputs" / "eda" / "summary.json").read_text())[
        "congestion"
    ]
    if (
        selection["selected_rule"] != "fixed_40_mph"
        or selection["threshold_mph"] != THRESHOLD_MPH
    ):
        raise ValueError("Update condition masks for the selected EDA rule")
    ds = load_metr_la()
    starts = ds.splits["test"]
    horizons = np.array(REPORT_HORIZONS)
    truth = ds.speed[starts[:, None] + IN_STEPS + horizons[None, :] - 1]
    recorded = pd.read_csv(ROOT / "outputs" / "results" / "metrics.csv")
    rows = []
    for model in MODELS:
        with np.load(ROOT / "outputs" / "predictions" / f"{model}.npz") as saved:
            np.testing.assert_array_equal(saved["starts"], starts)
            np.testing.assert_array_equal(saved["horizons"], horizons)
            pred = saved["pred"]
        if pred.shape != truth.shape or not np.isfinite(pred).all():
            raise ValueError(f"{model}: predictions have invalid shape or values")
        for i, horizon in enumerate(horizons):
            y, p = truth[:, i], pred[:, i]
            masks = {
                "all_observed": y > 0,
                "below_40_mph": (y > 0) & (y < THRESHOLD_MPH),
                "at_least_40_mph": y >= THRESHOLD_MPH,
            }
            if masks["all_observed"].sum() != sum(
                masks[k].sum() for k in ("below_40_mph", "at_least_40_mph")
            ):
                raise ValueError("Traffic conditions do not partition observed targets")
            aggregate = masked_errors(p, y)
            old = recorded[
                (recorded["model"] == model)
                & (recorded["split"] == "test")
                & (recorded["horizon"] == horizon)
            ]
            if len(old) != 1:
                raise ValueError(f"{model}: expected one recorded result per horizon")
            for metric in ("mae", "rmse", "mape", "n"):
                np.testing.assert_allclose(
                    aggregate[metric],
                    old.iloc[0][metric],
                    rtol=0,
                    atol=0.000051,
                    err_msg=f"{model}, horizon {horizon}: saved results differ",
                )
            for condition, mask in masks.items():
                rows.append(
                    {
                        "model": model,
                        "minutes": int(horizon * 5),
                        "condition": condition,
                        **masked_errors(p[mask], y[mask]),
                    }
                )

    table = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "condition_metrics.csv", index=False, float_format="%.6f")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
    for ax, condition, title in zip(
        axes,
        ("below_40_mph", "at_least_40_mph"),
        ("Observed target below 40 mph", "Observed target at least 40 mph"),
    ):
        for model, label in MODELS.items():
            part = table[(table.model == model) & (table.condition == condition)]
            ax.plot(part.minutes, part.mae, marker="o", label=label, linewidth=1.5)
        ax.set(title=title, xlabel="Forecast horizon (minutes)", xticks=[15, 30, 60])
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("MAE (mph)")
    axes[1].legend(fontsize=8)
    fig.suptitle("Test errors by traffic condition: training-selected 40 mph threshold")
    fig.tight_layout()
    fig.savefig(OUT / "condition_mae.png", dpi=180)
    plt.close(fig)

    summary = {
        "split": "test",
        "threshold_mph": THRESHOLD_MPH,
        "condition_uses": "observed target speed",
        "threshold_status": "selected by training-only EDA; relative rule retained as sensitivity analysis",
        "missing_targets": "zero speeds excluded from all conditions",
        "verification": f"All {len(MODELS) * len(horizons)} model/horizon aggregates match metrics.csv to its rounding precision; prediction starts, horizons, shape and finiteness checked; condition counts partition observed targets.",
        "limitations": [
            "Descriptive analysis of existing predictions; no model selection or tuning.",
            "Profile models exclude the target calendar week during training; chronological benchmark splits retain overlapping target timestamps at boundaries.",
            "Sliding windows are dependent; counts are forecast-target pairs, not independent samples.",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        table.pivot(index=["model", "condition"], columns="minutes", values="mae")
        .round(3)
        .to_string()
    )
    print(
        f"\nVerified all {len(MODELS) * len(horizons)} aggregate results against metrics.csv."
    )
    print("\nTarget counts by condition:")
    print(
        table[table.model == "persistence"]
        .pivot(index="condition", columns="minutes", values="n")
        .to_string()
    )


if __name__ == "__main__":
    main()
