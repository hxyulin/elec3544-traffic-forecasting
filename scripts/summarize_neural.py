"""Write the temporal baseline report and comparison figures from saved experiments."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from traffic.data import ROOT
from traffic.reporting import markdown_table, model_label

OUT = ROOT / "outputs" / "neural"


def main():
    metrics = pd.read_csv(ROOT / "outputs" / "results" / "metrics.csv")
    selected = [
        "persistence",
        "gradient_boosting",
        "gradient_boosting_neighbours",
        "gru",
    ]
    test = metrics[(metrics.split == "test") & metrics.model.isin(selected)]
    table = test.pivot(index="model", columns="minutes", values=["mae", "rmse"])
    table.columns = [
        f"{metric.upper()} {minutes} min" for metric, minutes in table.columns
    ]
    table = table.round(3).reset_index()
    conditions = pd.read_csv(
        ROOT / "outputs" / "error_analysis" / "condition_metrics.csv"
    )
    low = conditions[
        (conditions.condition == "below_40_mph") & conditions.model.isin(selected)
    ]
    ctable = (
        low.pivot(index="model", columns="minutes", values="mae").round(3).reset_index()
    )
    ctable.columns = ["Model", "15 min MAE", "30 min MAE", "60 min MAE"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, data, title in zip(
        axes, [test, low], ["All observed targets", "Observed targets below 40 mph"]
    ):
        for name, part in data.groupby("model", sort=False):
            ax.plot(part.minutes, part.mae, marker="o", label=model_label(name))
        ax.set(
            title=title,
            xlabel="Forecast horizon (minutes)",
            ylabel="Test MAE (mph)",
            xticks=[15, 30, 60],
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "comparison.png", dpi=160)
    plt.close(fig)
    metadata = json.loads((OUT / "run_metadata.json").read_text())
    tuning = pd.read_csv(OUT / "tuning.csv")
    gru = test[test.model == "gru"].set_index("minutes")
    statements = []
    for reference in ["persistence", "gradient_boosting_neighbours"]:
        ref = test[test.model == reference].set_index("minutes")
        delta = ref.mae - gru.mae
        if (delta >= 0).all() or (delta <= 0).all():
            direction = "lower" if (delta >= 0).all() else "higher"
            statements.append(
                f"GRU MAE is {', '.join(f'{abs(delta.loc[h]):.3f}' for h in (15, 30, 60))} mph {direction} than {reference.replace('_', ' ')} at 15, 30, and 60 minutes respectively."
            )
        else:
            statements.append(
                f"MAE differences (reference minus GRU) against {reference.replace('_', ' ')} are {', '.join(f'{delta.loc[h]:.3f}' for h in (15, 30, 60))} mph. Positive values favour GRU."
            )
    low_gru = low[low.model == "gru"].set_index("minutes")
    low_persistence = low[low.model == "persistence"].set_index("minutes")
    if (low_gru.mae > low_persistence.mae).all():
        statements.append(
            "The aggregate improvement over persistence does not extend to low-speed targets: GRU has higher MAE in that subset at every horizon. Report both conditions when assessing its usefulness."
        )
    report = f"""# Temporal neural baseline

The temporal baseline uses a single-layer GRU with {metadata["training_args"]["hidden_size"]} hidden units, shared across all 207 sensors. Each forecast uses only the sensor's own twelve five-minute input readings. The seven input channels are training-normalized filled speed, an observation flag, sine/cosine encodings of time of day and week, and a weekend flag. Calendar channels describe the input timestamps. No sensor ID, weekly-profile feature, or neighbouring sensor measurement is supplied.

The final hidden state feeds a linear head with three direct outputs for 15, 30, and 60 minutes. Outputs are residuals around the last input speed. The head starts at zero, so the untrained model reproduces persistence. There is no recursive use of future predicted or observed speeds. Model size is {metadata["parameters"]:,} trainable parameters.

## Training and selection

We train with Adam and masked MAE, excluding zero targets. The loss is divided by the training speed standard deviation for optimization, with gradient norm clipped at 1. The speed normalizer uses the same observed training-period values as the existing data pipeline. Training uses a fixed seeded sample of {metadata["training_args"]["train_rows"]:,} window/sensor rows, shared across both learning-rate trials and shuffled each epoch. This reduces computation; it does not cover every training row. Missing horizons remain masked within sampled rows.

The seed is {metadata["training_args"]["seed"]}. Batch size is {metadata["training_args"]["batch_size"]}; training stops after {metadata["training_args"]["patience"]} epochs without improvement, subject to a {metadata["training_args"]["epochs"]}-epoch cap per trial. Every epoch is evaluated on all observed validation targets. We select the learning rate and checkpoint by mean validation MAE across the three horizons. Test data is scored after selection. Both trials use the same initialized weights and training sample.

{markdown_table(tuning.round(4))}

The selected learning rate is {metadata["selected_learning_rate"]}, at epoch {metadata["best_epoch"]}. Its mean validation MAE is {metadata["selection_val_mae"]:.4f} mph. Training used {metadata["device"]} and PyTorch {metadata["torch"]}; full settings and data hash are saved in `outputs/neural/run_metadata.json`. The selected CPU state dictionary, configuration, normalization, horizons, and sensor order are saved in `outputs/neural/gru.pt`.

![Training and validation errors](../outputs/neural/learning_curves.png)

## Test comparison

{markdown_table(table)}

{" ".join(statements)}

![Aggregate and low-speed errors](../outputs/neural/comparison.png)

Low-speed MAE uses the training-selected 40 mph threshold and excludes missing targets:

{markdown_table(ctable)}

The saved GRU predictions use the existing `(window, horizon, sensor)` format, window starts, and horizon steps. The condition-analysis script checks their alignment and reproduces aggregate metrics before reporting these subsets.

## Interpretation and next experiment

These results measure one temporal architecture under one seed and a limited learning-rate search. They do not establish the best possible neural baseline. Classical boosting receives sensor identity and profile features that this GRU does not receive, so the comparison measures the implemented systems rather than isolating architecture alone. Small differences are descriptive, with no significance claim.

The benchmark target periods retain eleven overlapping timestamps at each split boundary. The training normalizer includes the full benchmark training period. The fixed sample and device-dependent floating-point arithmetic also limit reproducibility across hardware; repeated seeds have not been run.

The [graph comparison](graph_model.md) adds learned incoming and outgoing graph filtering under matched temporal settings, with a separately trained identity-graph control. Neural-model error by sensor, time of day, and representative trajectories also remains to be completed.

## Reproduction

```sh
uv sync --extra neural
uv run --extra neural scripts/train_gru.py --device cpu
uv run --extra neural scripts/train_gru.py --device cpu --evaluate-only
uv run scripts/error_analysis.py
uv run scripts/summarize_neural.py
```

The evaluate-only command restores the committed checkpoint and rejects a different data hash or sensor/horizon order. Learning curves and validation trial histories are saved alongside the checkpoint. Test predictions in `outputs/predictions/` remain local and can be regenerated from it.
"""
    (ROOT / "docs" / "temporal_model.md").write_text(report)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
