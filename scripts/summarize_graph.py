"""Write the graph comparison and control verification from saved experiments."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from traffic.data import ROOT
from traffic.reporting import markdown_table, model_label

OUT = ROOT / "outputs" / "neural"
GRAPH = OUT / "graph_gru"


def main():
    metrics = pd.read_csv(ROOT / "outputs/results/metrics.csv")
    names = [
        "persistence",
        "gradient_boosting_neighbours",
        "gru",
        "graph_gru_no_graph",
        "graph_gru",
    ]
    test = metrics[(metrics.split == "test") & metrics.model.isin(names)]
    table = test.pivot(index="model", columns="minutes", values=["mae", "rmse"])
    table.columns = [f"{metric.upper()} {h} min" for metric, h in table.columns]
    table = table.reindex(names).round(3).reset_index()
    conditions = pd.read_csv(ROOT / "outputs/error_analysis/condition_metrics.csv")
    low = conditions[
        (conditions.condition == "below_40_mph") & conditions.model.isin(names)
    ]
    ctable = (
        low.pivot(index="model", columns="minutes", values="mae")
        .reindex(names)
        .round(3)
        .reset_index()
    )
    ctable.columns = ["Model", "15 min MAE", "30 min MAE", "60 min MAE"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # The disabled control overlaps the temporal baseline, so show it in the table.
    for ax, data, title in zip(
        axes, [test, low], ["All observed targets", "Observed targets below 40 mph"]
    ):
        for name in [n for n in names if n != "graph_gru_no_graph"]:
            group = data[data.model == name]
            ax.plot(group.minutes, group.mae, marker="o", label=model_label(name))
        ax.set(
            title=title,
            xlabel="Forecast horizon (minutes)",
            ylabel="Test MAE (mph)",
            xticks=[15, 30, 60],
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(GRAPH / "comparison.png", dpi=160)
    plt.close(fig)
    # Verify the separately trained disabled control against the previous temporal run.
    reference = (
        metrics[metrics.model == "gru"].set_index(["split", "horizon"]).sort_index()
    )
    disabled = (
        metrics[metrics.model == "graph_gru_no_graph"]
        .set_index(["split", "horizon"])
        .sort_index()
    )
    error = float(
        np.max(
            np.abs(
                reference[["mae", "rmse", "mape"]].to_numpy()
                - disabled[["mae", "rmse", "mape"]].to_numpy()
            )
        )
    )
    t0 = pd.read_csv(OUT / "tuning.csv")
    t1 = pd.read_csv(OUT / "graph_gru_no_graph/tuning.csv")
    history0 = pd.read_csv(OUT / "history.csv")
    history1 = pd.read_csv(OUT / "graph_gru_no_graph/history.csv")
    pd.testing.assert_frame_equal(t0, t1, atol=1e-6, rtol=0)
    pd.testing.assert_frame_equal(
        history0.drop(columns="seconds"),
        history1.drop(columns="seconds"),
        atol=1e-6,
        rtol=0,
    )
    np.testing.assert_allclose(
        reference[["mae", "rmse", "mape", "n"]],
        disabled[["mae", "rmse", "mape", "n"]],
        atol=0.00011,
        rtol=0,
    )
    verification = {
        "control": "separately trained identity-graph model",
        "max_metric_difference_csv": error,
        "validation_trials_and_histories": "match temporal run within 1e-6 mph, excluding wall times",
        "tolerance_metrics": 0.00011,
    }
    (GRAPH / "control_verification.json").write_text(
        json.dumps(verification, indent=2) + "\n"
    )
    metadata = json.loads((GRAPH / "run_metadata.json").read_text())
    no_graph_meta = json.loads(
        (OUT / "graph_gru_no_graph/run_metadata.json").read_text()
    )
    tuning = pd.read_csv(GRAPH / "tuning.csv")
    comparisons = []
    g = test[test.model == "graph_gru"].set_index("minutes")
    for name in ["graph_gru_no_graph", "gradient_boosting_neighbours"]:
        ref = test[test.model == name].set_index("minutes")
        deltas = ref.mae - g.mae
        comparisons.append(
            "MAE differences ("
            + model_label(name)
            + " minus graph GRU) are "
            + ", ".join(f"{deltas.loc[h]:.3f}" for h in (15, 30, 60))
            + " mph at 15, 30, and 60 minutes. Positive values favour the graph model."
        )
    low_delta = (
        low[low.model == "graph_gru_no_graph"].set_index("minutes").mae
        - low[low.model == "graph_gru"].set_index("minutes").mae
    )
    comparisons.append(
        "On low-speed targets, the corresponding disabled-control minus graph differences are "
        + ", ".join(f"{low_delta.loc[h]:.3f}" for h in (15, 30, 60))
        + " mph."
    )
    report = f"""# Graph and temporal model comparison

This experiment tests whether spatial measurements help the temporal GRU. Both variants use the same 32-unit GRU and three-output persistence-residual head. A learned one-hop graph filter adds incoming and outgoing neighbour information to the speed and observation-flag channels before the recurrent layer. Calendar channels and the local persistence anchor stay unchanged. Recurrent hidden states remain local to each sensor.

## Graph construction and computation

Let `A[i,j]` be the existing road-distance weight from sensor i to j. We remove diagonal entries for neighbour aggregation because the model has a separate local path. The outgoing support `P_out` row-normalizes A; the incoming support `P_in` row-normalizes its transpose. Empty rows use the identity, giving no neighbour-minus-local signal. The two directions are kept separate. No symmetrization is applied.

At each observed input timestamp, `X` contains normalized filled speed and the observation flag. The filter is:

```text
D_in  = P_in X - X
D_out = P_out X - X
Z     = X + D_in W_in + D_out W_out
```

The implementation uses PyTorch linear-layer transpose conventions for each 2-by-2 matrix. `W_in` and `W_out` start at zero. The GRU receives Z and the original five calendar channels. The output residual is added to the original local speed, preserving the persistence anchor. Propagation uses only simultaneous input readings and the fixed road graph; it uses no future speeds or fitted test statistics. Filled neighbour speeds carry observation flags, but stale readings are not excluded or reweighted by gap length.

Bidirectional graph aggregation is motivated by the [DCRNN traffic-forecasting paper](https://arxiv.org/abs/1707.01926). Our experiment uses a one-hop linear input filter with a local recurrent state and direct horizon outputs. It does not implement diffusion in recurrent hidden states or an encoder-decoder training procedure. Graph direction represents listed source-to-destination road paths; physical upstream/downstream traffic direction has not been independently verified.

## Matched training and disabled-graph control

Both graph variants have {metadata["parameters"]:,} parameters, including eight graph-filter coefficients. The disabled variant sets both supports to the identity. Its two difference signals are exactly zero, so the graph weights have zero gradient and the remaining network reduces to the temporal GRU. We train it separately, rather than disabling messages only after fitting a graph model. Unit tests check predictions, gradients, and optimizer updates against the temporal model. Its trial scores and epoch histories match the previous temporal run within 1e-6 mph, excluding wall times; aggregate CSV metrics differ by at most {error:.6f}. These checks are recorded in `control_verification.json`.

All variants share seed {metadata["training_args"]["seed"]}, a fixed sample of {metadata["training_args"]["train_rows"]:,} training window/sensor rows, batch size {metadata["training_args"]["batch_size"]}, the same initial GRU/head parameters, Adam, masked MAE, gradient clipping at 1, and training-only speed normalization. Each variant separately selects among learning rates 0.001 and 0.003 using mean validation MAE over all three horizons. Training has a 25-epoch cap and patience of four epochs. The graph matrices do not affect sample selection or shuffling. The graph and disabled variants have the same selection budget; their chosen epochs and rates can differ.

Graph-model validation trials:

{markdown_table(tuning.round(4))}

The graph model selects learning rate {metadata["selected_learning_rate"]} at epoch {metadata["best_epoch"]}, with validation MAE {metadata["selection_val_mae"]:.4f} mph. The disabled control selects {no_graph_meta["selected_learning_rate"]} at epoch {no_graph_meta["best_epoch"]}. Test scores are used after selection for comparison, not for choosing settings.

![Graph training and validation errors](../outputs/neural/graph_gru/learning_curves.png)

## Test results

{markdown_table(table)}

{" ".join(comparisons)}

![Graph, temporal and classical comparison](../outputs/neural/graph_gru/comparison.png)

Low-speed targets are observed speeds below the training-selected 40 mph threshold:

{markdown_table(ctable)}

The separately trained disabled graph duplicates the temporal reference within numerical tolerance, so the plot shows the temporal curve once. The table retains both records to make the ablation explicit. All models share the benchmark windows and missing-target masks. Classical neighbour boosting receives additional sensor/profile features, so its comparison with neural models is between implemented systems. The graph-versus-disabled comparison controls the neural architecture and training budget.

## Limits and remaining evaluation

Results use one seed, a fixed training sample, two learning rates, and a short training budget. They do not establish the best graph architecture or statistical significance. Adjacent benchmark split target periods overlap by eleven timestamps (55 minutes). Errors from overlapping windows, sensors, and reciprocal links are dependent. The graph filter is restricted to one-hop input aggregation and can compress neighbour information before recurrence. The identity control tests the added graph messages; a shuffled-edge control would be needed to isolate the value of this particular road topology. Repeated seeds, deeper propagation, graph effects on hidden states, and a purged-split sensitivity study remain possible extensions.

Final analysis still needs errors by sensor and time of day, representative forecast traces, and failure cases. The final report and presentation should separate measured spatial gains from limitations and avoid claiming that graph structure guarantees an improvement.

## Reproduction

```sh
uv run --extra neural scripts/train_gru.py --model graph_gru --device cpu
uv run --extra neural scripts/train_gru.py --model graph_gru_no_graph --device cpu
uv run --extra neural scripts/train_gru.py --model graph_gru --device cpu --evaluate-only
uv run --extra neural scripts/train_gru.py --model graph_gru_no_graph --device cpu --evaluate-only
uv run scripts/error_analysis.py
uv run scripts/summarize_graph.py
```

Each variant saves its checkpoint, validation histories, selected settings, learning curves, and run metadata under `outputs/neural/<model>/`. Checkpoint restoration checks the speed-data hash, road-adjacency hash, sensor order, horizons, and normalizer. Predictions retain the existing `(window, horizon, sensor)` format in `outputs/predictions/`.
"""
    (ROOT / "docs/graph_model.md").write_text(report)
    print(table.to_string(index=False))
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
