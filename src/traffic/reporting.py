"""Shared model labels and Markdown tables for experiment reports."""

from __future__ import annotations

import pandas as pd

MODEL_LABELS = {
    "persistence": "Persistence",
    "historical_average": "Historical average",
    "ridge_lags": "Ridge (lags only)",
    "ridge": "Ridge",
    "ridge_neighbours": "Ridge + neighbours",
    "gradient_boosting": "Gradient boosting",
    "gradient_boosting_neighbours": "Gradient boosting + neighbours",
    "gru": "Temporal GRU",
    "graph_gru_no_graph": "Graph GRU (disabled)",
    "graph_gru": "Graph GRU",
}


def model_label(name: str) -> str:
    return MODEL_LABELS.get(name, name.replace("_", " "))


def markdown_table(frame: pd.DataFrame) -> str:
    rows = [frame.columns, ["---"] * len(frame.columns)]
    rows.extend(frame.itertuples(index=False, name=None))
    return "\n".join("| " + " | ".join(map(str, row)) + " |" for row in rows)
