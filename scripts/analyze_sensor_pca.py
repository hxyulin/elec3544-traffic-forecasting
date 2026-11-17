"""Training-only sensor PCA, scaling/missingness sensitivity and location profiles.

Run: uv run python scripts/analyze_sensor_pca.py
Exploratory covariance analysis, not a forecasting comparison or intrinsic-dimension estimate.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from traffic.data import load_metr_la


def main():
    ds = load_metr_la()
    stop = int(ds.splits["val"][0])
    raw = ds.speed[:stop].astype(np.float64)
    observed = np.isfinite(raw) & (raw > 0)
    means = np.nanmean(np.where(observed, raw, np.nan), axis=0)
    filled = (
        pd.DataFrame(np.where(observed, raw, np.nan))
        .ffill()
        .fillna(pd.Series(means))
        .to_numpy()
    )
    assert np.isfinite(filled).all() and np.isfinite(means).all()
    good = observed.mean(axis=1) >= 0.95
    datasets = {
        "Centered speeds (mph)": filled,
        "Standardized sensors": StandardScaler().fit_transform(filled),
        "Standardized, ≥95% observed": StandardScaler().fit_transform(filled[good]),
    }
    out = Path("output/figures/sensor_pca")
    out.mkdir(parents=True, exist_ok=True)
    records = {}
    fits = {}
    for name, x in datasets.items():
        model = PCA(svd_solver="full").fit(x)
        cumulative = np.cumsum(model.explained_variance_ratio_)
        counts = {
            str(p): int(np.searchsorted(cumulative, p / 100) + 1)
            for p in [80, 90, 95, 99]
        }
        records[name] = {
            "rows": len(x),
            "components_for_variance_percent": counts,
            "first_three_variance_ratio": model.explained_variance_ratio_[:3].tolist(),
        }
        fits[name] = model
        pd.DataFrame(
            {
                "component": np.arange(1, 208),
                "explained_variance_ratio": model.explained_variance_ratio_,
                "cumulative": cumulative,
            }
        ).to_csv(
            out
            / (
                name.split(",")[0].replace(" ", "_").replace("(", "").replace(")", "")
                + ("_high_coverage" if "95%" in name else "")
                + ".csv"
            ),
            index=False,
        )
    plt.rcParams.update(
        {"font.size": 12, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    for (name, model), color in zip(
        fits.items(), ["#9B784E", "#146E80", "#7568A3"], strict=True
    ):
        ax.plot(
            np.arange(1, 208),
            np.cumsum(model.explained_variance_ratio_) * 100,
            label=name,
            color=color,
            lw=2,
        )
    ax.axhline(90, color=".7", ls="--", lw=1)
    ax.axhline(95, color=".7", ls=":", lw=1)
    ax.set(
        xlabel="Number of principal components",
        ylabel="Cumulative variance explained (%)",
        xlim=(1, 207),
        ylim=(0, 100),
    )
    ax.legend(loc="lower right", fontsize=10)
    fig.savefig(out / "explained_variance.png", dpi=190)
    fig.savefig(out / "explained_variance.pdf")
    plt.close(fig)
    model = fits["Standardized sensors"]
    loadings = model.components_[:3].copy()
    # Fix arbitrary PCA sign for display: largest absolute loading is positive.
    for row in loadings:
        if row[np.argmax(abs(row))] < 0:
            row *= -1
    limit = float(abs(loadings).max())
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), layout="constrained")
    for i, ax in enumerate(axes):
        sc = ax.scatter(
            ds.locations.longitude,
            ds.locations.latitude,
            c=loadings[i],
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            s=18,
        )
        ax.set(
            title=f"PC{i + 1}: {model.explained_variance_ratio_[i] * 100:.1f}% variance",
            xlabel="Longitude",
            ylabel="Latitude",
        )
        ax.set_aspect(1 / np.cos(np.deg2rad(ds.locations.latitude.mean())))
    fig.colorbar(sc, ax=axes, label="Sensor loading", shrink=0.7)
    fig.savefig(out / "component_maps.png", dpi=190)
    fig.savefig(out / "component_maps.pdf")
    plt.close(fig)
    pd.DataFrame(
        {
            "sensor_id": ds.sensor_ids,
            "training_observed_mean_mph": means,
            "observed_fraction": observed.mean(axis=0),
            **{f"PC{i + 1}": loadings[i] for i in range(3)},
        }
    ).to_csv(out / "sensor_profiles.csv", index=False)
    meta = {
        "analysis": "AI-assisted training-only exploratory PCA",
        "shape": list(raw.shape),
        "fit_start": str(ds.timestamps[0]),
        "fit_end_inclusive": str(ds.timestamps[stop - 1]),
        "cutoff_rule": "Strictly before first validation window start. No validation/test values used.",
        "raw_sha256": hashlib.sha256(ds.speed[:stop].tobytes()).hexdigest(),
        "missing_fraction": float(1 - observed.mean()),
        "filling": "Past-value carry forward; leading fallback is each sensor observed training mean. PCA sensitivity uses rows with ≥95% original observations.",
        "scaling": "Primary PCA uses per-sensor mean/std fitted on filled training rows; this is separate from forecasting global z-score normalization. High-coverage sensitivity refits scaling on its own training subset.",
        "results": records,
        "training_sensor_means": {
            "min": float(means.min()),
            "median": float(np.median(means)),
            "max": float(means.max()),
        },
        "limits": [
            "Variance explained is not forecast accuracy or intrinsic nonlinear dimension.",
            "Sensor-column PCA is not PCA of the engineered lag feature matrix used by Ridge.",
            "Filling and scaling affect components; spatial loading patterns are descriptive, not causal.",
            "No Lasso comparison or explicit learned sensor-bias fit was performed.",
        ],
    }
    (out / "summary.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
