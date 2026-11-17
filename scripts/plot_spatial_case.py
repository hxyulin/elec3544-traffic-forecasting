"""Select a truth-based high-contrast validation case and plot saved forecast errors.

Run: uv run --with plotly python scripts/plot_spatial_case.py
Selection deliberately finds a high spatial-contrast example, not a typical case.
No fitting or checkpoint selection occurs.
"""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm
from plot_speed_landscape import triangles

from traffic.data import load_metr_la

OUT = Path("output/figures/spatial_case")
OUT.mkdir(parents=True, exist_ok=True)
PATH = Path("outputs/predictions/followup_spatial_seed42_val.npz")
data = load_metr_la()
with np.load(PATH) as saved:
    pred = saved["pred"]
    starts = saved["starts"]
    hs = saved["horizons"]
np.testing.assert_array_equal(starts, data.splits["val"])
np.testing.assert_array_equal(hs, [3, 6, 12])
assert pred.shape == (len(starts), 3, data.n_sensors) and np.isfinite(pred).all()
hidx = 2
horizon = int(hs[hidx])
origin = starts + 11
target = origin + horizon
y = data.speed[target]
mask = y > 0
valid = np.where(mask, y, np.nan)
iqr = np.full(len(valid), np.nan)
has_observations = mask.any(axis=1)
iqr[has_observations] = np.nanpercentile(
    valid[has_observations], 75, axis=1
) - np.nanpercentile(valid[has_observations], 25, axis=1)
ts = data.timestamps[target]
eligible = (
    (mask.mean(axis=1) >= 0.98) & (ts.dayofweek < 5) & (ts.hour >= 7) & (ts.hour < 19)
)
# Sort with deterministic earliest-time tie break; selection never uses predictions.
ranking = pd.DataFrame(
    {
        "row": np.arange(len(starts)),
        "timestamp": ts,
        "observed": mask.sum(axis=1),
        "iqr_mph": iqr,
        "eligible": eligible,
    }
)
ranked = ranking[eligible].sort_values(
    ["iqr_mph", "timestamp"], ascending=[False, True]
)
if ranked.empty:
    raise ValueError("No eligible case")
row = int(ranked.iloc[0]["row"])
ranking.to_csv(OUT / "selection_audit.csv", index=False)
truth = y[row]
forecast = pred[row, hidx]
ok = mask[row]
error = forecast - truth
lon = data.locations.longitude.to_numpy()
lat = data.locations.latitude.to_numpy()
faces, local, _ = triangles(lon, lat, 2.0)
tri = mtri.Triangulation(lon, lat, triangles=faces)
tri.set_mask(~(local & ok[faces].all(axis=1)))
reslim = max(5, float(np.ceil(np.abs(error[ok]).max() / 5) * 5))
resnorm = TwoSlopeNorm(vmin=-reslim, vcenter=0, vmax=reslim)
speednorm = Normalize(0, 80)
fig = plt.figure(figsize=(15, 6.2), layout="constrained")
for j, (values, title) in enumerate(
    [
        (truth, "Observed speed"),
        (forecast, "Spatial residual MLP forecast"),
        (error, "Signed forecast error"),
    ]
):
    ax = fig.add_subplot(1, 3, j + 1, projection="3d")
    residual = j == 2
    norm = resnorm if residual else speednorm
    cmap = "RdBu_r" if residual else "RdYlGn"
    if residual:
        xx, yy = np.meshgrid([lon.min(), lon.max()], [lat.min(), lat.max()])
        ax.plot_surface(xx, yy, np.zeros_like(xx), color="gray", alpha=0.12)
    else:
        ax.plot_trisurf(
            tri,
            np.where(ok, values, np.nan),
            cmap=cmap,
            norm=norm,
            alpha=0.5,
            linewidth=0.12,
        )
    ax.scatter(
        lon[ok],
        lat[ok],
        values[ok],
        c=values[ok],
        cmap=cmap,
        norm=norm,
        s=14,
        depthshade=False,
        edgecolors=".2",
        linewidths=0.2,
    )
    ax.set(
        xlabel="Longitude (degrees)",
        ylabel="Latitude (degrees)",
        zlim=(-reslim, reslim) if residual else (0, 80),
        title=title,
    )
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    ax.view_init(elev=28, azim=-65)
    ax.set_box_aspect((1.3, 1, 0.9))
    ax.tick_params(labelsize=8)
    fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        shrink=0.5,
        pad=0.02,
        label="Error (mph)" if residual else "Speed (mph)",
    )
fig.suptitle(
    f"60-minute forecast · issued {data.timestamps[origin[row]]:%Y-%m-%d %H:%M} · target {ts[row]:%H:%M}",
    fontsize=15,
)
fig.supxlabel(
    "High spatial-contrast validation case selected using observations only; not representative aggregate performance.\n"
    "Surface interpolates sensor points with edges ≤2 km. Missing targets excluded. Error = forecast − observation; red means speed overestimated.",
    fontsize=9,
)
for ext in ["png", "pdf"]:
    fig.savefig(OUT / f"observed_forecast_error_3d.{ext}", dpi=240)
plt.close(fig)
# Standalone observed landscape for introducing the dataset.
fig = plt.figure(figsize=(8, 6.5), layout="constrained")
ax = fig.add_subplot(111, projection="3d")
ax.plot_trisurf(
    tri,
    np.where(ok, truth, np.nan),
    cmap="RdYlGn",
    norm=speednorm,
    alpha=0.55,
    linewidth=0.15,
)
ax.scatter(
    lon[ok],
    lat[ok],
    truth[ok],
    c=truth[ok],
    cmap="RdYlGn",
    norm=speednorm,
    s=20,
    depthshade=False,
    edgecolors=".2",
    linewidths=0.25,
)
ax.set(
    xlabel="Longitude (degrees)",
    ylabel="Latitude (degrees)",
    zlabel="Speed (mph)",
    zlim=(0, 80),
)
ax.ticklabel_format(axis="x", style="plain", useOffset=False)
ax.view_init(elev=28, azim=-65)
ax.set_box_aspect((1.3, 1, 0.85))
fig.colorbar(
    matplotlib.cm.ScalarMappable(norm=speednorm, cmap="RdYlGn"),
    ax=ax,
    shrink=0.55,
    pad=0.08,
    label="Observed speed (mph)",
)
fig.suptitle(f"METR-LA spatial speed contrast · {ts[row]:%Y-%m-%d %H:%M}", fontsize=14)
fig.supxlabel(
    f"{int(ok.sum())}/207 sensors observed. High-contrast case selected from validation observations.\nSurface is illustrative interpolation with edges ≤2 km; gaps are not filled.",
    fontsize=9,
)
for ext in ["png", "pdf"]:
    fig.savefig(OUT / f"selected_speed_landscape.{ext}", dpi=240)
plt.close(fig)

# Compact 2D counterpart avoids occlusion and allows spatial error patterns to be read.
fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), layout="constrained")
for ax, values, title, residual in zip(
    axes,
    [truth, forecast, error],
    ["Observed speed", "Spatial residual MLP forecast", "Signed forecast error"],
    [False, False, True],
    strict=True,
):
    p = ax.scatter(
        lon[ok],
        lat[ok],
        c=values[ok],
        s=24,
        cmap="RdBu_r" if residual else "RdYlGn",
        norm=resnorm if residual else speednorm,
        edgecolors=".25",
        linewidths=0.3,
    )
    ax.set(xlabel="Longitude (degrees)", ylabel="Latitude (degrees)", title=title)
    ax.set_aspect(1 / np.cos(np.deg2rad(lat.mean())))
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    ax.tick_params(labelsize=8)
    fig.colorbar(
        p, ax=ax, shrink=0.7, label="Error (mph)" if residual else "Speed (mph)"
    )
fig.suptitle(
    f"60-minute forecast at {ts[row]:%Y-%m-%d %H:%M} · observed sensor locations",
    fontsize=13,
)
fig.supxlabel(
    "Deliberately selected high-contrast case. Error = forecast − observation. Red: speed too high; blue: speed too low.\nMissing targets excluded; no spatial interpolation in these maps. Timestamps are as stored in the dataset.",
    fontsize=9,
)
for ext in ["png", "pdf"]:
    fig.savefig(OUT / f"observed_forecast_error_2d.{ext}", dpi=240)
plt.close(fig)
values = pd.DataFrame(
    {
        "sensor_id": data.sensor_ids,
        "longitude": lon,
        "latitude": lat,
        "observed": truth,
        "forecast": forecast,
        "error": np.where(ok, error, np.nan),
        "target_observed": ok,
    }
)
values.to_csv(OUT / "sensor_values.csv", index=False)
metadata = {
    "model": "earlier one-hop spatial residual MLP, seed 42; not the later wider two-hop model",
    "prediction_file": str(PATH),
    "prediction_sha256": hashlib.sha256(PATH.read_bytes()).hexdigest(),
    "selection_rule": "Among validation targets on weekdays 07:00–18:55 with >=98% observed sensors, maximize observed cross-sensor IQR; earliest timestamp breaks ties. Predictions are not used for selection.",
    "eligible_timestamps": int(eligible.sum()),
    "origin": str(data.timestamps[origin[row]]),
    "target": str(ts[row]),
    "horizon_minutes": 60,
    "row": row,
    "observed_sensors": int(ok.sum()),
    "iqr_mph": float(iqr[row]),
    "mae_mph": float(np.abs(error[ok]).mean()),
    "bias_mph": float(error[ok].mean()),
    "rmse_mph": float(np.sqrt(np.mean(error[ok] ** 2))),
    "residual_color_limit": reslim,
    "selection_scope": "deliberately high-contrast illustrative case on reused validation; not typical performance or a holdout claim",
}
(OUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(json.dumps(metadata, indent=2))
html = (
    '<html><body style="margin:0"><img src="'
    + str((OUT / "observed_forecast_error_3d.png").resolve())
    + '" style="width:100%;height:auto"><p style="font:14px system-ui;color:var(--foreground)">A deliberately selected high spatial-contrast example. Red errors mean the forecast speed is too high; blue errors mean it is too low. The 2D companion in the downloadable files shows the same errors without 3D occlusion.</p></body></html>'
)
(OUT / "preview.html").write_text(html)
