"""Road-aware alternative to interpolating a speed surface over empty land.

Run after plot_spatial_case.py: uv run python scripts/plot_road_landscape.py
Uses its exact saved sensor values, with an observation-independent graph display.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from traffic.data import load_metr_la

out = Path("output/figures/spatial_case")
values = pd.read_csv(out / "sensor_values.csv", dtype={"sensor_id": str})
meta = json.loads((out / "metadata.json").read_text())
data = load_metr_la()
assert list(values.sensor_id) == data.sensor_ids
lon = values.longitude.to_numpy()
lat = values.latitude.to_numpy()
x = (lon - lon.mean()) * 111.32 * np.cos(np.deg2rad(lat.mean()))
y = (lat - lat.mean()) * 111.32
ok = values.target_observed.to_numpy(bool)
# Display only the strongest two supplied undirected graph neighbors per sensor.
# Union makes the display undirected; this is not verified physical flow direction.
a = np.maximum(data.adjacency, data.adjacency.T).copy()
np.fill_diagonal(a, 0)
edges = set()
for i in range(len(x)):
    order = np.argsort(-a[i], kind="stable")
    for j in order[:2]:
        if a[i, j] > 0:
            edges.add(tuple(sorted((i, int(j)))))
edges = np.array(sorted(edges))
edges = edges[ok[edges].all(axis=1)]
xy = np.column_stack([x, y])
ground = np.column_stack([xy, np.zeros(len(x))])
fig = plt.figure(figsize=(14, 6), layout="constrained")
for panel, (column, title) in enumerate(
    [
        ("observed", "Observed speed"),
        ("forecast", "60-minute forecast"),
        ("error", "Signed forecast error"),
    ]
):
    z = values[column].to_numpy()
    residual = column == "error"
    limit = max(5, float(np.ceil(np.abs(z[ok]).max() / 5) * 5)) if residual else 80
    norm = (
        TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
        if residual
        else Normalize(0, 80)
    )
    cmap = plt.get_cmap("RdBu_r" if residual else "RdYlGn")
    ax = fig.add_subplot(1, 3, panel + 1, projection="3d")
    ax.add_collection3d(
        Line3DCollection(ground[edges], colors=".65", linewidths=0.75, alpha=0.5)
    )
    locations = np.column_stack([xy, np.where(ok, z, 0)])
    ax.add_collection3d(
        Line3DCollection(
            locations[edges],
            colors=cmap(norm(z[edges].mean(axis=1))),
            linewidths=2,
            alpha=0.8,
        )
    )
    stems = np.stack([ground[ok], locations[ok]], axis=1)
    ax.add_collection3d(
        Line3DCollection(stems, colors=".6", linewidths=0.5, alpha=0.23)
    )
    ax.scatter(
        x[ok],
        y[ok],
        z[ok],
        c=z[ok],
        cmap=cmap,
        norm=norm,
        s=13,
        depthshade=False,
        edgecolors=".25",
        linewidths=0.15,
    )
    ax.set(
        xlabel="East of center (km)",
        ylabel="North of center (km)",
        title=title,
        zlim=(-limit, limit) if residual else (0, 80),
    )
    ax.set_box_aspect((1.3, 1, 0.85))
    ax.view_init(elev=29, azim=-65)
    ax.tick_params(labelsize=8)
    fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        shrink=0.5,
        pad=0.02,
        label="Error (mph)" if residual else "Speed (mph)",
    )
fig.suptitle(f"Road-network landscape · {meta['target'][:16]}", fontsize=15)
fig.supxlabel(
    "Gray lines: graph projected onto the zero-height reference plane. Colored lines: sensor values joined along the same displayed edges.\nTwo strongest supplied undirected graph neighbors per sensor (union); straight segments are schematic, not road geometry. Error = forecast − observation.",
    fontsize=9,
)
for ext in ["png", "pdf"]:
    fig.savefig(out / f"road_network_landscape.{ext}", dpi=240)
plt.close(fig)
meta["display_edges"] = len(edges)
meta["display_origin_longitude"] = float(lon.mean())
meta["display_origin_latitude"] = float(lat.mean())
meta["display_rule"] = (
    "Union of two strongest max(A,A.T) neighbors per sensor, omitting missing endpoints. Selected independently of speed/error; no physical flow orientation claim."
)
(out / "road_network_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
print(f"Created road-network landscape with {len(edges)} displayed edges.")

# A single panel keeps the network readable when used on a presentation slide.
fig = plt.figure(figsize=(8, 6.5), layout="constrained")
ax = fig.add_subplot(111, projection="3d")
z = values.observed.to_numpy()
locations = np.column_stack([xy, z])
cmap = plt.get_cmap("RdYlGn")
norm = Normalize(0, 80)
ax.add_collection3d(
    Line3DCollection(ground[edges], colors=".6", linewidths=0.8, alpha=0.5)
)
ax.add_collection3d(
    Line3DCollection(
        locations[edges],
        colors=cmap(norm(z[edges].mean(axis=1))),
        linewidths=2,
        alpha=0.85,
    )
)
ax.add_collection3d(
    Line3DCollection(
        np.stack([ground[ok], locations[ok]], axis=1),
        colors=".6",
        linewidths=0.5,
        alpha=0.2,
    )
)
ax.scatter(
    x[ok],
    y[ok],
    z[ok],
    c=z[ok],
    norm=norm,
    cmap=cmap,
    s=22,
    depthshade=False,
    edgecolors=".2",
    linewidths=0.2,
)
ax.set(
    xlabel="East of sensor center (km)",
    ylabel="North of sensor center (km)",
    zlabel="Speed (mph)",
    zlim=(0, 80),
)
ax.view_init(elev=29, azim=-65)
ax.set_box_aspect((1.3, 1, 0.85))
fig.colorbar(
    matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
    ax=ax,
    shrink=0.55,
    pad=0.08,
    label="Observed speed (mph)",
)
fig.suptitle(
    f"Observed traffic speeds over the sensor graph\n{meta['target'][:16]}", fontsize=14
)
fig.supxlabel(
    "Gray: geographic reference at zero height. Color and height: observed speed.\nSparse supplied-graph connections are schematic; high-contrast case, not a typical snapshot.",
    fontsize=9,
)
for ext in ["png", "pdf"]:
    fig.savefig(out / f"road_connected_observed.{ext}", dpi=240)
plt.close(fig)

# Plotly is a one-off visualization dependency; embed it for offline use.
from plotly.offline import get_plotlyjs


def joined_segments(points, links):
    result = [[], [], []]
    for i, j in links:
        for dim in range(3):
            result[dim].extend([float(points[i, dim]), float(points[j, dim]), None])
    return dict(zip(["x", "y", "z"], result, strict=True))


modes = {}
for column in ["observed", "forecast", "error"]:
    z = values[column].to_numpy()
    error_mode = column == "error"
    limit = max(5, float(np.ceil(np.abs(z[ok]).max() / 5) * 5)) if error_mode else 80
    positions = np.column_stack([xy, np.where(ok, z, 0)])
    stems = np.concatenate([ground, positions])
    stem_links = np.column_stack([np.flatnonzero(ok), np.flatnonzero(ok) + len(x)])
    line_color = []
    for i, j in edges:
        line_color.extend([float(z[i]), float(z[j]), 0.0])
    scale_map = plt.get_cmap("RdBu_r" if error_mode else "RdYlGn")
    scale = [
        [float(t), matplotlib.colors.to_hex(scale_map(t))]
        for t in np.linspace(0, 1, 11)
    ]
    modes[column] = dict(
        limit=limit,
        data=[
            dict(
                type="scatter3d",
                mode="lines",
                **joined_segments(ground, edges),
                line=dict(color="rgba(140,140,140,.5)", width=2),
                hoverinfo="skip",
                name="Reference graph",
            ),
            dict(
                type="scatter3d",
                mode="lines",
                **joined_segments(stems, stem_links),
                line=dict(color="rgba(140,140,140,.22)", width=1),
                hoverinfo="skip",
                name="Height guides",
            ),
            dict(
                type="scatter3d",
                mode="lines",
                **joined_segments(positions, edges),
                line=dict(
                    color=line_color,
                    colorscale=scale,
                    cmin=-limit if error_mode else 0,
                    cmax=limit,
                    width=4,
                ),
                hoverinfo="skip",
                name="Elevated graph",
            ),
            dict(
                type="scatter3d",
                mode="markers",
                x=x[ok].tolist(),
                y=y[ok].tolist(),
                z=z[ok].tolist(),
                customdata=values.sensor_id[ok].tolist(),
                marker=dict(
                    size=4,
                    color=z[ok].tolist(),
                    colorscale=scale,
                    cmin=-limit if error_mode else 0,
                    cmax=limit,
                    colorbar=dict(
                        title="Error (mph)" if error_mode else "Speed (mph)",
                        thickness=14,
                    ),
                ),
                hovertemplate="Sensor %{customdata}<br>East %{x:.2f} km · North %{y:.2f} km<br>%{z:.2f} mph<extra></extra>",
                name="Sensors",
            ),
        ],
    )
html = (
    """<!doctype html><html><head><meta charset="utf-8"><style>
body{margin:0;background:transparent;color:var(--foreground);font-family:var(--font-sans,system-ui)}h3{margin:8px 0}p{font-size:13px;line-height:1.5}select{font:inherit;padding:5px}#plot{height:520px;width:100%}
</style><script>"""
    + get_plotlyjs()
    + """</script></head><body><h3>Road-connected traffic landscape</h3>
<label>Display <select id="mode"><option value="observed">Observed speed</option><option value="forecast">60-minute forecast</option><option value="error">Signed forecast error</option></select></label>
<span style="font-size:13px">Target 25 May 2012 16:25 · forecast issued 15:25</span><div id="plot"></div>
<p>Drag to rotate; scroll to zoom. Gray lines locate the graph on the zero-height reference plane. Colored edges connect sensor values through the supplied graph; they are schematic, not surveyed roads. For readability, display uses the union of each sensor’s two strongest undirected graph neighbors.</p>
<p>205 of 207 sensors observed; missing targets excluded. Errors are forecast minus observation: red means speed too high, blue means too low. This deliberately high-contrast case uses the earlier one-hop spatial MLP, seed 42, not the wider two-hop model. Timestamps are as stored.</p><script>
const modes=__MODES__;const fg=getComputedStyle(document.documentElement).getPropertyValue('--foreground').trim()||'#b6bbc4';
const layout={margin:{l:0,r:0,t:0,b:0},showlegend:false,paper_bgcolor:'rgba(0,0,0,0)',font:{color:fg},uirevision:'camera',scene:{bgcolor:'rgba(0,0,0,0)',xaxis:{title:{text:'East of center (km)'}},yaxis:{title:{text:'North of center (km)'}},zaxis:{title:{text:'Speed (mph)'},range:[0,80]},aspectmode:'manual',aspectratio:{x:1.3,y:1,z:.85},camera:{eye:{x:1.5,y:-1.7,z:1.1}}}};
function update(){const key=document.getElementById('mode').value;layout.scene.zaxis.range=key==='error'?[-modes[key].limit,modes[key].limit]:[0,80];layout.scene.zaxis.title.text=key==='error'?'Error (mph)':'Speed (mph)';Plotly.react('plot',modes[key].data,layout,{responsive:true,displaylogo:false});}
document.getElementById('mode').onchange=update;update();</script></body></html>"""
)
html = html.replace("__MODES__", json.dumps(modes))
(out / "road_connected_interactive.html").write_text(html)
print("Created standalone observed figure and offline interactive graph.")
