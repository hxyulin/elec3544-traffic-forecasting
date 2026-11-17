"""Plot observed METR-LA speeds over coordinates, plus an illustrative surface.

Run with: uv run --with plotly python scripts/plot_speed_landscape.py
No models are fitted. Dates are read as stored, without assigning a timezone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

from traffic.data import load_metr_la


def triangles(lon, lat, edge_limit):
    """Triangulate geography and omit edges above the chosen display limit."""
    xy = np.column_stack(
        [
            (lon - lon.mean()) * 111.32 * np.cos(np.deg2rad(lat.mean())),
            (lat - lat.mean()) * 111.32,
        ]
    )
    tri = mtri.Triangulation(xy[:, 0], xy[:, 1])
    vertices = xy[tri.triangles]
    edges = np.linalg.norm(vertices - np.roll(vertices, 1, axis=1), axis=2)
    keep = edges.max(axis=1) <= edge_limit
    return tri.triangles, keep, edge_limit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="2012-05-10")
    parser.add_argument("--hour", type=int, default=17, choices=range(24))
    parser.add_argument(
        "--output", type=Path, default=Path("output/figures/speed_landscape")
    )
    parser.add_argument("--max-edge-km", type=float, default=2.0)
    args = parser.parse_args()
    data = load_metr_la()
    lon = data.locations.longitude.to_numpy()
    lat = data.locations.latitude.to_numpy()
    valid_coords = np.isfinite(lon) & np.isfinite(lat)
    if not valid_coords.all():
        raise ValueError(
            "Missing sensor coordinates; handle explicitly before plotting."
        )
    indices = []
    for hour in range(24):
        stamp = pd.Timestamp(f"{args.date} {hour:02d}:00:00")
        indices.append(int(data.timestamps.get_loc(stamp)))
    faces, keep, edge_limit = triangles(lon, lat, args.max_edge_km)
    args.output.mkdir(parents=True, exist_ok=True)
    norm = matplotlib.colors.Normalize(0, 80)
    cmap = plt.get_cmap("RdYlGn")
    index = indices[args.hour]
    speed = data.speed[index]
    observed = np.isfinite(speed) & (speed > 0)
    face_keep = keep & observed[faces].all(axis=1)
    tri = mtri.Triangulation(lon, lat, triangles=faces)
    tri.set_mask(~face_keep)
    fig = plt.figure(figsize=(13, 6.5), layout="constrained")
    for col, title in enumerate(
        ["Observed sensor speeds", "Illustrative surface between nearby sensors"]
    ):
        ax = fig.add_subplot(1, 2, col + 1, projection="3d")
        if col == 1 and face_keep.any():
            ax.plot_trisurf(
                tri,
                np.where(observed, speed, np.nan),
                cmap=cmap,
                norm=norm,
                alpha=0.72,
                linewidth=0.2,
                antialiased=True,
            )
        else:
            for x, y, z in zip(
                lon[observed], lat[observed], speed[observed], strict=True
            ):
                ax.plot([x, x], [y, y], [0, z], color="0.65", alpha=0.25, lw=0.6)
        ax.scatter(
            lon[observed],
            lat[observed],
            speed[observed],
            c=speed[observed],
            cmap=cmap,
            norm=norm,
            s=12,
            edgecolors="0.2",
            linewidths=0.2,
            depthshade=False,
        )
        ax.set(
            xlabel="Longitude (degrees)",
            ylabel="Latitude (degrees)",
            zlabel="Speed (mph)" if col == 1 else "",
            zlim=(0, 80),
            title=title,
        )
        ax.view_init(elev=27, azim=-65)
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        ax.set_box_aspect((1.3, 1, 0.8))
    fig.suptitle(
        f"METR-LA speed landscape · {data.timestamps[index]:%Y-%m-%d %H:%M}",
        fontsize=15,
    )
    fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=fig.axes,
        shrink=0.55,
        pad=0.06,
        label="Speed (mph)",
    )
    fig.supxlabel(
        f"{observed.sum()}/207 sensors observed. Missing zeros excluded. Surface is linear interpolation, not a road map.\n"
        f"Triangles with an edge longer than {edge_limit:.2f} km are omitted. Timestamp is as stored in the dataset.",
        fontsize=9,
    )
    for ext in ["png", "pdf"]:
        fig.savefig(args.output / f"speed_landscape.{ext}", dpi=240)
    plt.close(fig)

    # Two traces per frame: observed points and explicitly illustrative local mesh.
    def traces(hour):
        z = data.speed[indices[hour]]
        ok = np.isfinite(z) & (z > 0)
        f = faces[keep & ok[faces].all(axis=1)]
        scatter = go.Scatter3d(
            x=lon[ok],
            y=lat[ok],
            z=z[ok],
            mode="markers",
            marker=dict(
                size=4,
                color=z[ok],
                colorscale="RdYlGn",
                cmin=0,
                cmax=80,
                colorbar=dict(title="mph", thickness=12),
            ),
            customdata=np.array(data.sensor_ids)[ok],
            hovertemplate="Sensor %{customdata}<br>Longitude %{x:.5f}<br>Latitude %{y:.5f}<br>%{z:.1f} mph<extra></extra>",
            name="Observed sensors",
        )
        mesh = go.Mesh3d(
            x=lon,
            y=lat,
            z=np.where(ok, z, 0),
            i=f[:, 0],
            j=f[:, 1],
            k=f[:, 2],
            intensity=z,
            colorscale="RdYlGn",
            cmin=0,
            cmax=80,
            opacity=0.42,
            showscale=False,
            hoverinfo="skip",
            name="Illustrative surface",
            visible=False,
        )
        return [scatter, mesh]

    payload = []
    for hour in range(24):
        payload.append(json.loads(go.Figure(data=traces(hour)).to_json())["data"])
    html = (
        """<!doctype html><html><head><meta charset="utf-8"><style>
:root{font-family:var(--font-sans,system-ui);color:var(--foreground);color-scheme:light dark}
html,body{margin:0;background:transparent}h2{font-size:18px;margin:8px 0}p{font-size:13px;line-height:1.5;color:var(--muted-foreground)}
.controls{display:flex;gap:14px;align-items:center;flex-wrap:wrap;font-size:13px}input[type=range]{width:220px}#chart{height:510px;width:100%}
</style><script>"""
        + get_plotlyjs()
        + """</script></head><body>
<h2>Traffic speed over sensor locations</h2>
<div class="controls"><label>Hour <input id="hour" type="range" min="0" max="23" value="HOUR"></label><strong id="stamp"></strong><label><input id="surface" type="checkbox" checked> Join nearby sensors</label></div>
<div id="chart"></div><p id="count"></p>
<p>Drag to rotate; scroll to zoom; hover for sensor IDs and speeds. Heights and colors show observed speed in mph. Zero denotes missing and is excluded. The optional surface interpolates between sensors; it is not a measured traffic field or a road map. Long triangles are omitted. Geographic separation does not establish road connectivity. Timestamps are shown as stored, without a verified timezone.</p>
<script>
const frames=__PAYLOAD__, date=__DATE__, limit=__LIMIT__;
const chart=document.getElementById('chart'),slider=document.getElementById('hour'),surface=document.getElementById('surface');
const color=getComputedStyle(document.documentElement).getPropertyValue('--foreground').trim()||'#b6bbc4';
const layout={margin:{l:0,r:0,t:5,b:0},paper_bgcolor:'rgba(0,0,0,0)',font:{color},showlegend:false,uirevision:'keep-camera',
scene:{bgcolor:'rgba(0,0,0,0)',xaxis:{title:{text:'Longitude (degrees)'}},yaxis:{title:{text:'Latitude (degrees)'}},zaxis:{title:{text:'Speed (mph)'},range:[0,80]},camera:{eye:{x:1.5,y:-1.7,z:1.1}},aspectmode:'manual',aspectratio:{x:1.3,y:1,z:.8}}};
function update(){const h=+slider.value;const traces=structuredClone(frames[h]);traces[1].visible=surface.checked;Plotly.react(chart,traces,layout,{responsive:true,displaylogo:false});document.getElementById('stamp').textContent=date+' '+String(h).padStart(2,'0')+':00';document.getElementById('count').textContent=traces[0].customdata.length+'/207 observed sensors. Maximum surface edge: '+limit.toFixed(2)+' km.';}
slider.oninput=update;surface.onchange=update;update();
</script></body></html>"""
    )
    html = (
        html.replace("__PAYLOAD__", json.dumps(payload))
        .replace("__DATE__", json.dumps(args.date))
        .replace("__LIMIT__", str(edge_limit))
        .replace('value="HOUR"', f'value="{args.hour}"')
    )
    (args.output / "speed_landscape.html").write_text(html)
    metadata = dict(
        date=args.date,
        snapshot=str(data.timestamps[index]),
        observed=int(observed.sum()),
        missing=int((~observed).sum()),
        sensors=207,
        triangles_retained=int(face_keep.sum()),
        maximum_edge_km=edge_limit,
        missing_policy="exclude raw zero readings; no imputation",
        surface="linear triangulation in geographic coordinates; local distance edge mask",
        purpose="descriptive visualization, not model selection or predictions",
    )
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
