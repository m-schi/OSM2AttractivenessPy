"""
eval_maps_funcs.py

Helper functions for attractiveness map generation.
Uses matplotlib for static PNG maps and plotly for interactive HTML maps.
"""

import math
from datetime import datetime
from pathlib import Path
from typing import Optional

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def msg(*args):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {''.join(str(a) for a in args)}")


def as_char_vec(x) -> list:
    if x is None: return []
    if isinstance(x, str): return [x]
    if isinstance(x, (list, tuple)): return [str(i) for i in x]
    return [str(x)]


def as_named_numeric(x) -> dict:
    if not x: return {}
    out = {}
    for k, v in x.items():
        try:
            fv = float(v)
            if np.isfinite(fv):
                out[str(k)] = fv
        except (ValueError, TypeError):
            pass
    return out


def find_latest_run_dir(base: Path) -> Optional[Path]:
    base = Path(base)
    if not base.is_dir():
        return None
    runs = sorted(
        [d for d in base.iterdir()
         if d.is_dir() and d.name.startswith("run_")
         and (d / "attractiveness.csv").exists()],
        key=lambda d: d.name, reverse=True,
    )
    return runs[0] if runs else None


# ---------------------------------------------------------------------------
# HTML detail builder
# ---------------------------------------------------------------------------

def build_purpose_details_html(df: pd.DataFrame, purpose: str) -> list:
    detail_cols = [c for c in df.columns if c.endswith(f"_{purpose}") and c != "geometry"]
    if not detail_cols:
        return [""] * len(df)
    labels = [c.replace(f"_{purpose}", "") for c in detail_cols]
    result = []
    for _, row in df.iterrows():
        items = "".join(
            f"<li><strong>{lbl}:</strong> {row.get(col, 0) if pd.notna(row.get(col, 0)) else 0:.2f}</li>"
            for col, lbl in zip(detail_cols, labels))
        result.append(f"<strong>Details:</strong><ul style='margin:4px 0 0 16px;padding:0;'>{items}</ul>")
    return result


# ---------------------------------------------------------------------------
# Regular grid aggregation
# ---------------------------------------------------------------------------

def build_regular_grid_attractiveness(
    poi_subset: gpd.GeoDataFrame,
    purposes: list,
    cellsize_m: float = 1000,
) -> Optional[gpd.GeoDataFrame]:
    if poi_subset is None or len(poi_subset) == 0:
        return None
    poi = poi_subset[["purpose", "attractiveness", "geometry"]].copy()
    poi = poi[~poi.geometry.is_empty]
    if len(poi) == 0:
        return None

    from shapely.geometry import box
    xmin, ymin, xmax, ymax = poi.total_bounds
    nx, ny = max(1, math.ceil((xmax - xmin) / cellsize_m)), max(1, math.ceil((ymax - ymin) / cellsize_m))

    cells, ids = [], []
    for ix in range(nx):
        for iy in range(ny):
            x0, y0 = xmin + ix * cellsize_m, ymin + iy * cellsize_m
            cells.append(box(x0, y0, x0 + cellsize_m, y0 + cellsize_m))
            ids.append(str(ix * ny + iy + 1))

    grid = gpd.GeoDataFrame({"grid_id": ids}, geometry=cells, crs=poi.crs)
    joined = gpd.sjoin(poi, grid, how="inner", predicate="within")
    if len(joined) == 0:
        return None
    joined = joined[joined["purpose"].isin(purposes) & joined["attractiveness"].notna()]
    if len(joined) == 0:
        return None

    wide = (joined.groupby(["grid_id", "purpose"], as_index=False)["attractiveness"].sum()
            .pivot(index="grid_id", columns="purpose", values="attractiveness")
            .reset_index())
    wide.columns.name = None
    grid = grid.merge(wide, on="grid_id", how="left")
    for p in purposes:
        if p not in grid.columns:
            grid[p] = np.nan
    has_data = grid[purposes].notna().any(axis=1)
    grid[purposes] = grid[purposes].fillna(0)
    return grid[has_data].copy()


# ---------------------------------------------------------------------------
# Static maps (matplotlib)
# ---------------------------------------------------------------------------

def _plot(gdf, col, title, out_path, **kw):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    gdf.plot(column=col, ax=ax, cmap="plasma", legend=True,
             missing_kwds={"color": "lightgrey"}, **kw)
    ax.set_title(title); ax.axis("off"); fig.tight_layout()
    fig.savefig(str(out_path), dpi=300); plt.close(fig)


def make_image_map(gdf, zone_type, purpose, out_dir: Path):
    _plot(gdf, purpose, f"Attractiveness: {purpose} (type {zone_type})",
          Path(out_dir) / f"attractiveness_zones_{zone_type}_{purpose}.png")


def make_image_map_grid(gdf, zone_type, purpose, out_dir: Path):
    _plot(gdf, purpose, f"Attractiveness (grid): {purpose} (type {zone_type})",
          Path(out_dir) / f"attractiveness_grid_zones_{zone_type}_{purpose}.png")


def _overview(gdf, zone_type, purposes, out_dir: Path, prefix: str, title_extra: str = ""):
    n = len(purposes)
    nc = min(4, n); nr = (n + nc - 1) // nc
    vals = [v for p in purposes if p in gdf.columns for v in gdf[p].dropna()]
    vmin, vmax = (min(vals), max(vals)) if vals else (0, 1)
    fig, axes = plt.subplots(nr, nc, figsize=(6 * nc, 6 * nr))
    axes = np.atleast_1d(axes).flatten()
    for i, p in enumerate(purposes):
        gdf.plot(column=p, ax=axes[i], cmap="plasma", vmin=vmin, vmax=vmax,
                 missing_kwds={"color": "lightgrey"})
        axes[i].set_title(p); axes[i].axis("off")
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f"Attractiveness{title_extra} — type {zone_type}", fontsize=14)
    fig.tight_layout()
    fig.savefig(str(Path(out_dir) / f"{prefix}_{zone_type}.png"), dpi=300); plt.close(fig)


def make_overview_image_map(gdf, zone_type, purposes, out_dir):
    _overview(gdf, zone_type, purposes, out_dir, "attractiveness_overview_zones")


def make_overview_image_map_grid(gdf, zone_type, purposes, out_dir):
    _overview(gdf, zone_type, purposes, out_dir, "attractiveness_overview_grid_zones", " (grid)")


# ---------------------------------------------------------------------------
# Interactive HTML maps (plotly)
# ---------------------------------------------------------------------------

def _choropleth_trace(gdf, val_col, id_col, hover, custom_cols, zmin, zmax,
                      line_w=1, line_c="white", visible=True, showscale=True, cb_title=""):
    geojson = gdf.__geo_interface__
    ids = gdf[id_col].astype(str).tolist()
    for i, f in enumerate(geojson["features"]):
        f["id"] = ids[i]
    return go.Choroplethmapbox(
        geojson=geojson, locations=ids, z=gdf[val_col].fillna(0).tolist(),
        colorscale="Viridis", zmin=zmin, zmax=zmax,
        marker_opacity=0.7, marker_line_width=line_w, marker_line_color=line_c,
        visible=visible, showscale=showscale,
        colorbar=dict(title=cb_title) if showscale else None,
        hovertemplate=hover,
        customdata=gdf[custom_cols].fillna("").values.tolist() if custom_cols else None,
    )


def make_html_map(zones_sf_leaflet_subset, subset_name, purposes, output_dir,
                  timestamp_string, grid_sf_leaflet_subset=None):
    if not HAS_PLOTLY:
        msg("plotly not installed — skipping interactive map."); return

    out = Path(output_dir) / f"attractiveness_interactive_map_zones_{subset_name}_{timestamp_string}.html"
    z = zones_sf_leaflet_subset
    g = grid_sf_leaflet_subset
    has_grid = g is not None and len(g) > 0

    if z.crs and z.crs.to_epsg() != 4326: z = z.to_crs(epsg=4326)
    if has_grid and g.crs and g.crs.to_epsg() != 4326: g = g.to_crs(epsg=4326)
    if "NO" not in z.columns: z = z.copy(); z["NO"] = range(len(z))
    z["NO"] = z["NO"].astype(str)
    if has_grid: g["grid_id"] = g["grid_id"].astype(str)

    bounds = z.total_bounds
    ctr = {"lat": (bounds[1]+bounds[3])/2, "lon": (bounds[0]+bounds[2])/2}

    fig = go.Figure()
    tpp = []
    hover_cols = [c for c in ("NO", "typ", "NAME") if c in z.columns]

    for purpose in purposes:
        vals = z[purpose].dropna().tolist()
        if has_grid and purpose in g.columns:
            vals += g[purpose].dropna().tolist()
        if not vals: vals = [0]
        vmin, vmax = min(vals), max(vals)
        if vmin == vmax: vmax += 1

        zh = "<br>".join(f"<b>{c}:</b> %{{customdata[{i}]}}" for i, c in enumerate(hover_cols))
        zh += f"<br><b>{purpose}:</b> %{{z:.2f}}<extra></extra>"
        fig.add_trace(_choropleth_trace(z, purpose, "NO", zh, hover_cols, vmin, vmax,
                                         visible=False, cb_title=purpose))
        nt = 1
        if has_grid and purpose in g.columns:
            gh = f"<b>Grid:</b> %{{customdata[0]}}<br><b>{purpose}:</b> %{{z:.2f}}<extra></extra>"
            fig.add_trace(_choropleth_trace(g, purpose, "grid_id", gh, ["grid_id"], vmin, vmax,
                                             line_w=0.4, line_c="#666", visible=False, showscale=False))
            nt += 1
        tpp.append(nt)

    tot = sum(tpp)
    buttons, off = [], 0
    for i, p in enumerate(purposes):
        vis = [False] * tot; vis[off] = True
        buttons.append(dict(label=p, method="update",
                            args=[{"visible": vis}, {"title": f"Attractiveness: {p} — type {subset_name}"}]))
        off += tpp[i]

    if tpp: fig.data[0].visible = True

    menus = [dict(type="dropdown", x=0.01, xanchor="left", y=1.0, yanchor="top",
                  buttons=buttons, showactive=True, active=0)]

    fig.update_layout(
        mapbox=dict(style="open-street-map", center=ctr, zoom=10),
        margin=dict(l=0, r=0, t=40, b=0),
        title=f"Attractiveness: {purposes[0]} — type {subset_name}" if purposes else "",
        updatemenus=menus,
    )
    fig.write_html(str(out), include_plotlyjs="cdn")
    msg(f"Interactive map written: {out}")