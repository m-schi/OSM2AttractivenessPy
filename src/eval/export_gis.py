#!/usr/bin/env python3
"""
export_gis.py -- Georeference attractiveness.csv with the zone geometries and
export ready-to-use GIS layers (GeoJSON + GeoPackage) plus QGIS style files
(QML) for each attractiveness purpose.

Usage: python export_gis.py [config_name] [run_subdir]
"""

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import yaml

script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir))

from eval_maps_funcs import find_latest_run_dir, msg


# ---------------------------------------------------------------------------
# QML style generation
# ---------------------------------------------------------------------------

def compute_class_breaks(values: pd.Series, n_classes: int = 5) -> np.ndarray:
    """Quantile breaks for *values*, deduplicated (handles skewed/zero-heavy data)."""
    values = values.dropna()
    qs = np.linspace(0, 1, n_classes + 1)
    breaks = np.unique(values.quantile(qs).to_numpy())
    if len(breaks) < 2:
        lo = float(values.min()) if len(values) else 0.0
        breaks = np.array([lo, lo + 1.0])
    return breaks


def ramp_colors(n_classes: int, cmap_name: str = "YlOrRd") -> list:
    """RGBA 0-255 tuples sampled evenly from a matplotlib colormap."""
    cmap = matplotlib.colormaps[cmap_name]
    positions = np.linspace(0.15, 0.95, n_classes) if n_classes > 1 else [0.6]
    return [tuple(int(round(c * 255)) for c in cmap(p)) for p in positions]


def build_graduated_qml(field: str, breaks: np.ndarray) -> str:
    """Build a QGIS QML (graduated, quantile-style) for a polygon layer field."""
    n_classes = len(breaks) - 1
    colors = ramp_colors(n_classes)

    ranges_xml = []
    symbols_xml = []
    for i in range(n_classes):
        lower, upper = float(breaks[i]), float(breaks[i + 1])
        label = f"{lower:.2f} - {upper:.2f}"
        r, g, b, a = colors[i]
        ranges_xml.append(
            f'      <range symbol="{i}" render="true" label="{label}" '
            f'upper="{upper!r}" lower="{lower!r}"/>'
        )
        symbols_xml.append(f'''      <symbol type="fill" name="{i}" alpha="1" clip_to_extent="1">
        <layer class="SimpleFill" locked="0" pass="0">
          <prop k="color" v="{r},{g},{b},255"/>
          <prop k="outline_color" v="35,35,35,150"/>
          <prop k="outline_width" v="0.2"/>
          <prop k="outline_style" v="solid"/>
          <prop k="style" v="solid"/>
        </layer>
      </symbol>''')

    return f'''<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34" styleCategories="AllStyleCategories">
  <renderer-v2 type="graduatedSymbol" symbollevels="0" attr="{field}" forceraster="0" graduatedMethod="GraduatedColor" enableorderby="0">
    <ranges>
{chr(10).join(ranges_xml)}
    </ranges>
    <symbols>
{chr(10).join(symbols_xml)}
    </symbols>
  </renderer-v2>
  <blendMode>0</blendMode>
  <featureBlendMode>0</featureBlendMode>
  <layerOpacity>1</layerOpacity>
  <labeling type="simple">
    <settings calloutType="simple">
      <text-style fieldName="{field}" fontSize="8"/>
    </settings>
  </labeling>
  <custom-properties>
    <property key="labeling/enabled" value="false"/>
  </custom-properties>
</qgis>
'''


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    config_name = sys.argv[1] if len(sys.argv) > 1 else "config_rastatt_example"
    run_subdir = sys.argv[2] if len(sys.argv) > 2 else None

    config_file = (script_dir / ".." / ".." / "config" / f"{config_name}.yaml").resolve()
    area_cfg = yaml.safe_load(config_file.read_text(encoding="utf-8"))

    def resolve(p: str) -> Path:
        return Path(p).expanduser().resolve()

    zones_file = resolve(area_cfg["paths"]["zones_file"])
    out_base = resolve(area_cfg["paths"]["attractiveness_output_root"]) / area_cfg["paths"]["attractiveness_output_sub_attractiveness"]

    if run_subdir:
        out_dir = out_base / run_subdir
    else:
        out_dir = find_latest_run_dir(out_base)
        if out_dir is None:
            raise FileNotFoundError(
                f"No completed run directory found in {out_base}. "
                "Run the attractiveness calculation first (POIs2attractiveness.py)."
            )

    csv_attr = out_dir / "attractiveness.csv"
    gis_dir = out_dir / "GIS"
    styles_dir = gis_dir / "styles"

    print(f"Config: {config_file}")
    print(f"Zones: {zones_file}")
    print(f"Attractiveness: {csv_attr}")
    print(f"Output: {gis_dir}\n")

    for p, label in [(zones_file, "Zones"), (csv_attr, "Attractiveness CSV")]:
        if not p.exists():
            raise FileNotFoundError(f"{label} not found: {p}")

    attr = pd.read_csv(str(csv_attr))
    assert "zoneId" in attr.columns, "attractiveness.csv is missing the 'zoneId' column."
    purposes = [c for c in attr.columns if c != "zoneId"]
    assert purposes, "No purpose columns in attractiveness.csv."
    attr["zoneId"] = attr["zoneId"].astype(str)

    zones = gpd.read_file(str(zones_file))
    zones["NO"] = zones["NO"].astype(str)

    zones_attr = zones.merge(attr, left_on="NO", right_on="zoneId", how="left")
    zones_attr = zones_attr.drop(columns=["zoneId"])
    zones_attr[purposes] = zones_attr[purposes].fillna(0)

    n_unmatched = int((~attr["zoneId"].isin(zones["NO"])).sum())
    if n_unmatched:
        msg(f"Warning: {n_unmatched} attractiveness rows had no matching zone (dropped).")

    gis_dir.mkdir(parents=True, exist_ok=True)
    styles_dir.mkdir(parents=True, exist_ok=True)

    geojson_path = gis_dir / "attractiveness_zones.geojson"
    gpkg_path = gis_dir / "attractiveness_zones.gpkg"

    zones_attr.to_file(str(geojson_path), driver="GeoJSON")
    msg(f"Written: {geojson_path}")
    zones_attr.to_file(str(gpkg_path), driver="GPKG", layer="attractiveness_zones")
    msg(f"Written: {gpkg_path}")

    # Default style (used when either file is opened directly in QGIS) uses
    # the first purpose column; per-purpose styles go into styles/ for
    # manual loading via "Layer Properties > Symbology > Style > Load Style".
    default_field = purposes[0]
    default_qml = build_graduated_qml(default_field, compute_class_breaks(zones_attr[default_field]))
    (gis_dir / "attractiveness_zones.qml").write_text(default_qml, encoding="utf-8")
    msg(f"Written default style (field '{default_field}'): {gis_dir / 'attractiveness_zones.qml'}")

    for purpose in purposes:
        breaks = compute_class_breaks(zones_attr[purpose])
        qml = build_graduated_qml(purpose, breaks)
        qml_path = styles_dir / f"attractiveness_zones__{purpose}.qml"
        qml_path.write_text(qml, encoding="utf-8")
    msg(f"Written {len(purposes)} per-purpose styles to: {styles_dir}")

    print(f"\nDone. In QGIS: open {geojson_path.name} or {gpkg_path.name} from {gis_dir}")
    print("(default style applied automatically); to switch fields use "
          "Layer Properties > Symbology > Style > Load Style and pick a file "
          f"from {styles_dir.name}/.")


if __name__ == "__main__":
    main()
