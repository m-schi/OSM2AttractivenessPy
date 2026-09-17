#!/usr/bin/env python3
"""
eval_maps.py — Generate attractiveness maps by zone type and activity.

Usage: python eval_maps.py [config_name] [run_subdir]
"""

import sys
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd
import yaml

script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(script_dir / ".." / "make"))

from eval_maps_funcs import (
    as_char_vec, as_named_numeric, build_regular_grid_attractiveness,
    find_latest_run_dir, make_html_map, make_image_map, make_image_map_grid,
    make_overview_image_map, make_overview_image_map_grid, msg,
)


def main():
    config_name = sys.argv[1] if len(sys.argv) > 1 else "config_rastatt_example"
    run_subdir = sys.argv[2] if len(sys.argv) > 2 else None

    config_file = (script_dir / ".." / ".." / "config" / f"{config_name}.yaml").resolve()
    area_cfg = yaml.safe_load(config_file.read_text(encoding="utf-8"))

    def resolve(p: str) -> Path:
        return Path(p).expanduser().resolve()

    zones_file = resolve(area_cfg["paths"]["zones_file"])
    out_base = resolve(area_cfg["paths"]["attractiveness_output_root"]) / area_cfg["paths"]["attractiveness_output_sub_attractiveness"]

    eval_cfg = area_cfg.get("eval", {}).get("attractiveness_maps", {})
    zone_types_cfg = as_char_vec(eval_cfg.get("zone_types"))
    grid_default = float(eval_cfg.get("grid_cellsize_m_default", 1000))
    grid_by_type = as_named_numeric(eval_cfg.get("grid_cellsize_m_by_type", {"1": 1000, "2": 2000, "3": 5000}))

    if run_subdir:
        out_dir = out_base / run_subdir
    else:
        out_dir = find_latest_run_dir(out_base) or out_base

    maps_dir = out_dir / "Maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    csv_attr = out_dir / "attractiveness.csv"
    gpkg_poi = out_dir / "all_poi_attractiveness.gpkg"
    csv_detail = out_dir / "attractiveness_detailed_by_category_purpose.csv"

    print(f"Config: {config_file}")
    print(f"Zones: {zones_file}")
    print(f"Output: {out_dir}\n")

    for p, label in [(zones_file, "Zones"), (csv_attr, "Attractiveness CSV"),
                     (gpkg_poi, "POI GPKG"), (csv_detail, "Detail CSV")]:
        if not p.exists():
            raise FileNotFoundError(f"{label} not found: {p}")

    attr = pd.read_csv(str(csv_attr))
    purposes = [c for c in attr.columns if c != "zoneId"]
    assert purposes, "No purpose columns in attractiveness.csv."

    zones = gpd.read_file(str(zones_file))
    zones["NO"] = zones["NO"].astype(str); zones["typ"] = zones["typ"].astype(str)
    attr["zoneId"] = attr["zoneId"].astype(str)
    zones = zones.merge(attr, left_on="NO", right_on="zoneId", how="left")

    pois = gpd.read_file(str(gpkg_poi))
    assert len(pois) > 0, "all_poi_attractiveness.gpkg is empty."
    if pois.crs != zones.crs: pois = pois.to_crs(zones.crs)
    poi_zj = gpd.sjoin(pois, zones[["NO", "typ", "geometry"]], how="inner", predicate="within")

    det = pd.read_csv(str(csv_detail))
    det["zoneId"] = det["zoneId"].astype(str)
    zones = zones.merge(det, left_on="NO", right_on="zoneId", how="left", suffixes=("", "_detail"))

    avail = sorted(zones["typ"].unique())
    ztypes = [z for z in zone_types_cfg if z in avail] if zone_types_cfg else avail
    assert ztypes, "No zone types to process."

    for zt in ztypes:
        msg(f"Zone type: {zt}")
        zs = zones[zones["typ"] == zt].copy()
        if len(zs) == 0: continue

        ps = poi_zj[poi_zj["typ"].astype(str) == zt].copy()
        gs = build_regular_grid_attractiveness(ps, purposes, grid_by_type.get(zt, grid_default))

        for p in purposes:
            make_image_map(zs, zt, p, maps_dir)
            if gs is not None and len(gs) > 0:
                make_image_map_grid(gs, zt, p, maps_dir)

        make_overview_image_map(zs, zt, purposes, maps_dir)
        if gs is not None and len(gs) > 0:
            make_overview_image_map_grid(gs, zt, purposes, maps_dir)

        if eval_cfg.get("make_html_maps", False):
            zl = zs.to_crs(epsg=4326) if zs.crs else zs
            gl = gs.to_crs(epsg=4326) if gs is not None and len(gs) > 0 and gs.crs else None
            make_html_map(zl, str(zt), purposes, maps_dir, timestamp, gl)

    msg("Finished map generation.")


if __name__ == "__main__":
    main()