#!/usr/bin/env python3
"""
POIs2attractiveness.py — Calculate zone attractiveness from POI files.

Usage: python POIs2attractiveness.py [config_name]
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from POIs2attractiveness_helpers import load_attractiveness_factors_from_json, msg
from POIs2attractiveness_funcs import cleaning_rules
from POIs2attractiveness_main import calculate_attractiveness


def main():
    config_name = sys.argv[1] if len(sys.argv) > 1 else "config_rastatt_example"
    script_dir = Path(__file__).resolve().parent
    config_file = (script_dir / ".." / ".." / "config" / f"{config_name}.yaml").resolve()

    area_cfg = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    paths = area_cfg["paths"]
    area_name = area_cfg["area_name"]

    def resolve(p: str) -> Path:
        return Path(p).expanduser().resolve()

    poi_dir = resolve(paths["pois_root_dir"]) / paths["pois_dir_name"]
    zones_file = resolve(paths["zones_file"])
    factors_json = resolve(paths["attractiveness_factors_json"])

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    out_dir = resolve(paths["attractiveness_output_root"]) / paths["attractiveness_output_sub_attractiveness"] / f"run_{timestamp}"
    out_csv = out_dir / "attractiveness.csv"
    out_gpkg_pois = out_dir / "all_pois.gpkg"
    out_gpkg_attr = out_dir / "all_poi_attractiveness.gpkg"

    print("=" * 80)
    print("POIs2attractiveness")
    print("=" * 80)
    print(f"\nArea: {area_name}")
    print(f"Config: {config_file}")
    print(f"POI dir: {poi_dir}")
    print(f"Zones: {zones_file}")
    print(f"Factors: {factors_json}")
    print(f"Output: {out_dir}\n")

    out_dir.mkdir(parents=True, exist_ok=True)

    factors = load_attractiveness_factors_from_json(factors_json)
    defaults = {"area_field": "Area", "levels_field": "Level_number",
                "floor_field": "FloorArea", "level_default": 1}

    res = calculate_attractiveness(
        poi_dir=poi_dir, zones_file=zones_file, attractiveness_factors=factors,
        poi_file_prefix=f"{area_name}_", poi_file_suffix=".geojson",
        zone_id_field="NO", cleaning_rules=cleaning_rules,
        percentile_imputation=area_cfg.get("percentile_imputation", 0.15),
        defaults=defaults,
        output_csv=out_csv,
        output_gpkg_all_pois=out_gpkg_pois,
        output_gpkg_all_poi_attractiveness=out_gpkg_attr,
    )

    print(res["zone_attractiveness"])

    all_attr = res["all_poi_attractiveness"]
    det = all_attr.pivot_table(index="zoneId", columns=["category", "purpose"],
                               values="attractiveness", aggfunc="sum", fill_value=0)
    det.columns = [f"{c}_{p}" for c, p in det.columns]
    det.reset_index().to_csv(str(out_dir / "attractiveness_detailed_by_category_purpose.csv"), index=False)
    all_attr.to_csv(str(out_dir / "attractiveness_detailed_all_pois.csv"), index=False)


if __name__ == "__main__":
    main()