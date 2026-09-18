"""
POIs2attractiveness.py

Calculate zone-level attractiveness from POI files.

Usage: python POIs2attractiveness.py [config_name]
"""

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml

from POIs2attractiveness_helpers import (
    assert_true, extract_spec, find_poi_file, hash_category_prefix,
    load_attractiveness_factors_from_json,
    msg, print_progress, report_poi_coverage, validate_categories_map, write_gpkg,
)
from POIs2attractiveness_funcs import (
    apply_adjuster, apply_category_filter, apply_cleaning_rules,
    assign_zones, calc_metric_vector, cleaning_rules,
    extract_category_purpose_specs, resolve_category_percentile_imputation,
)


def calculate_attractiveness(
    poi_dir: Path,
    zones_file: Path,
    attractiveness_factors: dict,
    poi_file_prefix: str = "",
    poi_file_suffix: str = ".geojson",
    zone_id_field: str = "zoneId",
    cleaning_rules: Optional[dict] = None,
    percentile_imputation: float = 0.15,
    defaults: Optional[dict] = None,
    output_csv: Optional[Path] = None,
    output_gpkg_all_pois: Optional[Path] = None,
    output_gpkg_all_poi_attractiveness: Optional[Path] = None,
) -> dict:
    """Calculate zone-level attractiveness from POI files."""
    poi_dir = Path(poi_dir)
    zones_file = Path(zones_file)

    if cleaning_rules is None:
        cleaning_rules = {}
    if defaults is None:
        defaults = {"area_field": "Area", "levels_field": "Level_number",
                     "floor_field": "FloorArea", "level_default": 1}

    msg("Starting attractiveness calculation.")
    validate_categories_map(attractiveness_factors)

    assert_true(poi_dir.is_dir(), f"POI folder does not exist: {poi_dir}")
    assert_true(zones_file.exists(), f"Zone file does not exist: {zones_file}")
    report_poi_coverage(poi_dir, attractiveness_factors,
                        file_prefix=poi_file_prefix, file_suffix=poi_file_suffix)

    zones_sf = gpd.read_file(str(zones_file))
    assert_true(len(zones_sf) > 0, "Zone file contains no features.")
    assert_true(zone_id_field in zones_sf.columns,
                 f"Zone field '{zone_id_field}' missing in zone file.")
    zones_sf[zone_id_field] = zones_sf[zone_id_field].astype(str)
    zone_ids_all = zones_sf[zone_id_field].unique().tolist()

    agg_list, poi_export_all, poi_export_purpose = [], [], []

    categories = list(attractiveness_factors.keys())
    poi_paths = [poi_dir / f"{poi_file_prefix}{c}{poi_file_suffix}" for c in categories]
    poi_sizes = [p.stat().st_size if p.exists() else 0 for p in poi_paths]
    total_mb = sum(poi_sizes) / 1024**2
    size_done = 0.0
    msg(f"Processing {len(categories)} POI files ({total_mb:.2f} MB total).")

    start = time.time()

    for i, category in enumerate(categories, start=1):
        msg(f"Processing category: {category}")
        print_progress(start, i, len(categories), name="POI files",
                       unit="mins", show_timelapse=False, show_timestamp=False)
        size_done += poi_sizes[i - 1] / 1024**2
        pct = (size_done / total_mb * 100) if total_mb > 0 else 100
        msg(f"File-size progress: {size_done:.2f} / {total_mb:.2f} MB ({pct:.1f}%)")

        poi_path = find_poi_file(poi_dir, category,
                                 file_prefix=poi_file_prefix, file_suffix=poi_file_suffix)
        msg(f"Reading file: {poi_path.name}")

        poi_sf = gpd.read_file(str(poi_path))
        if len(poi_sf) == 0:
            msg(f"Category '{category}': file is empty — skipping.")
            continue
        if poi_sf.crs != zones_sf.crs:
            poi_sf = poi_sf.to_crs(zones_sf.crs)

        df = poi_sf.drop(columns="geometry").copy()
        df["source_row__"] = range(len(df))
        df = apply_cleaning_rules(df, category, cleaning_rules)
        assert_true(len(df) > 0, f"Category '{category}': no records after cleaning.")

        poi_sf = poi_sf.iloc[df["source_row__"].values].copy()
        df = df.drop(columns="source_row__").reset_index(drop=True)
        poi_sf = poi_sf.reset_index(drop=True)

        zone_ids = assign_zones(poi_sf, zones_sf, zone_id_field, category)
        df["zoneId"] = zone_ids
        keep = zone_ids != None  # noqa: E711
        df, poi_sf = df[keep].reset_index(drop=True), poi_sf[keep].reset_index(drop=True)
        assert_true(len(df) > 0, f"Category '{category}': no POIs within zones.")

        prefix = hash_category_prefix(category)
        df["poi_id"] = [f"{prefix}_{j:06d}" for j in range(1, len(df) + 1)]
        df["category"] = category

        export = poi_sf.copy()
        for col in df.columns:
            if col != "geometry":
                export[col] = df[col].values
        poi_export_all.append(export)

        cat_specs = attractiveness_factors[category]
        df = apply_category_filter(df, cat_specs, category, defaults)
        pct_imp = resolve_category_percentile_imputation(cat_specs, percentile_imputation, category)

        reserved = {"filter", "percentile_imputation"}
        for purpose in (k for k in cat_specs if k not in reserved):
            spec = extract_spec(cat_specs[purpose])
            mv = calc_metric_vector(df, category, purpose, spec, defaults, pct_imp)
            mv = apply_adjuster(mv, df, spec, category, purpose)
            attr = mv * spec["coefficient"]

            tmp = pd.DataFrame({"poi_id": df["poi_id"].values, "zoneId": df["zoneId"].values,
                                "category": category, "purpose": purpose, "attractiveness": attr})
            agg_list.append(tmp.groupby(["zoneId", "purpose"], as_index=False)["attractiveness"].sum())
            poi_export_purpose.append(tmp[["poi_id", "zoneId", "category", "purpose", "attractiveness"]].copy())
            msg(f"  {category}/{purpose}: {len(tmp)} POIs processed.")

    assert_true(len(agg_list) > 0, "No aggregation results generated.")
    final_long = pd.concat(agg_list, ignore_index=True).groupby(
        ["zoneId", "purpose"], as_index=False)["attractiveness"].sum()
    final_wide = final_long.pivot(index="zoneId", columns="purpose", values="attractiveness").reset_index()
    final_wide.columns.name = None
    final_wide = pd.DataFrame({"zoneId": zone_ids_all}).merge(final_wide, on="zoneId", how="left").fillna(0)

    all_pois = pd.concat(poi_export_all, ignore_index=True)
    all_attr = pd.concat(poi_export_purpose, ignore_index=True)

    meta_cols = ["poi_id"] + [c for c in ("id", "name", "origin", "Area") if c in all_pois.columns]
    meta = all_pois[meta_cols].drop_duplicates(subset=["poi_id"]).rename(
        columns={"id": "osm_id", "name": "osm_name"})
    all_attr_ext = all_attr.merge(meta, on="poi_id", how="left")

    id2geom = dict(zip(
        [pid for g in poi_export_all for pid in g["poi_id"]],
        [geom for g in poi_export_all for geom in g.geometry],
    ))
    all_attr_sf = gpd.GeoDataFrame(
        all_attr_ext,
        geometry=all_attr_ext["poi_id"].map(id2geom).values,
        crs=poi_export_all[0].crs if poi_export_all else None,
    )

    if output_csv:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        final_wide.to_csv(str(output_csv), index=False)
        msg(f"Result written: {output_csv}")
    if output_gpkg_all_pois:
        write_gpkg(all_pois, Path(output_gpkg_all_pois), "all_pois")
    if output_gpkg_all_poi_attractiveness:
        write_gpkg(all_attr_sf, Path(output_gpkg_all_poi_attractiveness), "all_poi_attractiveness")

    msg(f"Attractiveness calculation completed in {(time.time() - start) / 60:.2f} min.")

    return {"zone_attractiveness": final_wide, "zone_attractiveness_long": final_long,
            "all_pois": all_pois, "all_poi_attractiveness": all_attr_ext}


# ---------------------------------------------------------------------------
# CLI runner
# ---------------------------------------------------------------------------

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
    out_dir = (resolve(paths["attractiveness_output_root"])
               / paths["attractiveness_output_sub_attractiveness"]
               / f"run_{timestamp}")
    out_csv = out_dir / "attractiveness.csv"
    out_gpkg_pois = out_dir / "all_pois.gpkg"
    out_gpkg_attr = out_dir / "all_poi_attractiveness.gpkg"

    print("=" * 80)
    print("POIs2attractiveness")
    print("=" * 80)
    print(f"\nArea:    {area_name}")
    print(f"Config:  {config_file}")
    print(f"POI dir: {poi_dir}")
    print(f"Zones:   {zones_file}")
    print(f"Factors: {factors_json}")
    print(f"Output:  {out_dir}\n")

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
    det = all_attr.pivot_table(
        index="zoneId", columns=["category", "purpose"],
        values="attractiveness", aggfunc="sum", fill_value=0,
    )
    det.columns = [f"{c}_{p}" for c, p in det.columns]
    det.reset_index().to_csv(
        str(out_dir / "attractiveness_detailed_by_category_purpose.csv"), index=False)
    all_attr.to_csv(
        str(out_dir / "attractiveness_detailed_all_pois.csv"), index=False)


if __name__ == "__main__":
    main()