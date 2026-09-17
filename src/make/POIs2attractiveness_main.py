"""
POIs2attractiveness_main.py

Main processing function for POI-to-attractiveness calculation.
Python equivalent of POIs2attractiveness_main.r
"""

import os
import time
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd

from POIs2attractiveness_helpers import (
    assert_true,
    extract_spec,
    find_poi_file,
    hash_category_prefix,
    msg,
    print_progress,
    report_poi_coverage,
    validate_categories_map,
    write_gpkg,
)
from POIs2attractiveness_funcs import (
    apply_adjuster,
    apply_category_filter,
    apply_cleaning_rules,
    assign_zones,
    calc_metric_vector,
    extract_category_purpose_specs,
    resolve_category_percentile_imputation,
)


def calculate_attractiveness(
    poi_dir: str,
    zones_file: str,
    attractiveness_factors: dict,
    poi_file_prefix: str = "",
    poi_file_suffix: str = ".geojson",
    zone_id_field: str = "zoneId",
    cleaning_rules: Optional[dict] = None,
    percentile_imputation: float = 0.15,
    defaults: Optional[dict] = None,
    output_csv: Optional[str] = None,
    output_gpkg_all_pois: Optional[str] = None,
    output_gpkg_all_poi_attractiveness: Optional[str] = None,
) -> dict:
    """
    Calculate zone-level attractiveness from POI files.

    Parameters
    ----------
    poi_dir : str
        Directory containing POI GeoJSON files.
    zones_file : str
        Path to zone geometries (GeoJSON or similar).
    attractiveness_factors : dict
        Categories mapping with attractiveness factor specifications.
    poi_file_prefix : str
        Prefix for POI filenames.
    poi_file_suffix : str
        Suffix for POI filenames.
    zone_id_field : str
        Name of the zone ID field in the zone file.
    cleaning_rules : dict, optional
        Cleaning rules per category.
    percentile_imputation : float
        Default percentile for area imputation.
    defaults : dict, optional
        Default field names and values.
    output_csv : str, optional
        Path to write the attractiveness CSV.
    output_gpkg_all_pois : str, optional
        Path to write the all-POIs GeoPackage.
    output_gpkg_all_poi_attractiveness : str, optional
        Path to write the POI attractiveness GeoPackage.

    Returns
    -------
    dict with keys:
        zone_attractiveness : pd.DataFrame (wide format)
        zone_attractiveness_long : pd.DataFrame (long format)
        all_pois : gpd.GeoDataFrame
        all_poi_attractiveness : pd.DataFrame
    """
    if cleaning_rules is None:
        cleaning_rules = {}
    if defaults is None:
        defaults = {
            "area_field": "Area",
            "levels_field": "Level_number",
            "floor_field": "FloorArea",
            "level_default": 1,
        }

    msg("Starting attractiveness calculation.")
    validate_categories_map(attractiveness_factors)

    assert_true(os.path.isdir(poi_dir), f"POI folder does not exist: {poi_dir}")
    assert_true(os.path.exists(zones_file), f"Zone file does not exist: {zones_file}")
    report_poi_coverage(poi_dir, attractiveness_factors, file_prefix=poi_file_prefix, file_suffix=poi_file_suffix)

    zones_sf = gpd.read_file(zones_file)
    assert_true(len(zones_sf) > 0, "Zone file contains no features.")
    assert_true(zone_id_field in zones_sf.columns, f"Zone field '{zone_id_field}' is missing in the zone file.")
    zones_sf[zone_id_field] = zones_sf[zone_id_field].astype(str)

    zone_ids_all = zones_sf[zone_id_field].unique().tolist()
    agg_list = []
    poi_export_list_all_pois = []
    poi_export_list_by_purpose = []

    # File sizes for progress reporting
    categories = list(attractiveness_factors.keys())
    poi_filenames = [os.path.join(poi_dir, f"{poi_file_prefix}{cat}{poi_file_suffix}") for cat in categories]
    poi_sizes = [os.path.getsize(f) if os.path.exists(f) else 0 for f in poi_filenames]
    total_size_mb = sum(poi_sizes) / (1024 * 1024)
    size_done = 0
    n_files = len(categories)
    msg(f"Processing {n_files} POI files with a total size of {total_size_mb:.2f} MB.")

    start = time.time()
    msg(f"Processing start: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    for i, category in enumerate(categories, start=1):
        msg(f"Processing category: {category}")
        print_progress(start, i, n_files, name="POI files", unit="mins", show_timelapse=False, show_timestamp=False)
        size_done += poi_sizes[i - 1] / (1024 * 1024)
        pct = (size_done / total_size_mb * 100) if total_size_mb > 0 else 100
        msg(f"File-size progress: {size_done:.2f} MB of {total_size_mb:.2f} MB ({pct:.1f}%)")

        poi_file = find_poi_file(poi_dir, category, file_prefix=poi_file_prefix, file_suffix=poi_file_suffix)
        msg(f"Reading file: {os.path.basename(poi_file)}")

        poi_sf = gpd.read_file(poi_file)
        if len(poi_sf) == 0:
            msg(f"Category '{category}': file is empty. Skipping category.")
            continue

        # CRS alignment
        if poi_sf.crs != zones_sf.crs:
            poi_sf = poi_sf.to_crs(zones_sf.crs)

        # Work with a plain DataFrame for attribute operations
        this_category_pois_df = poi_sf.drop(columns="geometry").copy()
        this_category_pois_df["source_row__"] = range(len(this_category_pois_df))

        this_category_pois_df = apply_cleaning_rules(this_category_pois_df, category, cleaning_rules)
        assert_true(len(this_category_pois_df) > 0, f"Category '{category}': no records remain after cleaning.")

        poi_sf = poi_sf.iloc[this_category_pois_df["source_row__"].values].copy()
        this_category_pois_df = this_category_pois_df.drop(columns="source_row__").reset_index(drop=True)
        poi_sf = poi_sf.reset_index(drop=True)

        # Assign zones
        zone_ids = assign_zones(poi_sf, zones_sf, zone_id_field, category)
        this_category_pois_df["zoneId"] = zone_ids
        keep_idx = zone_ids != None  # noqa: E711
        this_category_pois_df = this_category_pois_df[keep_idx].reset_index(drop=True)
        poi_sf = poi_sf[keep_idx].reset_index(drop=True)
        assert_true(len(this_category_pois_df) > 0, f"Category '{category}': no POIs within zones.")

        # Generate deterministic IDs
        n_pois_category = len(this_category_pois_df)
        category_prefix = hash_category_prefix(category)
        poi_ids = [f"{category_prefix}_{j:06d}" for j in range(1, n_pois_category + 1)]
        this_category_pois_df["poi_id"] = poi_ids
        this_category_pois_df["category"] = category

        # Build export GeoDataFrame for all POIs
        poi_export_base = poi_sf.copy()
        for col in this_category_pois_df.columns:
            if col != "geometry":
                poi_export_base[col] = this_category_pois_df[col].values
        poi_export_list_all_pois.append(poi_export_base)

        cat_specs = attractiveness_factors[category]

        # Category-specific overrides
        this_category_pois_df = apply_category_filter(this_category_pois_df, cat_specs, category, defaults)
        percentile_this = resolve_category_percentile_imputation(cat_specs, percentile_imputation, category)

        # Process each trip purpose
        reserved = {"filter", "percentile_imputation"}
        for purpose in [k for k in cat_specs if k not in reserved]:
            spec = extract_spec(cat_specs[purpose])

            metric_vals = calc_metric_vector(this_category_pois_df, category, purpose, spec, defaults, percentile_this)
            metric_vals_adjusted = apply_adjuster(metric_vals, this_category_pois_df, spec, category, purpose)
            attractiveness_vals = metric_vals_adjusted * spec["coefficient"]

            tmp = pd.DataFrame({
                "poi_id": this_category_pois_df["poi_id"].values,
                "zoneId": this_category_pois_df["zoneId"].values,
                "category": category,
                "purpose": purpose,
                "attractiveness": attractiveness_vals,
            })

            tmp_agg = tmp.groupby(["zoneId", "purpose"], as_index=False)["attractiveness"].sum()
            agg_list.append(tmp_agg)

            poi_export_list_by_purpose.append(
                tmp[["poi_id", "zoneId", "category", "purpose", "attractiveness"]].copy()
            )

            msg(f"Category '{category}', purpose '{purpose}': {len(tmp)} POIs processed.")

    assert_true(len(agg_list) > 0, "No aggregation results were generated.")
    all_agg = pd.concat(agg_list, ignore_index=True)

    # Aggregate by zoneId and purpose
    final_long = all_agg.groupby(["zoneId", "purpose"], as_index=False)["attractiveness"].sum()

    # Pivot to wide format
    final_wide = final_long.pivot(index="zoneId", columns="purpose", values="attractiveness").reset_index()
    final_wide.columns.name = None

    # Ensure all zones are present
    all_zones_df = pd.DataFrame({"zoneId": zone_ids_all})
    final_wide = all_zones_df.merge(final_wide, on="zoneId", how="left")
    final_wide = final_wide.fillna(0)

    # Combine all POIs
    all_pois = pd.concat(poi_export_list_all_pois, ignore_index=True)

    # Combine all POI attractiveness
    all_poi_attractiveness = pd.concat(poi_export_list_by_purpose, ignore_index=True)

    # Merge with POI metadata
    poi_meta_cols = ["poi_id"]
    for col_name in ["id", "name", "origin", "Area"]:
        if col_name in all_pois.columns:
            poi_meta_cols.append(col_name)

    poi_meta = all_pois[poi_meta_cols].drop_duplicates(subset=["poi_id"]).copy()
    rename_map = {}
    if "id" in poi_meta.columns:
        rename_map["id"] = "osm_id"
    if "name" in poi_meta.columns:
        rename_map["name"] = "osm_name"
    poi_meta = poi_meta.rename(columns=rename_map)

    all_poi_attractiveness_extended = all_poi_attractiveness.merge(poi_meta, on="poi_id", how="left")

    # Attach geometry
    poi_ids_all = []
    geoms_all = []
    for gdf in poi_export_list_all_pois:
        poi_ids_all.extend(gdf["poi_id"].tolist())
        geoms_all.extend(gdf.geometry.tolist())

    poi_id_to_geom = dict(zip(poi_ids_all, geoms_all))
    geom_series = all_poi_attractiveness_extended["poi_id"].map(poi_id_to_geom)
    all_poi_attractiveness_extended_sf = gpd.GeoDataFrame(
        all_poi_attractiveness_extended,
        geometry=geom_series.values,
        crs=poi_export_list_all_pois[0].crs if poi_export_list_all_pois else None,
    )

    # Export results
    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        final_wide.to_csv(output_csv, index=False)
        msg(f"Result written: {output_csv}")

    if output_gpkg_all_pois:
        write_gpkg(all_pois, output_gpkg_all_pois, layer_name="all_pois")

    if output_gpkg_all_poi_attractiveness:
        write_gpkg(all_poi_attractiveness_extended_sf, output_gpkg_all_poi_attractiveness, layer_name="all_poi_attractiveness")

    elapsed_mins = (time.time() - start) / 60
    msg("Attractiveness calculation completed.")
    msg(f"Duration: {elapsed_mins:.2f} minutes.")

    return {
        "zone_attractiveness": final_wide,
        "zone_attractiveness_long": final_long,
        "all_pois": all_pois,
        "all_poi_attractiveness": all_poi_attractiveness_extended,
    }
