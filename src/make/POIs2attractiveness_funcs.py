"""
POIs2attractiveness_funcs.py

Core calculation functions for POI-to-attractiveness processing.
Python equivalent of POIs2attractiveness_funcs.r
"""

from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd

from POIs2attractiveness_helpers import (
    assert_true,
    extract_spec,
    msg,
    pick_first_existing_field,
)


# ---------------------------------------------------------------------------
# Adjuster functions
# ---------------------------------------------------------------------------

def linear_size_multiplier(
    size_values: np.ndarray,
    size_min: float,
    size_max: float,
    mult_at_min: float,
    mult_at_max: float,
    size_cap: Optional[float] = None,
) -> np.ndarray:
    """
    Compute a linear size-dependent multiplier for metric values.

    Values below size_min get mult_at_min; values above size_max get mult_at_max
    (or are capped if size_cap is given); values in between are linearly
    interpolated.
    """
    sz = np.asarray(size_values, dtype=float)

    assert_true(
        np.isfinite(size_min) and np.isfinite(size_max) and size_max > size_min,
        "Invalid bounds for linear interpolation.",
    )
    assert_true(
        np.isfinite(mult_at_min) and np.isfinite(mult_at_max),
        "Invalid multiplier values for linear interpolation.",
    )
    if size_cap is not None:
        assert_true(
            np.isfinite(size_cap) and size_cap > size_max,
            "size_cap must be finite and greater than size_max.",
        )

    result = np.full_like(sz, np.nan, dtype=float)
    valid = np.isfinite(sz)
    if not np.any(valid):
        return np.ones_like(sz)

    sz_v = sz[valid]
    t = (sz_v - size_min) / (size_max - size_min)
    t = np.clip(t, 0, 1)
    result[valid] = mult_at_min + t * (mult_at_max - mult_at_min)

    # Cap above the active upper bound
    upper_bound = size_cap if size_cap is not None else size_max
    cap_idx = valid & (sz > upper_bound)
    if np.any(cap_idx):
        result[cap_idx] = mult_at_max * upper_bound / sz[cap_idx]

    # Treat missing/invalid neutrally
    result[~valid] = 1
    return result


# ---------------------------------------------------------------------------
# Metric capping
# ---------------------------------------------------------------------------

def cap_metric_values(
    values: np.ndarray,
    max_size: Optional[float],
    category: str,
    purpose: str,
    metric: str,
) -> np.ndarray:
    """Cap metric values at max_size if defined."""
    if max_size is None:
        return values

    assert_true(
        np.isfinite(max_size) and max_size > 0,
        f"Category '{category}', purpose '{purpose}': 'max_size' must be > 0.",
    )

    v = np.asarray(values, dtype=float)
    idx_cap = np.isfinite(v) & (v > max_size)
    n_capped = int(np.sum(idx_cap))
    msg(
        f"Category '{category}', purpose '{purpose}': applied cap for {metric} "
        f"at max_size={max_size:.6f} to {n_capped} features."
    )
    v[idx_cap] = max_size
    return v


# ---------------------------------------------------------------------------
# Cleaning rules
# ---------------------------------------------------------------------------

def apply_cleaning_rules(
    df: pd.DataFrame,
    category: str,
    cleaning_rules: dict,
) -> pd.DataFrame:
    """Apply filtering/cleaning rules to a category DataFrame."""
    rules = cleaning_rules.get(category)
    if rules is None:
        return df

    n_before = len(df)
    keep = pd.Series(True, index=df.index)

    # require_any: at least one field matches one of the allowed values
    require_any = rules.get("require_any", {})
    if require_any:
        any_keep = pd.Series(False, index=df.index)
        for field, allowed in require_any.items():
            assert_true(field in df.columns,
                         f"Category '{category}': field '{field}' from require_any is missing.")
            allowed_lower = [str(a).lower() for a in (allowed if isinstance(allowed, list) else [allowed])]
            vals = df[field].astype(str).str.strip().str.lower()
            any_keep = any_keep | vals.isin(allowed_lower)
        keep = keep & any_keep

    # require_all: every field must match one of the allowed values
    require_all = rules.get("require_all", {})
    if require_all:
        for field, allowed in require_all.items():
            assert_true(field in df.columns,
                         f"Category '{category}': field '{field}' from require_all is missing.")
            allowed_lower = [str(a).lower() for a in (allowed if isinstance(allowed, list) else [allowed])]
            vals = df[field].astype(str).str.strip().str.lower()
            keep = keep & vals.isin(allowed_lower)

    # exclude_any: exclude if any field matches a blocked value
    exclude_any = rules.get("exclude_any", {})
    if exclude_any:
        ex = pd.Series(False, index=df.index)
        for field, blocked in exclude_any.items():
            assert_true(field in df.columns,
                         f"Category '{category}': field '{field}' from exclude_any is missing.")
            blocked_lower = [str(b).lower() for b in (blocked if isinstance(blocked, list) else [blocked])]
            vals = df[field].astype(str).str.strip().str.lower()
            ex = ex | vals.isin(blocked_lower)
        keep = keep & ~ex

    out = df[keep].copy()
    msg(f"Category '{category}': cleaning reduced records from {n_before} to {len(out)}.")
    return out


# ---------------------------------------------------------------------------
# Category-level helpers
# ---------------------------------------------------------------------------

def extract_category_purpose_specs(category_spec: dict) -> dict:
    """Return the purpose specs from a category spec, excluding reserved keys."""
    return {k: v for k, v in category_spec.items() if k not in ("filter", "percentile_imputation")}


def resolve_category_percentile_imputation(
    category_spec: dict,
    default_percentile: float,
    category: str,
) -> float:
    """Resolve the percentile imputation value for a category."""
    p = category_spec.get("percentile_imputation", default_percentile)
    p = float(p)
    assert_true(0 < p < 1,
                 f"Category '{category}': percentile_imputation must be in (0, 1).")
    return p


def apply_category_filter(
    df: pd.DataFrame,
    category_spec: dict,
    category: str,
    defaults: dict,
) -> pd.DataFrame:
    """Apply category-level area filter if configured."""
    filter_cfg = category_spec.get("filter")
    if filter_cfg is None:
        return df

    out = df.copy()

    if "min_area" in filter_cfg:
        min_area = float(filter_cfg["min_area"])
        area_field = filter_cfg.get("area_field", defaults["area_field"])

        assert_true(np.isfinite(min_area) and min_area >= 0,
                     f"Category '{category}': filter.min_area must be >= 0.")
        assert_true(area_field in out.columns,
                     f"Category '{category}': filter area field '{area_field}' is missing.")

        area_vals = pd.to_numeric(out[area_field], errors="coerce")
        keep = area_vals.isna() | (area_vals >= min_area)
        n_before = len(out)
        out = out[keep].copy()
        msg(
            f"Category '{category}': category filter min_area>={min_area:.6f} "
            f"on field '{area_field}' reduced records from {n_before} to {len(out)}."
        )

    return out


# ---------------------------------------------------------------------------
# Imputation
# ---------------------------------------------------------------------------

def impute_with_percentile(x: np.ndarray, label: str, percentile: float = 0.15) -> dict:
    """Impute missing/non-positive values using the given percentile of valid values."""
    x = np.asarray(x, dtype=float).copy()
    idx_missing = np.where(np.isnan(x))[0]
    if len(idx_missing) == 0:
        return {"values": x, "imputed": 0, "p_n": np.nan}

    valid = x[np.isfinite(x) & (x > 0)]
    assert_true(len(valid) > 0,
                 f"No valid values available to impute '{label}' via the {percentile*100:.0f}th percentile.")

    p_n = float(np.quantile(valid, percentile))
    assert_true(np.isfinite(p_n) and p_n > 0,
                 f"Invalid {percentile*100:.0f}th percentile for '{label}'.")

    x[idx_missing] = p_n
    return {"values": x, "imputed": len(idx_missing), "p_n": p_n}


# ---------------------------------------------------------------------------
# Metric calculation
# ---------------------------------------------------------------------------

def ensure_numeric_field(df: pd.DataFrame, field_name: str, category: str, purpose: str) -> np.ndarray:
    """Extract and validate a numeric field from the DataFrame."""
    assert_true(field_name in df.columns,
                 f"Category '{category}', purpose '{purpose}': required field '{field_name}' is missing.")
    v = pd.to_numeric(df[field_name], errors="coerce").values
    assert_true(np.any(~np.isnan(v)),
                 f"Category '{category}', purpose '{purpose}': field '{field_name}' contains no usable values.")
    return v


def calc_metric_vector(
    df: pd.DataFrame,
    category: str,
    purpose: str,
    spec: dict,
    defaults: dict,
    percentile_imputation: float,
) -> np.ndarray:
    """Calculate the raw metric vector for a purpose specification."""
    metric = spec["metric"]
    area_field = spec.get("area_field") or defaults["area_field"]
    levels_field = spec.get("levels_field") or defaults["levels_field"]
    floor_field = spec.get("floor_field") or defaults["floor_field"]

    if metric == "Count":
        return np.ones(len(df))

    if metric == "Area":
        area = ensure_numeric_field(df, area_field, category, purpose)
        imp = impute_with_percentile(area, f"{category}/{purpose}/{area_field}", percentile_imputation)
        if imp["imputed"] > 0:
            msg(
                f"Category '{category}', purpose '{purpose}': area imputation: "
                f"{imp['imputed']} features imputed; value used (P{percentile_imputation*100:.0f})={imp['p_n']:.6f}."
            )
        else:
            msg(f"Category '{category}', purpose '{purpose}': area imputation: 0 features imputed.")
        return cap_metric_values(imp["values"], spec.get("max_size"), category, purpose, metric)

    if metric == "FloorArea":
        floor_available = floor_field in df.columns
        if floor_available:
            floor_vals = pd.to_numeric(df[floor_field], errors="coerce").values
        else:
            floor_vals = np.full(len(df), np.nan)

        area = ensure_numeric_field(df, area_field, category, purpose)
        area_imp = impute_with_percentile(area, f"{category}/{purpose}/{area_field}", percentile_imputation)
        area2 = area_imp["values"]
        if area_imp["imputed"] > 0:
            msg(
                f"Category '{category}', purpose '{purpose}': area imputation (for floor area): "
                f"{area_imp['imputed']} features imputed; value used (P{percentile_imputation*100:.0f})={area_imp['p_n']:.6f}."
            )
        else:
            msg(f"Category '{category}', purpose '{purpose}': area imputation (for floor area): 0 features imputed.")

        if levels_field in df.columns:
            levels = pd.to_numeric(df[levels_field], errors="coerce").values
        else:
            levels = np.full(len(df), np.nan)
        # Replace invalid levels with default
        invalid_levels = ~np.isfinite(levels) | np.isnan(levels) | (levels <= 0)
        levels[invalid_levels] = defaults["level_default"]

        computed_floor = area2 * levels
        use_floor = np.where(
            np.isfinite(floor_vals) & ~np.isnan(floor_vals) & (floor_vals > 0),
            floor_vals,
            computed_floor,
        )
        return cap_metric_values(use_floor, spec.get("max_size"), category, purpose, metric)

    raise ValueError(f"Metric '{metric}' is not implemented.")


# ---------------------------------------------------------------------------
# Adjuster application
# ---------------------------------------------------------------------------

def apply_adjuster(
    metric_values: np.ndarray,
    df: pd.DataFrame,
    spec: dict,
    category: str,
    purpose: str,
) -> np.ndarray:
    """Apply an optional adjuster function to metric values."""
    adjuster = spec.get("adjuster")
    if adjuster is None:
        return metric_values

    assert_true(callable(adjuster),
                 f"Category '{category}', purpose '{purpose}': 'adjuster' must be callable.")

    import inspect
    n_params = len(inspect.signature(adjuster).parameters)
    if n_params >= 2:
        adj = adjuster(df, metric_values)
    else:
        adj = adjuster(df)

    adj = np.asarray(adj, dtype=float)
    assert_true(len(adj) == len(df),
                 f"Category '{category}', purpose '{purpose}': 'adjuster' must return a vector of length len(df).")

    return metric_values * np.where(np.isnan(adj), 1, adj)


# ---------------------------------------------------------------------------
# Zone assignment
# ---------------------------------------------------------------------------

def assign_zones(
    pois_gdf: gpd.GeoDataFrame,
    zones_gdf: gpd.GeoDataFrame,
    zone_id_field: str,
    category: str,
) -> np.ndarray:
    """
    Assign each POI to a zone using spatial within-join.

    Returns an array of zone IDs (or NaN for unassigned POIs).
    """
    # Spatial join: find which zone each POI falls within
    joined = gpd.sjoin(pois_gdf, zones_gdf[[zone_id_field, "geometry"]], how="left", predicate="within")

    # Handle duplicates (POI in multiple zones): keep first
    duplicated_mask = joined.index.duplicated(keep="first")
    multiple = int(duplicated_mask.sum())
    joined = joined[~duplicated_mask]

    zone_ids = joined[zone_id_field].values.astype(str)
    # Replace 'nan' strings with actual None
    zone_ids = np.where(zone_ids == "nan", None, zone_ids)

    outside = int(np.sum(zone_ids == None))  # noqa: E711

    if multiple > 0:
        msg(f"Category '{category}': {multiple} POIs are in multiple zones; the first assignment is used.")
    if outside > 0:
        msg(f"Category '{category}': {outside} POIs could not be assigned to any zone and will be excluded.")

    return zone_ids


# ---------------------------------------------------------------------------
# Default cleaning rules
# ---------------------------------------------------------------------------

cleaning_rules = {
    "Krankenhaus": {
        "require_any": {
            "building": "hospital",
            "amenity": "hospital",
        }
    }
}