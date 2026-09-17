"""
POIs2attractiveness_helpers.py

Helper functions for POI-to-attractiveness processing.
"""

import binascii
import json
import math
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def print_progress(
    timing_start: float,
    current_step: int,
    total_steps: int,
    name: Optional[str] = None,
    unit: str = "secs",
    digits: int = 0,
    show_timestamp: bool = True,
    show_timelapse: bool = True,
    show_percent: bool = True,
    every_n: int = 1,
    n_parts: Optional[int] = None,
    criterion: bool = True,
):
    """Print a progress message with optional elapsed/remaining time info."""
    elapsed_sec = time.time() - timing_start
    avg = elapsed_sec / current_step if current_step > 0 else 0
    remaining_sec = (total_steps - current_step) * avg

    if unit not in ("secs", "mins"):
        msg(f"Warning: Invalid unit '{unit}'; defaulting to 'secs'.")
        unit = "secs"
    div = 60 if unit == "mins" else 1
    unit_str = "[minutes]" if unit == "mins" else "[seconds]"

    if n_parts is not None:
        if isinstance(n_parts, (int, float)) and n_parts > 0:
            every_n = math.ceil(total_steps / n_parts)

    if not criterion:
        return
    if (current_step % every_n) != 0 and current_step != total_steps and current_step != 1:
        return

    pct = f" [{current_step / total_steps * 100:.1f}%]" if show_percent else ""
    tl = (f"Elapsed:{round(elapsed_sec/div, digits)} - per step:{round(avg/div, digits)} "
          f"- remaining:{round(remaining_sec/div, digits)}{unit_str}. ") if show_timelapse else ""
    ts = (f"ETA: {datetime.fromtimestamp(time.time() + remaining_sec):%Y-%m-%d %H:%M:%S}"
          if show_timestamp else "")

    prefix = f"{name}: " if name else ""
    msg(f"{prefix}Step {current_step} of {total_steps} finished{pct}. {tl}{ts}")


def msg(*args):
    """Print a timestamped log message."""
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {''.join(str(a) for a in args)}")


def assert_true(cond: bool, txt: str):
    if not cond:
        raise ValueError(txt)


def as_char_vec(x) -> list:
    """Convert input to a flat list of strings."""
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    if isinstance(x, (list, tuple)):
        result = []
        for item in x:
            if isinstance(item, (list, tuple)):
                result.extend(str(i) for i in item)
            else:
                result.append(str(item))
        return result
    return [str(x)]


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def find_poi_file(poi_dir: Path, category: str, file_prefix: str = "", file_suffix: str = ".geojson") -> Path:
    """Find a POI file by category name and naming convention."""
    p = poi_dir / f"{file_prefix}{category}{file_suffix}"
    if not p.exists():
        raise FileNotFoundError(
            f"No POI file found for category '{category}'. "
            f"Expected: '{p.name}' in '{poi_dir}'."
        )
    return p


def write_gpkg(gdf: gpd.GeoDataFrame, path_gpkg: Path, layer_name: str):
    """Write a GeoDataFrame to a GeoPackage file."""
    path_gpkg = Path(path_gpkg)
    path_gpkg.parent.mkdir(parents=True, exist_ok=True)
    path_gpkg.unlink(missing_ok=True)
    gdf.to_file(str(path_gpkg), layer=layer_name, driver="GPKG")
    msg(f"GeoPackage written: {path_gpkg} (layer: {layer_name})")


# ---------------------------------------------------------------------------
# Spec extraction
# ---------------------------------------------------------------------------

def extract_spec(spec) -> dict:
    """Extract a purpose specification into a standardised dict."""
    if isinstance(spec, dict) and "metric" in spec and "coefficient" in spec:
        return {
            "metric": str(spec["metric"]),
            "coefficient": float(spec["coefficient"]),
            "area_field": str(spec["area_field"]) if spec.get("area_field") else None,
            "levels_field": str(spec["levels_field"]) if spec.get("levels_field") else None,
            "floor_field": str(spec["floor_field"]) if spec.get("floor_field") else None,
            "max_size": float(spec["max_size"]) if spec.get("max_size") is not None else None,
            "adjuster": spec.get("adjuster"),
        }
    items = list(spec) if isinstance(spec, (list, tuple)) else list(spec.values())
    return {
        "metric": str(items[0]),
        "coefficient": float(items[1]),
        "area_field": None, "levels_field": None, "floor_field": None,
        "max_size": float(items[2]) if len(items) >= 3 else None,
        "adjuster": None,
    }


def pick_first_existing_field(field_names: list, candidates: list) -> Optional[str]:
    for c in candidates:
        if c in field_names:
            return c
    return None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_categories_map(pois: dict):
    """Validate the structure of the attractiveness-factors mapping."""
    assert_true(isinstance(pois, dict) and len(pois) > 0, "'pois' must be a non-empty dict.")
    reserved = {"filter", "percentile_imputation"}
    for cat, spec in pois.items():
        assert_true(isinstance(spec, dict) and len(spec) > 0,
                     f"Category '{cat}' contains no trip-purpose specifications.")
        if "percentile_imputation" in spec:
            assert_true(0 < float(spec["percentile_imputation"]) < 1,
                         f"Category '{cat}': percentile_imputation must be in (0, 1).")
        if "filter" in spec:
            fil = spec["filter"]
            assert_true(isinstance(fil, dict), f"Category '{cat}': filter must be a dict.")
            if "min_area" in fil:
                assert_true(float(fil["min_area"]) >= 0, f"Category '{cat}': filter.min_area must be >= 0.")
        purposes = [k for k in spec if k not in reserved]
        assert_true(len(purposes) > 0, f"Category '{cat}' has no trip-purpose specs after removing reserved keys.")
        for p in purposes:
            parsed = extract_spec(spec[p])
            assert_true(parsed["metric"] in ("Count", "Area", "FloorArea"),
                         f"Unknown metric '{parsed['metric']}' in '{cat}/{p}'.")
            assert_true(parsed["coefficient"] is not None and np.isfinite(parsed["coefficient"]),
                         f"Invalid coefficient in '{cat}/{p}'.")
            if parsed["max_size"] is not None:
                assert_true(np.isfinite(parsed["max_size"]) and parsed["max_size"] > 0,
                             f"Invalid max_size in '{cat}/{p}'.")


# ---------------------------------------------------------------------------
# JSON config loading
# ---------------------------------------------------------------------------

_JSON_METRIC_MAP = {"Area": "Area", "FloorArea": "FloorArea", "Count": "Count"}


def load_attractiveness_factors_from_json(json_path: Path) -> dict:
    """Load categories dict with attractiveness factors from a JSON file."""
    json_path = Path(json_path)
    assert_true(json_path.exists(), f"JSON config not found: {json_path}")
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    assert_true(isinstance(raw, dict) and len(raw) > 0, "JSON config is empty or not an object.")

    reserved = {"filter", "percentile_imputation"}

    def xlate(metric_raw, cat, purp):
        m = _JSON_METRIC_MAP.get(metric_raw)
        assert_true(m is not None, f"'{cat}/{purp}': unknown metric '{metric_raw}'.")
        return m

    def parse_spec(ps, cat, purp):
        if isinstance(ps, dict) and len(ps) > 0:
            r = {"metric": xlate(str(ps["metric"]), cat, purp),
                 "coefficient": float(ps["coefficient"])}
            if "max_size" in ps:
                r["max_size"] = float(ps["max_size"])
            if "adjuster" in ps:
                r["adjuster"] = build_adjuster_from_string(str(ps["adjuster"]), cat, purp)
            return r
        items = list(ps) if isinstance(ps, (list, tuple)) else list(ps.values())
        assert_true(len(items) >= 2, f"'{cat}/{purp}': need at least [metric, coefficient].")
        r = {"metric": xlate(str(items[0]), cat, purp), "coefficient": float(items[1])}
        if len(items) >= 3:
            r["max_size"] = float(items[2])
        return r

    pois = {}
    for cat, cspec in raw.items():
        assert_true(isinstance(cspec, dict) and len(cspec) > 0, f"Category '{cat}' is empty.")
        purposes = {p: parse_spec(cspec[p], cat, p) for p in cspec if p not in reserved}
        assert_true(len(purposes) > 0, f"Category '{cat}' has no purposes.")
        if "percentile_imputation" in cspec:
            purposes["percentile_imputation"] = float(cspec["percentile_imputation"])
        if "filter" in cspec:
            fc = {}
            if "min_area" in cspec["filter"]:
                fc["min_area"] = float(cspec["filter"]["min_area"])
            if "area_field" in cspec["filter"]:
                fc["area_field"] = str(cspec["filter"]["area_field"])
            purposes["filter"] = fc
        pois[cat] = purposes

    msg(f"Loaded {len(pois)} categories from '{json_path.name}'.")
    return pois


# ---------------------------------------------------------------------------
# POI coverage reporting
# ---------------------------------------------------------------------------

def report_poi_coverage(poi_dir: Path, pois: dict, file_prefix: str = "", file_suffix: str = ".geojson"):
    poi_dir = Path(poi_dir)
    found = {f.name.lower(): f.name for f in poi_dir.iterdir() if f.suffix.lower() == ".geojson"}
    expected = {f"{file_prefix}{cat}{file_suffix}": cat for cat in pois}

    extra = sorted(n for lo, n in found.items() if lo not in {e.lower() for e in expected})
    missing = sorted(e for e in expected if e.lower() not in found)

    if extra:
        msg(f"POI-Coverage: {len(extra)} uncovered files: {', '.join(extra)}")
    if missing:
        msg(f"POI-Coverage: {len(missing)} missing files: {', '.join(missing)}")
    return {"extra_in_folder": extra, "missing_in_folder": missing}


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def hash_category_prefix(category_name: str) -> str:
    crc = binascii.crc32(category_name.encode("utf-8")) & 0xFFFFFFFF
    return f"{crc:08x}"[:4]


# ---------------------------------------------------------------------------
# Adjuster builder
# ---------------------------------------------------------------------------

def build_adjuster_from_string(adj_string: str, category: str, purpose: str):
    assert_true(bool(adj_string), f"'{category}/{purpose}': adjuster must be non-empty.")
    match = re.match(r"(\w+)\s*\((.+)\)", adj_string.strip())
    assert_true(match is not None, f"'{category}/{purpose}': cannot parse adjuster '{adj_string}'.")
    fn_name, args_str = match.group(1), match.group(2)
    assert_true(fn_name == "linear_size_multiplier",
                 f"'{category}/{purpose}': unsupported adjuster '{fn_name}'.")

    named, positional = {}, []
    for part in (a.strip() for a in args_str.split(",")):
        if "=" in part:
            k, v = part.split("=", 1)
            named[k.strip()] = float(v)
        else:
            positional.append(float(part))

    if named:
        assert_true(not positional, f"'{category}/{purpose}': mix of named/positional args.")
        req4 = {"size_min", "size_max", "mult_at_min", "mult_at_max"}
        assert_true(set(named) in (req4, req4 | {"size_cap"}),
                     f"'{category}/{purpose}': bad named args.")
        size_min, size_max = named["size_min"], named["size_max"]
        mult_at_min, mult_at_max = named["mult_at_min"], named["mult_at_max"]
        size_cap = named.get("size_cap")
    else:
        assert_true(len(positional) in (4, 5), f"'{category}/{purpose}': need 4 or 5 args.")
        size_min, size_max = positional[0], positional[1]
        if len(positional) == 5:
            size_cap, mult_at_min, mult_at_max = positional[2], positional[3], positional[4]
        else:
            size_cap, mult_at_min, mult_at_max = None, positional[2], positional[3]

    assert_true(size_max > size_min, f"'{category}/{purpose}': need size_max > size_min.")

    from POIs2attractiveness_funcs import linear_size_multiplier

    def adjuster_fn(_df: pd.DataFrame, metric_values: np.ndarray) -> np.ndarray:
        return linear_size_multiplier(metric_values, size_min, size_max,
                                      mult_at_min, mult_at_max, size_cap)
    return adjuster_fn