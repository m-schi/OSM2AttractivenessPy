#!/usr/bin/env python3
"""
FilterOSM_pyosmium.py -- Filter OSM data by bounding box and categories using pyosmium.

Pure-Python replacement for FilterOSM.py (which requires Osmosis / Java).
Input:  raw OSM PBF file, category filter definitions
Output: per-category .osm XML files for use by OSM2POIs.py

Usage: python FilterOSM_pyosmium.py [config_name]
"""

import os
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import osmium
import yaml


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def resolve_path(path_value):
    if os.path.isabs(path_value) or path_value.startswith("\\\\") or path_value.startswith("//"):
        return os.path.normpath(path_value)
    return os.path.abspath(path_value)


def write_with_retry(output_path, write_fn, max_attempts=5, base_delay=1.5):
    """Runs write_fn() (which opens its own osmium.SimpleWriter and writes to
    output_path) with retries.

    Works around a Windows-only pyosmium/libosmium timing issue where opening
    a SimpleWriter shortly after a previous one (in the same directory, same
    process) intermittently fails with:
        RuntimeError: Open failed for '<path>': The system cannot move the
        file to a different disk drive.
    This happens even though source and destination are on the same drive;
    it appears related to the previous writer's internal background I/O
    threads not having fully released the directory/file yet. A short
    backoff before retrying reliably resolves it.
    """
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            write_fn()
            return
        except RuntimeError as e:
            if "different disk drive" not in str(e):
                raise
            last_err = e
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except OSError:
                    pass
            if attempt < max_attempts:
                wait = base_delay * attempt
                print(f"\n    Write failed (attempt {attempt}/{max_attempts}), "
                      f"retrying in {wait:.1f}s: {e}")
                time.sleep(wait)
    raise last_err


# ---------------------------------------------------------------------------
# Filter-file parsing
# ---------------------------------------------------------------------------

def read_filter_steps(path_to_file):
    """Read filter steps from a file.  One non-empty, non-comment line = one step."""
    steps = []
    with open(path_to_file, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.split("#", 1)[0].strip()
            if line:
                steps.append(line)
    return steps


def parse_filter_step(step_line):
    """Parse one filter line into a list of (key, values) conditions.

    Conditions on the same line are OR-combined (Osmosis semantics).
    ``key=val1,val2`` matches any listed value; ``key=*`` any value;
    ``key=''`` matches an empty-string value.
    """
    conditions = []
    for token in step_line.split():
        if "=" not in token:
            continue
        key, val_str = token.split("=", 1)
        if val_str == "*":
            values = ["*"]
        elif val_str == "''":
            values = [""]
        else:
            values = [v for v in val_str.split(",") if v]
        if key and values:
            conditions.append((key, values))
    return conditions


def parse_filter_file(path):
    """Return list of parsed steps from a filter file."""
    return [c for line in read_filter_steps(path) if (c := parse_filter_step(line))]


# ---------------------------------------------------------------------------
# Tag matching  (replicates Osmosis accept/reject chain)
# ---------------------------------------------------------------------------

class TagMatcher:
    """Check whether an element's tags pass the accept/reject filter chain.

    * All accept steps must pass (AND across lines).
    * Within one step, conditions are OR-combined.
    * Any matching reject step excludes the element.
    """

    def __init__(self, accept_steps, reject_steps):
        self.accept_steps = accept_steps
        self.reject_steps = reject_steps

    def matches(self, tags):
        for conditions in self.accept_steps:
            if not any(self._check(tags, k, v) for k, v in conditions):
                return False
        for conditions in self.reject_steps:
            if any(self._check(tags, k, v) for k, v in conditions):
                return False
        return True

    @staticmethod
    def _check(tags, key, values):
        if key not in tags:
            return False
        if values == ["*"]:
            return True
        tag_val = tags.get(key, "")
        if values == [""]:
            return tag_val == ""
        return tag_val in values


# ---------------------------------------------------------------------------
# Bbox extract  (equivalent to Osmosis --bounding-box)
# ---------------------------------------------------------------------------

def create_bbox_extract(input_path, output_path, bbox):
    """Create a PBF extract for *bbox* = (left, bottom, right, top)."""
    left, bottom, right, top = bbox

    def in_bbox(loc):
        return left <= loc.lon <= right and bottom <= loc.lat <= top

    # -- pass 1: nodes inside the bounding box ---------------------------------
    print("  Pass 1/4: identifying nodes in bounding box ...")
    bbox_node_ids = set()
    for entity in osmium.FileProcessor(str(input_path)):
        if isinstance(entity, osmium.osm.Node):
            if entity.location.valid() and in_bbox(entity.location):
                bbox_node_ids.add(entity.id)
    print(f"    {len(bbox_node_ids):,} nodes in bbox")

    # -- pass 2: ways touching bbox, relations referencing them ----------------
    print("  Pass 2/4: finding ways and relations ...")
    way_ids = set()
    relation_ids = set()
    all_node_ids = set(bbox_node_ids)

    for entity in osmium.FileProcessor(str(input_path)):
        if isinstance(entity, osmium.osm.Way):
            refs = [n.ref for n in entity.nodes]
            if any(r in bbox_node_ids for r in refs):
                way_ids.add(entity.id)
                all_node_ids.update(refs)
        elif isinstance(entity, osmium.osm.Relation):
            hit = False
            for m in entity.members:
                if m.type == "n" and m.ref in bbox_node_ids:
                    hit = True
                    break
                if m.type == "w" and m.ref in way_ids:
                    hit = True
                    break
            if hit:
                relation_ids.add(entity.id)
                for m in entity.members:
                    if m.type == "n":
                        all_node_ids.add(m.ref)
                    elif m.type == "w":
                        way_ids.add(m.ref)
    print(f"    {len(way_ids):,} ways, {len(relation_ids):,} relations")

    # -- pass 3: resolve node refs for ways added via relations ----------------
    print("  Pass 3/4: resolving back-references ...")
    for entity in osmium.FileProcessor(str(input_path)):
        if isinstance(entity, osmium.osm.Way) and entity.id in way_ids:
            for n in entity.nodes:
                all_node_ids.add(n.ref)
    print(f"    total: {len(all_node_ids):,} nodes, {len(way_ids):,} ways, "
          f"{len(relation_ids):,} relations")

    # -- pass 4: write the extract ---------------------------------------------
    print("  Pass 4/4: writing extract ...")

    def _do_write():
        with osmium.SimpleWriter(str(output_path)) as writer:
            for entity in osmium.FileProcessor(str(input_path)):
                if isinstance(entity, osmium.osm.Node) and entity.id in all_node_ids:
                    writer.add_node(entity)
                elif isinstance(entity, osmium.osm.Way) and entity.id in way_ids:
                    writer.add_way(entity)
                elif isinstance(entity, osmium.osm.Relation) and entity.id in relation_ids:
                    writer.add_relation(entity)

    write_with_retry(str(output_path), _do_write)


# ---------------------------------------------------------------------------
# Per-category tag filtering  (manual multi-pass, mirrors create_bbox_extract)
# ---------------------------------------------------------------------------

def filter_category(extract_path, output_path, accept_steps, reject_steps):
    """Apply tag filters to *extract_path* and write an .osm file.

    Matched ways/relations are completed with their referenced nodes/ways,
    equivalent to Osmosis' --used-way/--used-node. Implemented as plain
    multi-pass filtering with osmium.SimpleWriter (like create_bbox_extract)
    rather than osmium.BackReferenceWriter, which triggers a Windows-only
    "cannot move the file to a different disk drive" RuntimeError when its
    internal ThreadPool-backed writer finalizes the output file.
    """
    matcher = TagMatcher(accept_steps, reject_steps)

    # -- pass 1: direct matches; collect node refs of matched ways and
    #            member refs of matched relations --------------------------
    matched_node_ids = set()
    matched_way_ids = set()
    matched_relation_ids = set()
    required_node_ids = set()
    required_way_ids = set()

    for entity in osmium.FileProcessor(str(extract_path)):
        if isinstance(entity, osmium.osm.Node):
            if matcher.matches(entity.tags):
                matched_node_ids.add(entity.id)
        elif isinstance(entity, osmium.osm.Way):
            if matcher.matches(entity.tags):
                matched_way_ids.add(entity.id)
                required_node_ids.update(n.ref for n in entity.nodes)
        elif isinstance(entity, osmium.osm.Relation):
            if matcher.matches(entity.tags):
                matched_relation_ids.add(entity.id)
                for m in entity.members:
                    if m.type == "n":
                        required_node_ids.add(m.ref)
                    elif m.type == "w":
                        required_way_ids.add(m.ref)

    # -- pass 2: resolve node refs for relation-member ways not already
    #            matched by tags -----------------------------------------
    extra_way_ids = required_way_ids - matched_way_ids
    if extra_way_ids:
        for entity in osmium.FileProcessor(str(extract_path)):
            if isinstance(entity, osmium.osm.Way) and entity.id in extra_way_ids:
                required_node_ids.update(n.ref for n in entity.nodes)

    all_way_ids = matched_way_ids | required_way_ids
    all_node_ids = matched_node_ids | required_node_ids

    # -- pass 3: write output ----------------------------------------------
    def _do_write():
        with osmium.SimpleWriter(str(output_path)) as writer:
            for entity in osmium.FileProcessor(str(extract_path)):
                if isinstance(entity, osmium.osm.Node) and entity.id in all_node_ids:
                    writer.add_node(entity)
                elif isinstance(entity, osmium.osm.Way) and entity.id in all_way_ids:
                    writer.add_way(entity)
                elif isinstance(entity, osmium.osm.Relation) and entity.id in matched_relation_ids:
                    writer.add_relation(entity)

    write_with_retry(str(output_path), _do_write)


# ---------------------------------------------------------------------------
# PBF auto-download
# ---------------------------------------------------------------------------

def download_pbf(url, output_path):
    """Download a PBF file if it is not already present."""
    if os.path.exists(output_path):
        return
    print(f"Downloading PBF from {url} ...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    urllib.request.urlretrieve(url, output_path)
    print(f"Download complete: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    script_dir = Path(__file__).resolve().parent

    config_name = sys.argv[1] if len(sys.argv) > 1 else "config_rastatt_example"
    config_file = (script_dir / ".." / ".." / "config" / f"{config_name}.yaml").resolve()

    if not config_file.exists():
        print(f"Error: config file not found: {config_file}")
        sys.exit(1)

    area_cfg = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    paths_cfg = area_cfg["paths"]
    area_name = area_cfg["area_name"]

    bbox = (
        area_cfg["bbox_wgs84"]["left"],
        area_cfg["bbox_wgs84"]["bottom"],
        area_cfg["bbox_wgs84"]["right"],
        area_cfg["bbox_wgs84"]["top"],
    )

    path_osm_base = resolve_path(paths_cfg["osm_raw_base"])
    path_pbf = path_osm_base + ".osm.pbf"
    path_extract = path_osm_base + f"_extract-{area_name}.osm.pbf"
    path_filter_dir = resolve_path(paths_cfg["filter_dir"])
    path_export = resolve_path(paths_cfg["osm_filtered_dir"])

    # Auto-download PBF when a URL is configured
    download_url = paths_cfg.get("osm_download_url")
    if download_url and not os.path.exists(path_pbf):
        download_pbf(download_url, path_pbf)

    if not os.path.exists(path_pbf):
        print(f"Error: OSM PBF file not found: {path_pbf}")
        sys.exit(1)

    print("=" * 80)
    print("FilterOSM (pyosmium)")
    print("Filter OSM data with categories required for the model")
    print("=" * 80)
    print(f"\nArea:          {area_name}")
    print(f"Config:        {config_file}")
    print(f"Input PBF:     {path_pbf}")
    print(f"Bbox extract:  {path_extract}")
    print(f"Filter folder: {path_filter_dir}")
    print(f"Output folder: {path_export}")

    # ---- Step 1: create / reuse bbox extract ---------------------------------
    if os.path.exists(path_extract):
        print(f"\nBbox extract already exists, reusing: {path_extract}")
    else:
        print(f"\nCreating bbox extract ...")
        t0 = time.time()
        create_bbox_extract(path_pbf, path_extract, bbox)
        print(f"Extract created in {time.time() - t0:.1f} s")

    # ---- Step 2: per-category tag filtering ----------------------------------
    os.makedirs(path_export, exist_ok=True)

    categories = [
        "Districts", "Pharmacy", "Doctor", "Bank", "Authority", "Library",
        "LongTermShopping_DIYGardenCenter", "LongTermShopping_FurnitureStore",
        "LongTermShopping_Other", "LongTermShopping_DepartmentClothingElectronics",
        "DailyShopping_BakeryButcherKiosk", "DailyShopping_Drugstore",
        "DailyShopping_Other", "DailyShopping_Supermarket",
        "EV_ChargingStation", "Hairdresser",
        "SwimmingPool", "SwimmingPoolOutdoor", "Beach",
        "Universities", "Hotel", "Kindergarten", "Cinema",
        "Museums", "MuseumsOutdoor", "Theater", "Restaurant",
        "Church", "Hospital", "Park", "Cemetery", "Zoo",
        "AllotmentGardens", "PostOffice", "Playground",
        "FitnessCenter", "SportsHall", "SportsField", "SmallSportsField",
        "Mailbox", "Schools", "RegionalRail", "Buildings",
    ]

    overall_start = time.time()
    total = len(categories)
    print(f"\nProcessing {total} categories ...")
    print(f"Start: {datetime.now():%Y-%m-%d %H:%M:%S}\n")

    for i, category in enumerate(categories, start=1):
        t_cat = time.time()
        filter_path = os.path.join(path_filter_dir, f"filter_{category}.txt")
        reject_path = os.path.join(path_filter_dir, f"reject_{category}.txt")

        if not os.path.exists(filter_path):
            print(f"[{i}/{total}] {category} -- SKIPPED (no filter file)")
            continue

        output_file = os.path.join(path_export, f"{area_name}_{category}.osm")
        if os.path.exists(output_file):
            print(f"[{i}/{total}] {category} -- already exists, skipping")
            continue

        accept_steps = parse_filter_file(filter_path)
        reject_steps = parse_filter_file(reject_path) if os.path.exists(reject_path) else []

        print(f"[{i}/{total}] {category} ...", end=" ", flush=True)
        filter_category(path_extract, output_file, accept_steps, reject_steps)

        dt = time.time() - t_cat
        elapsed = time.time() - overall_start
        avg = elapsed / i
        eta = datetime.fromtimestamp(time.time() + avg * (total - i)).strftime("%H:%M:%S")
        print(f"{dt:.1f}s  (ETA {eta})")

    print(f"\nDONE -- total runtime: {(time.time() - overall_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
