# OSM2Attractiveness (Python)

Pure-Python implementation of [OSM2Attractiveness](https://github.com/kit-ifv/OSM2Attractiveness) by the [KIT Institute for Transport Studies](https://www.ifv.kit.edu/english/index.php). For methodology, background and general documentation see the original project.

This fork replaces R and Osmosis (Java) with Python equivalents so the entire pipeline runs with a single `pip install`.

| Component | Original | This fork |
|---|---|---|
| OSM filtering | Osmosis (Java) | pyosmium |
| Attractiveness calculation | R (data.table, sf) | pandas, geopandas |
| Evaluation maps (static) | R (ggplot2) | matplotlib |
| Evaluation maps (interactive) | R (leaflet) | plotly |

All filter definitions, configuration files and data formats are unchanged.


## Installation

```sh
git clone https://github.com/m-schi/OSM2AttractivenessPy.git
cd OSM2AttractivenessPy
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requirements: Python 3.10+. All dependencies are in [requirements.txt](requirements.txt).


## Usage

Adapt [config/config_rastatt_example.yaml](config/config_rastatt_example.yaml) or create your own config, then run from the repository root:

```sh
# 1) Filter OSM by categories
python src/make/FilterOSM_pyosmium.py config_rastatt_example

# 2) Convert filtered OSM to POIs
python src/make/OSM2POIs.py config_rastatt_example

# 3) Calculate zone attractiveness
python src/make/POIs2attractiveness.py config_rastatt_example

# Optional: generate evaluation maps
python src/eval/eval_maps.py config_rastatt_example

# Optional: export georeferenced results (GeoJSON/GPKG + QGIS styles)
python src/eval/export_gis.py config_rastatt_example
```

`export_gis.py` joins `attractiveness.csv` onto the zone geometries and writes
`attractiveness_zones.geojson`/`.gpkg` plus QGIS style files (`.qml`) into a
`GIS/` subfolder of the run directory.

The `osmosis_bin` path in the YAML config is no longer needed and is ignored.


## Data Prerequisites

- A raw OSM PBF file — downloaded automatically from [Geofabrik](https://download.geofabrik.de/) when `osm_download_url` is set in the config. Or place it manually under `data/osm-raw/`.
- Attractiveness factors (sample included in `config/locale/`)
- A zones layer (GeoJSON or similar)
- Optional: QGIS to repair buildings geometries (see [original project](https://github.com/kit-ifv/OSM2Attractiveness) for details)


## License

See [LICENSE.md](LICENSE.md).


## Acknowledgments

- Original workflow by [KIT Institute for Transport Studies](https://github.com/kit-ifv/OSM2Attractiveness).
- Python port created with assistance from Claude (Anthropic).
- Thanks to the OpenStreetMap community for providing the data.
