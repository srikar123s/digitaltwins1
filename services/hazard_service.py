import json
import os
import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import xarray as xr
except ImportError:
    xr = None


def build_upstream_map(graph):
    return {
        node: list(graph.predecessors(node))
        for node in graph.nodes
    }

def compute_flood_index(memory, slope, upstream_map, prev_flood,
                        alpha1=0.4, alpha2=0.2, decay=0.4):

    flood = decay * prev_flood

    flood += alpha1 * memory * (1 / (1 + slope))

    for node, upstream_nodes in upstream_map.items():

        if upstream_nodes:

            upstream_mean = sum(prev_flood[j] for j in upstream_nodes) / len(upstream_nodes)

            flood[node] += alpha2 * upstream_mean

    return flood


def compute_saturation(memory, flood):
    return 0.6 * memory + 0.4 * flood



def compute_landslide_stress(saturation, slope, resistance):

    critical_slope = 5

    stress = (
        saturation *
        (slope / (critical_slope + 1e-6)) /
        resistance
    )

    stress = np.clip(stress, 0, None)

    return stress


def compute_composite_risk(flood, landslide):

    flood_norm = (flood - np.min(flood)) / (np.max(flood) - np.min(flood) + 1e-6)
    landslide_norm = (landslide - np.min(landslide)) / (np.max(landslide) - np.min(landslide) + 1e-6)

    return 0.6 * flood_norm + 0.4 * landslide_norm


def normalize(values):

    min_val = np.min(values)
    max_val = np.max(values)

    if max_val - min_val == 0:
        return np.zeros_like(values)

    return (values - min_val) / (max_val - min_val)


def compute_probability(composite_risk, scale=5, theta=0.3):

    return 1 / (1 + np.exp(-scale * (composite_risk - theta)))


def _project_root():
    return Path(__file__).resolve().parent.parent


def _ml_model_path():
    return _project_root() / "ml_models" / "ml_risk_model.pkl"


def _rainfall_dir():
    return _project_root() / "data" / "rainfall"


def _clip01(value):
    return float(np.clip(value, 0.0, 1.0))


def _safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_cell_center(feature):
    geometry = feature.get("geometry") or {}
    coordinates = geometry.get("coordinates") or []
    if geometry.get("type") != "Polygon" or not coordinates:
        return None

    ring = coordinates[0]
    if not ring:
        return None

    longitudes = [point[0] for point in ring if len(point) >= 2]
    latitudes = [point[1] for point in ring if len(point) >= 2]
    if not longitudes or not latitudes:
        return None

    return float(np.mean(latitudes)), float(np.mean(longitudes))


def _physics_features_from_properties(properties, flood_min, flood_max, landslide_min, landslide_max):
    flood_raw = _safe_float(properties.get("flood"))
    landslide_raw = _safe_float(properties.get("landslide"))
    risk_raw = _safe_float(properties.get("risk"))
    probability_raw = _safe_float(properties.get("probability"))

    flood_probability = (flood_raw - flood_min) / (flood_max - flood_min + 1e-6)
    landslide_stress = (landslide_raw - landslide_min) / (landslide_max - landslide_min + 1e-6)

    return {
        "flood_probability": _clip01(flood_probability),
        "landslide_stress": _clip01(landslide_stress),
        "composite_risk": _clip01(risk_raw),
        "probability": _clip01(probability_raw),
        "flood_raw": flood_raw,
        "landslide_raw": landslide_raw,
    }


@lru_cache(maxsize=1)
def load_ml_artifact():
    model_path = _ml_model_path()
    if not model_path.exists():
        return None

    try:
        with open(model_path, "rb") as file_handle:
            return pickle.load(file_handle)
    except Exception:
        # If sklearn or model class dependencies are unavailable at runtime,
        # skip ML enrichment and continue with physics-derived fields.
        return None


@lru_cache(maxsize=1)
def load_rainfall_datasets():
    if xr is None:
        return {}

    datasets = {}
    for year in (2013, 2018, 2022):
        file_path = _rainfall_dir() / f"chirps-v2.0.{year}.days_p05.nc"
        if file_path.exists():
            datasets[year] = xr.open_dataset(file_path)
    return datasets


@lru_cache(maxsize=4096)
def extract_rainfall_features(lat, lon):
    datasets = load_rainfall_datasets()
    if not datasets:
        return {
            "rain_annual_mm": 0.0,
            "rain_monsoon_mm": 0.0,
            "rain_variability_cv": 0.0,
        }

    annual_values = []
    monsoon_values = []
    for year, dataset in datasets.items():
        point_series = dataset["precip"].sel(latitude=lat, longitude=lon, method="nearest")
        annual_total = float(point_series.sum(skipna=True).values)
        monsoon_mask = point_series["time"].dt.month.isin([6, 7, 8, 9])
        monsoon_total = float(point_series.where(monsoon_mask, drop=True).sum(skipna=True).values)
        annual_values.append(annual_total)
        monsoon_values.append(monsoon_total)

    annual_mean = float(np.mean(annual_values)) if annual_values else 0.0
    monsoon_mean = float(np.mean(monsoon_values)) if monsoon_values else 0.0
    annual_std = float(np.std(annual_values)) if annual_values else 0.0
    rain_variability_cv = annual_std / (annual_mean + 1e-6) if annual_mean > 0 else 0.0

    return {
        "rain_annual_mm": annual_mean,
        "rain_monsoon_mm": monsoon_mean,
        "rain_variability_cv": rain_variability_cv,
    }


def _assemble_feature_row(base_features, rainfall_features, region, hazard_type, feature_cols):
    region_map = {
        "assam": 0,
        "uttarakhand": 1,
        "western_ghats": 2,
    }
    hazard_map = {
        "none": 0,
        "flood": 1,
        "landslide": 2,
    }

    flood_probability = base_features["flood_probability"]
    landslide_stress = base_features["landslide_stress"]
    composite_risk = base_features["composite_risk"]
    rain_annual_mm = rainfall_features["rain_annual_mm"]
    rain_monsoon_mm = rainfall_features["rain_monsoon_mm"]
    rain_variability_cv = rainfall_features["rain_variability_cv"]

    row = {
        "flood_probability": flood_probability,
        "landslide_stress": landslide_stress,
        "composite_risk": composite_risk,
        "rain_annual_mm": rain_annual_mm,
        "rain_monsoon_mm": rain_monsoon_mm,
        "rain_variability_cv": rain_variability_cv,
        "flood_x_landslide": flood_probability * landslide_stress,
        "flood_plus_landslide": flood_probability + landslide_stress,
        "max_hazard": max(flood_probability, landslide_stress),
        "hazard_disagreement": abs(flood_probability - landslide_stress),
        "flood_x_rain_annual": flood_probability * rain_annual_mm,
        "landslide_x_rain_monsoon": landslide_stress * rain_monsoon_mm,
        "rain_intensity_ratio": rain_monsoon_mm / (rain_annual_mm + 1e-6),
        "region_encoded": region_map.get(str(region).lower(), 0),
        "disaster_type_encoded": hazard_map.get(str(hazard_type).lower(), 0),
        "flood_probability_log": np.log1p(flood_probability),
        "landslide_stress_log": np.log1p(landslide_stress),
    }

    if "year_norm" in feature_cols:
        row["year_norm"] = 0.0
    if "rain_year_used" in feature_cols:
        row["rain_year_used"] = 0.0

    return row


def _predict_ml_scores_for_feature_row(artifact, feature_row):
    model = artifact["model"]
    scaler = artifact["scaler"]
    feature_cols = artifact["feature_cols"]

    values = pd.DataFrame(
        [[feature_row.get(column, 0.0) for column in feature_cols]],
        columns=feature_cols,
    )
    scaled = scaler.transform(values)
    return float(model.predict_proba(scaled)[0, 1])


def enrich_geojson_with_ml(geojson_data, region, risk_strategy="hybrid"):
    """Enrich a GeoJSON feature collection with ML risk scores.

    risk_strategy:
    - "hybrid": combine physics + ML (historical mode)
    - "ml": use ML score directly (live/dynamic modes)
    """
    artifact = load_ml_artifact()
    if artifact is None:
        for feature in geojson_data.get("features", []):
            properties = feature.setdefault("properties", {})
            fallback_risk = _clip01(_safe_float(properties.get("risk"), _safe_float(properties.get("probability"))))
            properties.setdefault("ml_flood_score", fallback_risk)
            properties.setdefault("ml_landslide_score", fallback_risk)
            properties.setdefault("ml_risk_score", fallback_risk)
            properties.setdefault("hybrid_risk_score", fallback_risk)
            properties.setdefault("physics_risk_score", fallback_risk)
        return geojson_data

    features = geojson_data.get("features", [])
    if not features:
        return geojson_data

    flood_values = []
    landslide_values = []
    for feature in features:
        properties = feature.get("properties") or {}
        flood_values.append(_safe_float(properties.get("flood")))
        landslide_values.append(_safe_float(properties.get("landslide")))

    flood_min = min(flood_values) if flood_values else 0.0
    flood_max = max(flood_values) if flood_values else 1.0
    landslide_min = min(landslide_values) if landslide_values else 0.0
    landslide_max = max(landslide_values) if landslide_values else 1.0

    for feature in features:
        properties = feature.setdefault("properties", {})
        center = _extract_cell_center(feature)
        if center is None:
            continue

        lat, lon = center
        base_features = _physics_features_from_properties(
            properties,
            flood_min,
            flood_max,
            landslide_min,
            landslide_max,
        )
        rainfall_features = extract_rainfall_features(lat, lon)

        runtime_rainfall_mm = _safe_float(properties.get("runtime_rainfall_mm"), -1.0)
        if runtime_rainfall_mm >= 0.0:
            rain_factor = max(0.30, min(3.00, runtime_rainfall_mm / 100.0))
            rainfall_features = {
                "rain_annual_mm": rainfall_features["rain_annual_mm"] * (0.70 + 0.30 * rain_factor),
                "rain_monsoon_mm": rainfall_features["rain_monsoon_mm"] * rain_factor,
                "rain_variability_cv": rainfall_features["rain_variability_cv"] * (0.85 + 0.35 * rain_factor),
            }

        flood_row = _assemble_feature_row(base_features, rainfall_features, region, "flood", artifact["feature_cols"])
        landslide_row = _assemble_feature_row(base_features, rainfall_features, region, "landslide", artifact["feature_cols"])

        flood_score = _predict_ml_scores_for_feature_row(artifact, flood_row)
        landslide_score = _predict_ml_scores_for_feature_row(artifact, landslide_row)
        ml_score = max(flood_score, landslide_score)

        physics_risk = _clip01(base_features["composite_risk"])

        properties["ml_flood_score"] = flood_score
        properties["ml_landslide_score"] = landslide_score
        properties["ml_risk_score"] = ml_score
        if risk_strategy == "ml":
            properties["hybrid_risk_score"] = ml_score
        else:
            properties["hybrid_risk_score"] = 0.3 * physics_risk + 0.7 * ml_score
        properties["physics_risk_score"] = physics_risk

    return geojson_data


def _round_coordinates(coords, precision):
    if not isinstance(coords, list):
        return coords

    if coords and isinstance(coords[0], (int, float)):
        return [round(float(value), precision) for value in coords]

    return [_round_coordinates(item, precision) for item in coords]


def compact_geojson(geojson_data, coordinate_precision=3):
    """Create a lightweight GeoJSON payload with only required properties.

    Coordinate precision of 3 decimal places = ~111m accuracy, which is
    more than sufficient for grid-cell visualisation and cuts file size
    significantly compared to the previous default of 5 decimal places.
    """
    if not geojson_data:
        return geojson_data

    # Only keep what the frontend actually reads.
    # hybrid_risk_score is the frontend risk field:
    # - historical: 70% ML + 30% physics
    # - live/dynamic: ML-only
    required_properties = {
        "cell_id",
        "flood",
        "landslide",
        "risk",
        "probability",
        "hybrid_risk_score",
        "risk_level",
        "normal_prob",
        "heavy_prob",
        "extreme_prob",
    }

    compact_features = []
    for feature in geojson_data.get("features", []):
        properties = feature.get("properties") or {}
        compact_properties = {
            key: round(properties[key], 4) if isinstance(properties[key], float) else properties[key]
            for key in required_properties
            if key in properties
        }

        geometry = feature.get("geometry") or {}
        compact_geometry = {
            "type": geometry.get("type"),
            "coordinates": _round_coordinates(geometry.get("coordinates"), coordinate_precision),
        }

        compact_features.append({
            "type": "Feature",
            "id": feature.get("id"),
            "geometry": compact_geometry,
            "properties": compact_properties,
        })

    return {
        "type": geojson_data.get("type", "FeatureCollection"),
        "features": compact_features,
    }


