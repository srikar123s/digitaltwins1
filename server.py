import os
import sys
import subprocess
import json
import copy
import glob
from functools import lru_cache
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel

from services.hazard_service import compact_geojson, enrich_geojson_with_ml

try:
    import geopandas as gpd
except Exception:
    gpd = None

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# -------------------------------------------------------
# Tight geographic bounding boxes for each region.
# These clip out cells that fall over ocean, Pakistan,
# China, Bangladesh, Myanmar, etc.
# Format: (min_lat, max_lat, min_lon, max_lon)
# -------------------------------------------------------
REGION_BBOX = {
    "uttarakhand":   (8.4, 37.6, 68.7, 97.4),  # India full boundary
    "assam":         (8.4, 37.6, 68.7, 97.4),  # India full boundary
    "western_ghats": (8.4, 37.6, 68.7, 97.4),  # India full boundary
}

# Additional strict mode guardrails by region to avoid offshore cells.
# Format: (min_lat, max_lat, min_lon, max_lon)
REGION_STRICT_BBOX = {
    "uttarakhand": (28.4, 31.6, 77.0, 81.5),
    "assam": (24.0, 28.8, 89.5, 96.5),
    "western_ghats": (8.0, 21.5, 73.0, 78.8),
}


@lru_cache(maxsize=1)
def load_india_boundary_geometry():
    """Load India country boundary geometry from local geopandas datasets."""
    if gpd is None:
        return None

    try:
        world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
        india = world[world["name"] == "India"]
        if india.empty:
            return None
        return india.geometry.iloc[0]
    except Exception:
        return None


def _centroid_from_feature(feature):
    try:
        coords = feature["geometry"]["coordinates"][0]
        c_lon = sum(c[0] for c in coords) / len(coords)
        c_lat = sum(c[1] for c in coords) / len(coords)
        return c_lat, c_lon
    except Exception:
        return None


def clip_geojson_to_bbox(geojson_data: dict, region: str) -> dict:
    """Remove features outside India boundary (strict), fallback to India bbox."""
    bbox = REGION_BBOX.get(region)
    if bbox is None:
        return geojson_data  # no bbox defined, pass through

    min_lat, max_lat, min_lon, max_lon = bbox
    strict_bbox = REGION_STRICT_BBOX.get(region)
    india_boundary = load_india_boundary_geometry()

    kept = []
    for feature in geojson_data.get("features", []):
        centroid = _centroid_from_feature(feature)
        if centroid is None:
            kept.append(feature)  # keep on error to avoid silent data loss
            continue

        c_lat, c_lon = centroid
        inside_bbox = min_lat <= c_lat <= max_lat and min_lon <= c_lon <= max_lon
        if not inside_bbox:
            continue

        if strict_bbox is not None:
            s_min_lat, s_max_lat, s_min_lon, s_max_lon = strict_bbox
            inside_strict = s_min_lat <= c_lat <= s_max_lat and s_min_lon <= c_lon <= s_max_lon
            if not inside_strict:
                continue

        if india_boundary is not None:
            try:
                # Import locally to avoid hard dependency when shapely isn't present.
                from shapely.geometry import Point
                point = Point(c_lon, c_lat)
                if not india_boundary.covers(point):
                    continue
            except Exception:
                pass

        kept.append(feature)

    return {**geojson_data, "features": kept}


def resolve_simulation_python() -> str:
    """Resolve the safest interpreter for running main.py.

    Priority:
    1) DIGITALTWINS_PYTHON env override
    2) Project venv Python (Windows)
    3) Current interpreter
    """
    env_python = os.environ.get("DIGITALTWINS_PYTHON")
    if env_python and os.path.exists(env_python):
        return env_python

    project_venv_python = os.path.join(BASE_DIR, "venv", "Scripts", "python.exe")
    if os.path.exists(project_venv_python):
        return project_venv_python

    return sys.executable


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _load_fast_base_geojson(region: str):
    """Load the best available cached geojson for fast live/dynamic updates."""
    candidate_paths = [
        os.path.join(OUTPUTS_DIR, f"{region}_historical_summary.geojson"),
        os.path.join(OUTPUTS_DIR, f"{region}_historical.geojson"),
        os.path.join(OUTPUTS_DIR, f"{region}_live_summary.geojson"),
        os.path.join(OUTPUTS_DIR, f"{region}_dynamic_summary.geojson"),
    ]

    for path in candidate_paths:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as file_handle:
                return json.load(file_handle)

    return None


def _apply_fast_rainfall_adjustment(geojson_data: dict, rainfall_mm: float):

    adjusted = copy.deepcopy(geojson_data)
    features = adjusted.get("features", [])

    if not features:
        return adjusted

    rainfall = max(0.0, min(500.0, _safe_float(rainfall_mm, 100.0)))

    # ---------------------------------------------------
    # ✅ ZERO RAIN → NOTHING
    # ---------------------------------------------------
    if rainfall <= 0.0:
        for feature in features:
            properties = feature.setdefault("properties", {})
            properties.update({
                "runtime_rainfall_mm": 0.0,
                "flood": 0.0,
                "landslide": 0.0,
                "risk": 0.0,
                "probability": 0.0,
                "ml_risk_score": 0.0,
                "hybrid_risk_score": 0.0,
                "physics_risk_score": 0.0,
                "risk_level": "low"
            })
        return adjusted

    # ---------------------------------------------------
    # 🌧️ Rain factor
    # ---------------------------------------------------
    rain_factor = rainfall / 500.0   # 0 → 1 scale

    # ---------------------------------------------------
    # 🎯 THRESHOLD (MOST IMPORTANT)
    # ---------------------------------------------------
    # Low rain → only top risky cells active
    # High rain → more cells activate
    activation_threshold = 0.7 - 0.6 * rain_factor
    # Example:
    # rain=0   → threshold=0.7 (only very high cells)
    # rain=250 → threshold=0.4
    # rain=500 → threshold=0.1 (almost all cells)

    for feature in features:
        properties = feature.setdefault("properties", {})

        base_flood = _safe_float(properties.get("flood"))
        base_landslide = _safe_float(properties.get("landslide"))
        base_prob = _clip01(_safe_float(properties.get("probability")))

        # ---------------------------------------------------
        # 🚨 ACTIVATION LOGIC
        # ---------------------------------------------------
        if base_prob < activation_threshold:
            # ❌ Not active → no flood
            flood = 0.0
            landslide = 0.0
            probability = 0.0
        else:
            # ✅ Active → scaled
            scale = (base_prob - activation_threshold) / (1 - activation_threshold + 1e-6)

            flood = base_flood * scale * (1 + rain_factor)
            landslide = base_landslide * scale * (0.8 + rain_factor)
            probability = base_prob * scale * (0.8 + rain_factor)

        risk = 0.58 * flood + 0.42 * landslide

        # Clamp
        flood = _clip01(flood)
        landslide = _clip01(landslide)
        probability = _clip01(probability)
        risk = _clip01(risk)

        properties["runtime_rainfall_mm"] = rainfall
        properties["flood"] = flood
        properties["landslide"] = landslide
        properties["risk"] = risk
        properties["probability"] = probability

        properties["ml_risk_score"] = probability
        properties["hybrid_risk_score"] = probability
        properties["physics_risk_score"] = risk

        if probability < 0.25:
            properties["risk_level"] = "low"
        elif probability < 0.5:
            properties["risk_level"] = "medium"
        elif probability < 0.75:
            properties["risk_level"] = "high"
        else:
            properties["risk_level"] = "extreme"

    print("Rainfall:", rainfall_mm)
    return adjusted

app.add_middleware(GZipMiddleware, minimum_size=1000)  # compress responses > 1 KB
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure outputs directory exists
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Serve the generated GeoJSON maps and plot images statically
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")


def get_geojson_cache_path(region: str, mode: str, include_ml: bool, compact: bool, risk_strategy: str = "hybrid") -> str:
    if include_ml:
        if compact:
            return os.path.join(OUTPUTS_DIR, f"{region}_{mode}_{risk_strategy}_compact.geojson")
        return os.path.join(OUTPUTS_DIR, f"{region}_{mode}_{risk_strategy}.geojson")

    if compact:
        return os.path.join(OUTPUTS_DIR, f"{region}_{mode}_compact.geojson")
    return os.path.join(OUTPUTS_DIR, f"{region}_{mode}.geojson")

@app.get("/api/progress")
async def get_progress():
    progress_file = os.path.join(OUTPUTS_DIR, "progress.json")
    if os.path.exists(progress_file):
        try:
            with open(progress_file, "r") as f:
                return json.load(f)
        except Exception:
            return {"progress": 0, "status": "Starting..."}
    return {"progress": 0, "status": "Starting..."}

class SimulationRequest(BaseModel):
    region: str
    mode: str
    rainfall: float = 100.0


@app.post("/api/generate-plots")
def generate_plots(req: SimulationRequest):
    valid_regions = ["uttarakhand", "western_ghats", "assam"]
    if req.region not in valid_regions:
        raise HTTPException(status_code=400, detail="Invalid region")

    valid_modes = ["live", "historical", "dynamic"]
    if req.mode not in valid_modes:
        raise HTTPException(status_code=400, detail="Invalid mode")

    print(
        f"Generating plots via main.py --region {req.region} --mode {req.mode} "
        f"--rain_input {req.rainfall} --skip_plots 0"
    )

    with open(os.path.join(OUTPUTS_DIR, "progress.json"), "w") as f:
        json.dump({"progress": 1, "status": "Generating plots..."}, f)

    try:
        cmd = [
            resolve_simulation_python(), os.path.join(BASE_DIR, "main.py"),
            "--region", req.region,
            "--mode", req.mode,
            "--rain_input", str(req.rainfall),
            "--skip_plots", "0"
        ]

        process = subprocess.run(
            cmd,
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            check=True
        )

        return {
            "status": "success",
            "message": f"Plots generated for {req.region} in {req.mode} mode.",
            "plots": [
                f"/outputs/{req.region}_{req.mode}_prob_hist.png",
                f"/outputs/{req.region}_{req.mode}_multi_hazard.png",
                f"/outputs/{req.region}_{req.mode}_rainfall_scenario.png"
            ],
            "logs": process.stdout
        }
    except subprocess.CalledProcessError as e:
        print("ERROR generating plots:")
        print(e.stderr)
        raise HTTPException(status_code=500, detail=f"Plot generation failed: {e.stderr}")


@app.get("/api/geojson")
def get_geojson(region: str, mode: str, include_ml: bool = True, compact: bool = True):
    valid_regions = ["uttarakhand", "western_ghats", "assam"]
    valid_modes = ["live", "historical", "dynamic"]

    if region not in valid_regions:
        raise HTTPException(status_code=400, detail="Invalid region")
    if mode not in valid_modes:
        raise HTTPException(status_code=400, detail="Invalid mode")

    raw_geojson_path = os.path.join(OUTPUTS_DIR, f"{region}_{mode}.geojson")
    if not os.path.exists(raw_geojson_path):
        raise HTTPException(status_code=404, detail="GeoJSON file not found")

    risk_strategy = "hybrid" if mode == "historical" else "ml"

    cache_path = get_geojson_cache_path(region, mode, include_ml, compact, risk_strategy)

    # Historical: serve from stable cache when available.
    if mode == "historical" and include_ml and os.path.exists(cache_path):
        try:
            if os.path.getmtime(cache_path) >= os.path.getmtime(raw_geojson_path):
                with open(cache_path, "r", encoding="utf-8") as file_handle:
                    return json.load(file_handle)
        except OSError:
            pass

    # For live/dynamic: the data was already prepared by _apply_fast_rainfall_adjustment
    # which set all fields (flood, landslide, risk, probability, hybrid_risk_score).
    # Calling enrich_geojson_with_ml here would: load 3× CHIRPS GB NetCDF files,
    # run per-cell lat/lon queries (~1800 calls), AND overwrite the rainfall-adjusted
    # values with static climatology — causing the multi-minute hang + wrong output.
    # So for live/dynamic we skip ML enrichment and just clip+compact.
    run_ml_enrichment = include_ml and (mode in ["historical", "live"])
    # For historical: prefer the 1MB summary over the 600MB+ full file as source.
    if mode == "historical":
        summary_path = os.path.join(OUTPUTS_DIR, f"{region}_{mode}_summary.geojson")
        source_path = summary_path if os.path.exists(summary_path) else raw_geojson_path
    else:
        # For live/dynamic the raw path is already the fast-adjusted 1MB file.
        source_path = raw_geojson_path

    try:
        with open(source_path, "r", encoding="utf-8") as file_handle:
            geojson_data = json.load(file_handle)

        if run_ml_enrichment:
            geojson_data = enrich_geojson_with_ml(geojson_data, region, risk_strategy=risk_strategy)

        geojson_data = clip_geojson_to_bbox(geojson_data, region)

        if compact:
            geojson_data = compact_geojson(geojson_data)

        # Cache the result for historical mode only (live/dynamic change every request).
        if mode == "historical" and cache_path != raw_geojson_path:
            with open(cache_path, "w", encoding="utf-8") as file_handle:
                json.dump(geojson_data, file_handle, separators=(",", ":"))

        return geojson_data
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"GeoJSON file is invalid JSON: {exc}")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Unable to read GeoJSON file: {exc}")

@app.post("/api/simulate")
def run_simulation(req: SimulationRequest):
    valid_regions = ["uttarakhand", "western_ghats", "assam"]
    if req.region not in valid_regions:
        raise HTTPException(status_code=400, detail="Invalid region")
    
    # Modes: "live", "historical", "dynamic"
    valid_modes = ["live", "historical", "dynamic"]
    if req.mode not in valid_modes:
        raise HTTPException(status_code=400, detail="Invalid mode")
        
    cached_geojson = os.path.join(OUTPUTS_DIR, f"{req.region}_{req.mode}.geojson")
    if req.mode == "historical" and os.path.exists(cached_geojson):
        print(f"Cache hit for {req.region} in historical mode.")
        return {
            "status": "success", 
            "message": f"Loaded cached simulation for {req.region} in {req.mode} mode.", 
            "geojson": f"/outputs/{req.region}_{req.mode}.geojson",
            "plots": [
                f"/outputs/{req.region}_{req.mode}_prob_hist.png",
                f"/outputs/{req.region}_{req.mode}_multi_hazard.png",
                f"/outputs/{req.region}_{req.mode}_rainfall_scenario.png"
            ],
            "logs": "Loaded instantly from cache."
        }

    if req.mode == "dynamic":
        try:
            base_geojson = _load_fast_base_geojson(req.region)
            if base_geojson is None:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"No base cached GeoJSON available for fast {req.mode} mode in {req.region}. "
                        "Run historical once for this region."
                    )
                )

            # Apply rainfall scaling
            fast_geojson = _apply_fast_rainfall_adjustment(base_geojson, req.rainfall)

            # Clip to region and compact right here — no second /api/geojson round-trip needed.
            fast_geojson = clip_geojson_to_bbox(fast_geojson, req.region)
            compact_result = compact_geojson(fast_geojson)

            # Also persist full version so /api/geojson static fallback still works.
            with open(cached_geojson, "w", encoding="utf-8") as file_handle:
                json.dump(fast_geojson, file_handle, separators=(",", ":"))

            with open(os.path.join(OUTPUTS_DIR, "progress.json"), "w") as f:
                json.dump({"progress": 100, "status": f"{req.mode.capitalize()} simulation complete"}, f)

            # Return the GeoJSON inline — frontend uses resData.inline_geojson directly.
            return {
                "status": "success",
                "message": f"Fast simulation completed for {req.region} in {req.mode} mode.",
                "inline_geojson": compact_result,
                "geojson": f"/outputs/{req.region}_{req.mode}.geojson",
                "plots": None,
                "logs": "Fast path — results embedded in response."
            }
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Fast simulation failed: {exc}")
    
    skip_plots = req.mode == "dynamic"

    print(
        f"Executing main.py --region {req.region} --mode {req.mode} "
        f"--rain_input {req.rainfall} --skip_plots {1 if skip_plots else 0}"
    )
    
    with open(os.path.join(OUTPUTS_DIR, "progress.json"), "w") as f:
        json.dump({"progress": 1, "status": "Waking up simulation engine..."}, f)
    
    try:
        # Create full command calling main.py using the activated python environment
        cmd = [
            resolve_simulation_python(), os.path.join(BASE_DIR, "main.py"), 
            "--region", req.region, 
            "--mode", req.mode,
            "--rain_input", str(req.rainfall),
            "--skip_plots", "1" if skip_plots else "0"
        ]
        
        process = subprocess.run(
            cmd, 
            cwd=BASE_DIR,
            capture_output=True, 
            text=True, 
            check=True
        )
        
        plots = None
        if not skip_plots:
            plots = [
                f"/outputs/{req.region}_{req.mode}_prob_hist.png",
                f"/outputs/{req.region}_{req.mode}_multi_hazard.png",
                f"/outputs/{req.region}_{req.mode}_rainfall_scenario.png"
            ]

        return {
            "status": "success", 
            "message": f"Simulation completed for {req.region} in {req.mode} mode.", 
            "geojson": f"/outputs/{req.region}_{req.mode}.geojson",
            "plots": plots,
            "logs": process.stdout
        }
    except subprocess.CalledProcessError as e:
        print("ERROR running script:")
        print(e.stderr)
        raise HTTPException(status_code=500, detail=f"Simulation failed: {e.stderr}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8345)
