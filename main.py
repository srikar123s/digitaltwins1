# ==========================================
# Digital Twin – Terrain + Graph + Hazard
# ==========================================

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--region', type=str, default='uttarakhand')
parser.add_argument('--mode', type=str, default='historical')
parser.add_argument('--rain_input', type=float, default=100.0)
parser.add_argument('--skip_plots', type=int, default=0)
args, _ = parser.parse_known_args()

MODE = args.mode
REGION = args.region
DYNAMIC_RAIN = args.rain_input
SKIP_PLOTS = bool(args.skip_plots)

import os
import time

def write_progress(percent, msg):
    try:
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/progress.json", "w") as f:
            import json
            json.dump({"progress": percent, "status": msg}, f)
    except:
        pass

write_progress(5, "Loading Configuration...")
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
from tqdm import tqdm
from rasterio.features import rasterize
from services.live_rainfall_service import fetch_live_rainfall
from services.refinement_service import refine_cells
from services.memory_service import update_memory
from services.hazard_service import build_upstream_map
from services.hazard_service import (
    compute_flood_index,
    compute_saturation,
    compute_landslide_stress,
    compute_composite_risk,
    compute_probability,
)

from services.rainfall_service import (
    load_chirps_subset,
    build_rainfall_tree,
    precompute_rainfall_mapping,
    map_rainfall_to_grid_fast,
)

from core.reproject import reproject_to_utm
from core.grid import generate_base_grid
from core.graph import build_downhill_graph
from core.state import initialize_state

from core.terrain import (
    compute_slope,
    compute_mean_slope_per_cell,
    compute_mean_elevation_per_cell,
)

from core.hydrology import compute_flow_accumulation

# -----------------------------------
# REGION CONFIG (NEW)
# -----------------------------------

# Hardcoded REGION removed, handled by argparse at the top

REGION_CONFIG = {
    "western_ghats": {
        "dem": "data\\3dem\western_ghats_dem.tif",
        "utm": "data/dem/wg_utm_small.tif",
        "flow": "data/dem/wg_flow.npy",
        "output": "outputs/western_ghats.geojson",
        "lat": 10.0,
        "lon": 75.5,
        "rainfall":"data/rainfall/chirps-v2.0.2018.days_p05.nc"
    },
    "assam": {
        "dem": "data\\3dem\\assam_small.tif",
        "utm": "data/dem/assam_utm_small.tif",
        "flow": "data/dem/assam_flow.npy",
        "output": "outputs/assam.geojson",
        "lat": 25.0,
        "lon": 90.0,
        "rainfall":"data/rainfall/chirps-v2.0.2018.days_p05.nc"
    },
    "uttarakhand": {
        "dem": "data\\3dem\\uttarakhand_dem_small.tif",
        "utm": "data/dem/uk_utm_small.tif",
        "flow": "data/dem/uk_flow.npy",
        "output": "outputs/uttarakhand.geojson",
        "lat": 30.0,
        "lon": 80.0,
        "rainfall":"data/rainfall/chirps-v2.0.2018.days_p05.nc"
    }
}

cfg = REGION_CONFIG[REGION]

print("Digital Twin started")


# -----------------------------------
# Safe normalize
# -----------------------------------

def normalize(arr):

    mn = np.min(arr)
    mx = np.max(arr)

    if mx - mn == 0:
        return np.zeros_like(arr)

    return (arr - mn) / (mx - mn)


# -----------------------------------
# Digital Twin State Evolution
# -----------------------------------

def evolve_state(prev_state, rainfall, slope, flow, upstream_map):

    memory = update_memory(
        prev_state["memory"],
        rainfall,
        decay=0.6
    )

    flow_factor = np.log1p(flow)

    flood = compute_flood_index(
    memory * (1 + flow_factor),
    slope,
    upstream_map,
    prev_state["flood_index"]
    )

    river_boost = np.where(
        flow > np.percentile(flow, 90),
        1.5,
        1.0
    )

    flood *= river_boost

    saturation = compute_saturation(memory, flood)

    stress = compute_landslide_stress(
        saturation,
        slope,
        prev_state["resistance"]
    )

    risk = compute_composite_risk(flood, stress)

    return {
        "memory": memory,
        "flood_index": flood,
        "saturation": saturation,
        "landslide_stress": stress,
        "composite_risk": risk,
        "resistance": prev_state["resistance"],
        "theta": prev_state["theta"]
    }


# -----------------------------------
# Self Calibration
# -----------------------------------

def calibration_update(theta, probability, observed):

    error = observed - probability
    lr = 0.1

    return theta + lr * np.mean(error)


# -----------------------------------
# Adaptive Refinement
# -----------------------------------

def adaptive_refinement(grid, probability, state, max_depth=2):

    # Use an ABSOLUTE threshold (0.5) so that low-rainfall runs don't always
    # produce "extreme" cells just because percentile logic always picks top 10%.
    abs_threshold = 0.5
    percentile_threshold = np.percentile(probability, 90)
    # Take the higher of the two so we only refine genuinely high-risk cells.
    threshold = max(abs_threshold, percentile_threshold)

    cells = np.where(
        (probability >= threshold) &
        (grid["level"].values < max_depth)
    )[0]
    if len(cells) > 200:
        cells = cells[:200]

    print("Cells suggested for refinement:", len(cells))

    if len(cells) == 0:
        return grid, state

    grid, state = refine_cells(grid, state, cells)
    print("New grid size:", len(grid))

    return grid, state


# -----------------------------------
# Reproject DEM
# -----------------------------------

#input_dem = "data/dem/kerala_merged_dem.tif"
#utm_dem = "data/dem/kerala_utm_dem.tif"
# Override with region config
input_dem = cfg["dem"]
utm_dem = cfg["utm"]

def compute_mean_flow_per_cell(grid, flow_acc, transform):

    inv_transform = ~transform
    centroids = grid.geometry.centroid
    cols, rows = inv_transform * (centroids.x.values, centroids.y.values)
    rows = np.clip(np.round(rows).astype(int), 0, flow_acc.shape[0] - 1)
    cols = np.clip(np.round(cols).astype(int), 0, flow_acc.shape[1] - 1)
    return flow_acc[rows, cols]

def compute_mean_curvature_per_cell(grid, dem, transform):

    dem_curvature = compute_curvature_window(dem)
    
    inv_transform = ~transform
    centroids = grid.geometry.centroid
    cols, rows = inv_transform * (centroids.x.values, centroids.y.values)
    rows = np.clip(np.round(rows).astype(int), 0, dem.shape[0] - 1)
    cols = np.clip(np.round(cols).astype(int), 0, dem.shape[1] - 1)
    return dem_curvature[rows, cols]



def filter_grid_by_dem(grid, dem, transform, nodata):

    inv_transform = ~transform
    centroids = grid.geometry.centroid
    cols, rows = inv_transform * (centroids.x.values, centroids.y.values)
    rows = np.clip(np.round(rows).astype(int), 0, dem.shape[0] - 1)
    cols = np.clip(np.round(cols).astype(int), 0, dem.shape[1] - 1)
    
    vals = dem[rows, cols]
    if nodata is not None:
        is_valid = vals != nodata
    else:
        is_valid = ~np.isnan(vals)
        
    return grid[is_valid].copy()

def compute_curvature_window(window):

    dzdx = np.gradient(window, axis=1)
    dzdy = np.gradient(window, axis=0)

    d2zdx2 = np.gradient(dzdx, axis=1)
    d2zdy2 = np.gradient(dzdy, axis=0)

    curvature = d2zdx2 + d2zdy2

    return curvature
if not os.path.exists(utm_dem):
    print("Creating UTM file...")
    reproject_to_utm(input_dem, utm_dem)
else:
    print("Using existing UTM file")


# -----------------------------------
# Load DEM
# -----------------------------------

with rasterio.open(utm_dem) as src:

    dem = src.read(
    1,
    out_shape=(
        src.height // 6,
        src.width // 6
    ),
        resampling=rasterio.enums.Resampling.average
    ).astype(np.float32)

    orig_transform = src.transform

    # Scale transformation precisely to preserve Geographic bounds over downsampled dimensions
    transform = src.transform * rasterio.Affine.scale(
    src.width / (src.width // 6),
    src.height / (src.height // 6)
    )

    bounds = src.bounds
    crs = src.crs
    nodata = src.nodata

print("DEM loaded")


# -----------------------------------
# Flow accumulation
# -----------------------------------

#flow_file = "data/dem/flow_acc.npy"
flow_file = cfg["flow"]
print("Flow file exists:", os.path.exists(flow_file))
if os.path.exists(flow_file):

    flow_acc = np.load(flow_file)
    print("Loaded cached flow")

else:

    print("Computing flow accumulation (this may take time)...")

    flow_acc = compute_flow_accumulation(utm_dem)

    np.save(flow_file, flow_acc)

    print("Flow accumulation computed and saved")

# -----------------------------------
# Mean flow helper
# -----------------------------------

# -----------------------------------
# Memory Cache Logic
# -----------------------------------
slope_raster = compute_slope(dem, transform)

import pickle
cache_path = f"outputs/{REGION}_terrain_cache.pkl"
is_cached = os.path.exists(cache_path)

if is_cached:
    print(f"Loading terrain cache for {REGION}...")
    write_progress(15, "Loading Terrain Cache...")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    grid = cache["grid"]
    mean_slope = cache["mean_slope"]
    mean_elevation = cache["mean_elevation"]
    flow_acc_mean = cache["flow_acc_mean"]
    mean_curvature = cache["mean_curvature"]
    G = cache["G"]
    upstream_map = build_upstream_map(G)
else:
    # -----------------------------------
    # Grid generation
    # -----------------------------------
    grid = generate_base_grid(bounds, resolution_m=2000)
    grid = grid.set_crs(crs)

    write_progress(25, "Generating Grid Geometry...")
    print("Base grid cells:", len(grid))
    grid = filter_grid_by_dem(grid, dem, transform, nodata)

    grid["level"] = 0
    grid["region"] = REGION

    print("Filtered grid cells:", len(grid))

    # -----------------------------------
    # Terrain metrics
    # -----------------------------------
    write_progress(35, "Computing Terrain Metrics...")

    mean_slope = compute_mean_slope_per_cell(grid, slope_raster, transform)
    mean_elevation = compute_mean_elevation_per_cell(grid, dem, transform)

    # Flow accumulation is at full resolution, MUST use the original unscaled transform!
    flow_acc_mean = compute_mean_flow_per_cell(grid, flow_acc, orig_transform)

    mean_curvature = np.zeros_like(mean_slope)

    # -----------------------------------
    # Graph
    # -----------------------------------
    G = build_downhill_graph(grid, mean_elevation)
    
    upstream_map = build_upstream_map(G)
    
    print(f"Saving terrain cache for {REGION}...")
    os.makedirs("outputs", exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump({
            "grid": grid,
            "mean_slope": mean_slope,
            "mean_elevation": mean_elevation,
            "flow_acc_mean": flow_acc_mean,
            "mean_curvature": mean_curvature,
            "G": G
        }, f)

# -----------------------------------
# Rainfall source
# -----------------------------------

if MODE == "historical":

    rainfall_file = cfg["rainfall"]

    rainfall_data, rain_lat, rain_lon = load_chirps_subset(rainfall_file)

    rain_tree = build_rainfall_tree(rain_lat, rain_lon)

    # region center (for live / fallback)
    region_lat = cfg["lat"]
    region_lon = cfg["lon"]

else:

    region_lat = cfg["lat"]
    region_lon = cfg["lon"]

    if MODE == "dynamic":
        rainfall_series = [DYNAMIC_RAIN]
    else:
        rainfall_series = fetch_live_rainfall(region_lat, region_lon)
    
    rain_iterator = rainfall_series

# -----------------------------------
# Initialize state
# -----------------------------------

state = initialize_state(grid)

print("State initialized")


# -----------------------------------
# Rainfall simulation
# -----------------------------------

if MODE == "historical":
    rain_iterator = rainfall_data.time.values[:30]
    
    write_progress(38, "Precomputing Spatial Tree Mapping...")
    rain_idx_map = precompute_rainfall_mapping(grid, rain_tree)
    
elif MODE == "dynamic":
    rain_iterator = [DYNAMIC_RAIN]
else:
    rain_iterator = rainfall_series

print("Total rainfall steps:", len(rain_iterator))

for idx, rain_step in enumerate(tqdm(rain_iterator)):
    write_progress(40 + (idx / max(1, len(rain_iterator))) * 40, f"Simulating Rain Step {idx+1}/{len(rain_iterator)}...")
    print("\nUpdating digital twin")

    if MODE == "historical":

        rainfall_day = rainfall_data.sel(time=rain_step).values

        rainfall_values = map_rainfall_to_grid_fast(
          rainfall_day, rain_idx_map
      )
        

    else:

        rainfall_values = np.full(len(grid), rain_step)

    state = evolve_state(
    state,
    rainfall_values,
    mean_slope,
    flow_acc_mean,
    upstream_map
     )

    print("Mean flood:", np.mean(state["flood_index"]))


# -----------------------------------
# Hazard metrics
# -----------------------------------

state["saturation"] = compute_saturation(
    state["memory"], state["flood_index"]
)

# Normalize curvature first
curvature_factor = normalize(np.abs(mean_curvature))

terrain_factor = normalize(mean_slope) + 2 * curvature_factor

state["landslide_stress"] = compute_landslide_stress(
    state["saturation"] * (1 + terrain_factor),
    mean_slope,
    state["resistance"]
)

state["composite_risk"] = compute_composite_risk(
    state["flood_index"],
    state["landslide_stress"]
)

normalized_risk = normalize(state["composite_risk"])

# In dynamic mode, scale the normalized risk by the rainfall magnitude so that
# 0 mm input produces near-zero probability instead of terrain-shaped output.
# We use a logistic weight: rain_weight approaches 1.0 only at heavy rainfall.
if MODE == "dynamic":
    RAIN_SCALE = 150.0  # mm at which weight ≈ 0.73 (adjustable)
    rain_weight = 1.0 - np.exp(-DYNAMIC_RAIN / RAIN_SCALE)
    rain_weight = float(np.clip(rain_weight, 0.0, 1.0))
    print(f"Dynamic rain weight: {rain_weight:.4f} (rainfall={DYNAMIC_RAIN} mm)")
    normalized_risk = normalized_risk * rain_weight

probability = compute_probability(
    normalized_risk,
    scale=12,
    theta=state["theta"]
)


# -----------------------------------
# Adaptive refinement
# -----------------------------------

grid, state = adaptive_refinement(
    grid,
    probability,
    state,
    max_depth=3
)
# -----------------------------------
# Re-map rainfall for refined grid
# -----------------------------------

if MODE == "historical":

    rainfall_day = rainfall_data.sel(time=rain_step).values
    refined_rain_idx = precompute_rainfall_mapping(grid, rain_tree)
    
    rainfall_values = map_rainfall_to_grid_fast(
        rainfall_day,
        refined_rain_idx
    )

else:

    rainfall_values = np.full(len(grid), rain_step)

# -----------------------------------
# Recompute terrain for refined grid
# -----------------------------------

mean_slope = compute_mean_slope_per_cell(
    grid, slope_raster, transform
)

mean_elevation = compute_mean_elevation_per_cell(
    grid, dem, transform
)

flow_acc_mean = compute_mean_flow_per_cell(
    grid, flow_acc, orig_transform
)

mean_curvature = np.zeros_like(mean_slope)

# rebuild graph
G = build_downhill_graph(grid, mean_elevation)
upstream_map = build_upstream_map(G)

# recompute composite risk
state["composite_risk"] = compute_composite_risk(
    state["flood_index"],
    state["landslide_stress"]
)

normalized_risk = normalize(state["composite_risk"])

# Re-apply the same rain weight after grid refinement.
if MODE == "dynamic":
    RAIN_SCALE = 150.0
    rain_weight = 1.0 - np.exp(-DYNAMIC_RAIN / RAIN_SCALE)
    rain_weight = float(np.clip(rain_weight, 0.0, 1.0))
    normalized_risk = normalized_risk * rain_weight

probability = compute_probability(
    normalized_risk,
    scale=12,
    theta=state["theta"]
)


# -----------------------------------
# Learning update
# -----------------------------------

observed = (normalized_risk > 0.7).astype(int)

state["theta"] = calibration_update(
    state["theta"],
    probability,
    observed
)

print("Updated theta:", state["theta"])


# -----------------------------------
# Attach results
# -----------------------------------

grid["flood"] = state["flood_index"]
grid["landslide"] = state["landslide_stress"]
grid["risk"] = state["composite_risk"]
grid["probability"] = probability
grid["risk_level"] = np.where(probability < 0.25, "low",
                     np.where(probability < 0.5, "medium",
                     np.where(probability < 0.75, "high", "extreme")))


# -----------------------------------
# Scenario simulation
# -----------------------------------

normal_rain = rainfall_values
heavy_rain = rainfall_values * 1.8
extreme_rain = rainfall_values * 2.5

mem_n = update_memory(state["memory"], normal_rain, decay=0.6)
mem_h = update_memory(state["memory"], heavy_rain, decay=0.6)
mem_e = update_memory(state["memory"], extreme_rain, decay=0.6)

flood_n = compute_flood_index(mem_n, mean_slope, upstream_map, state["flood_index"])
flood_h = compute_flood_index(mem_h, mean_slope, upstream_map, state["flood_index"])
flood_e = compute_flood_index(mem_e, mean_slope, upstream_map, state["flood_index"])

risk_n = compute_composite_risk(
    flood_n,
    compute_landslide_stress(
        compute_saturation(mem_n, flood_n),
        mean_slope,
        state["resistance"]
    )
)

risk_h = compute_composite_risk(
    flood_h,
    compute_landslide_stress(
        compute_saturation(mem_h, flood_h),
        mean_slope,
        state["resistance"]
    )
)

risk_e = compute_composite_risk(
    flood_e,
    compute_landslide_stress(
        compute_saturation(mem_e, flood_e),
        mean_slope,
        state["resistance"]
    )
)

grid["normal_prob"] = compute_probability(normalize(risk_n), scale=6, theta=state["theta"])
grid["heavy_prob"] = compute_probability(normalize(risk_h), scale=6, theta=state["theta"])
grid["extreme_prob"] = compute_probability(normalize(risk_e), scale=6, theta=state["theta"])

print("Grid size:", len(grid))
print("Probability range:", np.min(probability), np.max(probability))
print("Flood range:", np.min(state["flood_index"]), np.max(state["flood_index"]))
print("Stress range:", np.min(state["landslide_stress"]), np.max(state["landslide_stress"]))
print("Risk range:", np.min(state["composite_risk"]), np.max(state["composite_risk"]))
# -----------------------------------
# Risk Category Counts (NEW)
# -----------------------------------

def categorize(arr):
    return {
        "low": int(np.sum(arr < 0.25)),
        "medium": int(np.sum((arr >= 0.25) & (arr < 0.5))),
        "high": int(np.sum((arr >= 0.5) & (arr < 0.75))),
        "extreme": int(np.sum(arr >= 0.75))
    }

# Normalize values for comparison
flood_norm = normalize(state["flood_index"])
landslide_norm = normalize(state["landslide_stress"])
prob_norm = probability  # already 0–1

# Compute stats
flood_stats = categorize(flood_norm)
landslide_stats = categorize(landslide_norm)
prob_stats = categorize(prob_norm)

# Print nicely
def print_stats(name, stats, total):
    print(f"\n===== {name} =====")
    for k, v in stats.items():
        print(f"{k}: {v} ({(v/total)*100:.2f}%)")

total = len(grid)

print_stats("FLOOD RISK", flood_stats, total)
print_stats("LANDSLIDE RISK", landslide_stats, total)
print_stats("OVERALL PROBABILITY", prob_stats, total)

# -----------------------------------
# Save GeoJSON
# -----------------------------------

os.makedirs("outputs", exist_ok=True)

#output_file = "outputs/hazard_map.geojson"
output_file = f"outputs/{REGION}_{MODE}.geojson"

grid_wgs84 = grid.to_crs(epsg=4326)
grid_wgs84.to_file(output_file, driver="GeoJSON")

summary_output = f"outputs/{REGION}_{MODE}_summary.geojson"
stats_output = f"outputs/{REGION}_{MODE}_stats.json"
summary_size = min(len(grid_wgs84), 1800)
summary_sample = grid_wgs84.sample(n=summary_size, random_state=42) if summary_size > 0 else grid_wgs84.head(0)
summary_sample.to_file(summary_output, driver="GeoJSON")

import json
with open(stats_output, "w", encoding="utf-8") as f:
    json.dump({
        "gridSize": int(total),
        "probRange": {
            "min": float(np.min(probability)),
            "max": float(np.max(probability))
        },
        "floodRange": {
            "min": float(np.min(state["flood_index"])),
            "max": float(np.max(state["flood_index"]))
        },
        "stressRange": {
            "min": float(np.min(state["landslide_stress"])),
            "max": float(np.max(state["landslide_stress"]))
        },
        "riskRange": {
            "min": float(np.min(state["composite_risk"])),
            "max": float(np.max(state["composite_risk"]))
        },
        "floodRisk": flood_stats,
        "landslideRisk": landslide_stats,
        "overallProbability": prob_stats
    }, f)

print("Saved:", output_file)
print("Saved summary:", summary_output)
print("Saved stats:", stats_output)


def grid_to_raster(grid, column, transform, shape):

    shapes = ((geom, value) for geom, value in zip(grid.geometry, grid[column]))

    raster = rasterize(
        shapes,
        out_shape=shape,
        transform=transform,
        fill=np.nan,
        dtype="float32"
    )

    return raster

def downsample(arr, factor=6):
    return arr[::factor, ::factor]


if SKIP_PLOTS:
    write_progress(100, "Simulation Complete (plots skipped for fast mode)")
    print("Skipping plot rendering for fast mode")
    print("Digital Twin run complete")
    raise SystemExit(0)


# -----------------------------------
# 1️⃣5️⃣ Visualization
# -----------------------------------
write_progress(85, "Rendering Hazard Maps...")

# Convert grid results to raster
prob_raster = downsample(
    grid_to_raster(grid, "probability", transform, dem.shape),
    6
)

flood_raster = downsample(grid_to_raster(grid,"flood",transform,dem.shape),6)
landslide_raster = downsample(grid_to_raster(grid,"landslide",transform,dem.shape),6)
risk_raster = downsample(grid_to_raster(grid,"risk",transform,dem.shape),6)

normal_raster = downsample(grid_to_raster(grid,"normal_prob",transform,dem.shape),6)
heavy_raster = downsample(grid_to_raster(grid,"heavy_prob",transform,dem.shape),6)
extreme_raster = downsample(grid_to_raster(grid,"extreme_prob",transform,dem.shape),6)



# -----------------------------------
# Probability map + histogram
# -----------------------------------

fig, axes = plt.subplots(1,2, figsize=(14,6))

pmin = np.nanpercentile(prob_raster, 5)
pmax = np.nanpercentile(prob_raster, 95)
im = axes[0].imshow(prob_raster, cmap="Reds", vmin=pmin, vmax=pmax)
axes[0].set_title("Hazard Probability Map")
axes[0].axis("off")
plt.colorbar(im, ax=axes[0])

axes[1].hist(probability, bins=40)
axes[1].set_title("Probability Distribution")

plt.tight_layout()
plt.savefig(f"outputs/{REGION}_{MODE}_prob_hist.png")
plt.close()


# -----------------------------------
# Multi-hazard maps
# -----------------------------------

fig, axes = plt.subplots(2,2, figsize=(16,12))

fmin = np.nanpercentile(flood_raster, 5)
fmax = np.nanpercentile(flood_raster, 95)

im1 = axes[0,0].imshow(flood_raster, cmap="Blues", vmin=fmin, vmax=fmax)
axes[0,0].set_title("Flood Index Map")
axes[0,0].axis("off")
plt.colorbar(im1, ax=axes[0,0])

lmin = np.nanpercentile(landslide_raster, 5)
lmax = np.nanpercentile(landslide_raster, 95)

im2 = axes[0,1].imshow(
    landslide_raster,
    cmap="Oranges",
    vmin=lmin,
    vmax=lmax
)
axes[0,1].set_title("Landslide Stress Map")
axes[0,1].axis("off")
plt.colorbar(im2, ax=axes[0,1])

rmin = np.nanpercentile(risk_raster, 5)
rmax = np.nanpercentile(risk_raster, 95)

im3 = axes[1,0].imshow(
    risk_raster,
    cmap="Reds",
    vmin=rmin,
    vmax=rmax
)
axes[1,0].set_title("Composite Risk Map")
axes[1,0].axis("off")
plt.colorbar(im3, ax=axes[1,0])


pmin = np.nanpercentile(prob_raster, 5)
pmax = np.nanpercentile(prob_raster, 95)

im4 = axes[1,1].imshow(
    prob_raster,
    cmap="Purples",
    vmin=pmin,
    vmax=pmax
)

axes[1,1].set_title("Hazard Probability Map")
axes[1,1].axis("off")
plt.colorbar(im4, ax=axes[1,1])

plt.tight_layout()
plt.savefig(f"outputs/{REGION}_{MODE}_multi_hazard.png")
plt.close()


# -----------------------------------
# Rainfall scenario comparison
# -----------------------------------

fig, axes = plt.subplots(1,3, figsize=(18,6))

im1 = axes[0].imshow(normal_raster, cmap="Purples")
axes[0].set_title("Normal Rain")
axes[0].axis("off")
plt.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(heavy_raster, cmap="Purples")
axes[1].set_title("Heavy Rain")
axes[1].axis("off")
plt.colorbar(im2, ax=axes[1])

im3 = axes[2].imshow(extreme_raster, cmap="Purples")
axes[2].set_title("Extreme Rain")
axes[2].axis("off")
plt.colorbar(im3, ax=axes[2])

plt.tight_layout()
plt.savefig(f"outputs/{REGION}_{MODE}_rainfall_scenario.png")
plt.close()

write_progress(100, "Simulation Complete")
print("Digital Twin run complete")