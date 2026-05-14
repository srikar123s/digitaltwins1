import xarray as xr
import numpy as np
from scipy.spatial import cKDTree
import pyproj


def load_chirps_subset(file_path):
    """
    Load CHIRPS rainfall subset for Kerala and August 2018.
    """

    ds = xr.open_dataset(file_path)

    rain = ds.sel(
        time=slice("2018-08-01", "2018-08-31"),
        latitude=slice(8.0, 14.5),
        longitude=slice(74.5, 77.7)
    )

    rainfall = rain["precip"]
    lat = rain.latitude.values
    lon = rain.longitude.values

    return rainfall, lat, lon


def build_rainfall_tree(lat, lon):

    lon_grid, lat_grid = np.meshgrid(lon, lat)

    points = np.column_stack((lat_grid.ravel(), lon_grid.ravel()))

    tree = cKDTree(points)

    return tree


def precompute_rainfall_mapping(grid, tree):
    import pyproj
    transformer = pyproj.Transformer.from_crs(
        grid.crs, "EPSG:4326", always_xy=True
    )

    x = grid.geometry.centroid.x.values
    y = grid.geometry.centroid.y.values

    # Perform vectorized CRS math
    lon_c, lat_c = transformer.transform(x, y)

    # cKDTree expects structured [lat, lon] array
    centroids = np.column_stack((lat_c, lon_c))

    _, idx = tree.query(centroids)
    return idx

def map_rainfall_to_grid_fast(rainfall_day, idx):
    
    rainfall_values = rainfall_day.ravel()[idx]
    rainfall_values = np.nan_to_num(rainfall_values, nan=0.0)

    return rainfall_values