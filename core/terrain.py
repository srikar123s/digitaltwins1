import numpy as np

def compute_slope(dem, transform):
    """
    Compute slope from DEM using central difference.
    """
    x_res = transform.a
    y_res = -transform.e

    dzdx = np.gradient(dem, axis=1) / x_res
    dzdy = np.gradient(dem, axis=0) / y_res

    slope = np.sqrt(dzdx**2 + dzdy**2)
    slope = np.degrees(np.arctan(slope))

    return slope


from tqdm import tqdm

def compute_mean_slope_per_cell(grid, slope_raster, transform):
    """
    Compute mean slope for each grid cell.
    """

    inv_transform = ~transform
    
    centroids = grid.geometry.centroid
    cols, rows = inv_transform * (centroids.x.values, centroids.y.values)
    
    rows = np.clip(np.round(rows).astype(int), 0, slope_raster.shape[0] - 1)
    cols = np.clip(np.round(cols).astype(int), 0, slope_raster.shape[1] - 1)
    
    return slope_raster[rows, cols]


def compute_mean_elevation_per_cell(grid, dem, transform):
    from tqdm import tqdm
    inv_transform = ~transform
    
    centroids = grid.geometry.centroid
    cols, rows = inv_transform * (centroids.x.values, centroids.y.values)
    
    rows = np.clip(np.round(rows).astype(int), 0, dem.shape[0] - 1)
    cols = np.clip(np.round(cols).astype(int), 0, dem.shape[1] - 1)
    
    return dem[rows, cols]
