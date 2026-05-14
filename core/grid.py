import geopandas as gpd
from shapely.geometry import box
import numpy as np

def generate_base_grid(bounds, resolution_m=3000):
    minx, miny, maxx, maxy = bounds

    grid_cells = []

    y = miny
    while y < maxy:
        x = minx
        while x < maxx:
            grid_cells.append(
                box(x, y, x + resolution_m, y + resolution_m)
            )
            x += resolution_m
        y += resolution_m

    grid = gpd.GeoDataFrame(geometry=grid_cells)
    grid["cell_id"] = range(len(grid))

    return grid
