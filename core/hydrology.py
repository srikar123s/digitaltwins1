from pysheds.grid import Grid
import numpy as np

def compute_flow_accumulation(dem_path):

    grid = Grid.from_raster(dem_path)
    dem = grid.read_raster(dem_path)

    # Fill pits
    flooded = grid.fill_pits(dem)

    # Fill depressions
    flooded = grid.fill_depressions(flooded)

    # Resolve flats
    inflated = grid.resolve_flats(flooded)

    # Flow direction
    flow_dir = grid.flowdir(inflated)

    # Flow accumulation
    acc = grid.accumulation(flow_dir)

    return acc