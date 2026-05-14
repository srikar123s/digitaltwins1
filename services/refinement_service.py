from shapely.geometry import box
import geopandas as gpd
import numpy as np

def refine_cells(grid, state, cells_to_refine, refinement_factor=2):

    refine_set = set(cells_to_refine)

    new_cells = []
    new_levels = []

    new_memory = []
    new_flood = []
    new_saturation = []
    new_resistance = []
    new_landslide = []
    new_composite = []

    for pos, row in enumerate(grid.itertuples()):

        current_level = row.level

        if pos in refine_set:

            minx, miny, maxx, maxy = row.geometry.bounds
            dx = (maxx - minx) / refinement_factor
            dy = (maxy - miny) / refinement_factor

            memory_child = state["memory"][pos] / (refinement_factor**2)
            flood_child = state["flood_index"][pos] / (refinement_factor**2)
            sat_child = state["saturation"][pos] / (refinement_factor**2)
            land_child = state["landslide_stress"][pos] / (refinement_factor**2)
            comp_child = state["composite_risk"][pos] / (refinement_factor**2)

            for i in range(refinement_factor):
                for j in range(refinement_factor):

                    new_cells.append(
                        box(
                            minx + i*dx,
                            miny + j*dy,
                            minx + (i+1)*dx,
                            miny + (j+1)*dy
                        )
                    )

                    new_levels.append(current_level + 1)

                    new_memory.append(memory_child)
                    new_flood.append(flood_child)
                    new_saturation.append(sat_child)
                    new_resistance.append(state["resistance"][pos])
                    new_landslide.append(land_child)
                    new_composite.append(comp_child)

        else:
            new_cells.append(row.geometry)
            new_levels.append(current_level)

            new_memory.append(state["memory"][pos])
            new_flood.append(state["flood_index"][pos])
            new_saturation.append(state["saturation"][pos])
            new_resistance.append(state["resistance"][pos])
            new_landslide.append(state["landslide_stress"][pos])
            new_composite.append(state["composite_risk"][pos])

    new_grid = gpd.GeoDataFrame(
        {"geometry": new_cells, "level": new_levels},
        crs=grid.crs
    )

    new_grid["cell_id"] = range(len(new_grid))

    new_state = {
        "memory": np.array(new_memory),
        "flood_index": np.array(new_flood),
        "saturation": np.array(new_saturation),
        "resistance": np.array(new_resistance),
        "landslide_stress": np.array(new_landslide),
        "composite_risk": np.array(new_composite),
        "theta": state["theta"]
    }

    return new_grid, new_state