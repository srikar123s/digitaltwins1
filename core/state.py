import numpy as np

def initialize_state(grid):
    n = len(grid)

    state = {
        "memory": np.zeros(n),
        "flood_index": np.zeros(n),
        "saturation": np.zeros(n),
        "resistance": np.ones(n) * 1.0,
        "landslide_stress": np.zeros(n),
        "composite_risk": np.zeros(n),
        "theta": 0.1  # learnable bias
    }

    return state