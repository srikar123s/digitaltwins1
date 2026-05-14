import numpy as np

def update_memory(prev_memory, rainfall, decay=0.85):
    """
    Memory(t) = decay * Memory(t-1) + Rain(t)
    """
    return decay * prev_memory + rainfall
