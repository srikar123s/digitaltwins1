import os

with open('main.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

out = []
in_block = False
funcs_to_move = []
block_lines = []

start_marker = 'if not os.path.exists(utm_dem):'
end_marker = 'G = build_downhill_graph(grid, mean_elevation)'

idx = 0
while idx < len(lines):
    line = lines[idx]
    
    if line.startswith(start_marker):
        in_block = True
    
    if not in_block:
        out.append(line)
        idx += 1
        continue
    
    # We are inside the block
    if line.startswith('def '):
        func_lines = [line]
        idx += 1
        while idx < len(lines) and (lines[idx].startswith('    ') or lines[idx].strip() == ''):
            func_lines.append(lines[idx])
            idx += 1
        funcs_to_move.extend(func_lines)
        continue
        
    block_lines.append(line)
    
    if line.startswith(end_marker):
        in_block = False
        out.extend(funcs_to_move)
        
        out.append('import pickle\n')
        out.append('cache_path = f"outputs/{REGION}_terrain_cache.pkl"\n')
        out.append('is_cached = os.path.exists(cache_path)\n\n')
        out.append('if is_cached:\n')
        out.append('    print(f"Loading terrain cache for {REGION}...")\n')
        out.append('    write_progress(15, "Loading Terrain Cache...")\n')
        out.append('    with open(cache_path, "rb") as f:\n')
        out.append('        cache = pickle.load(f)\n')
        out.append('    grid = cache["grid"]\n')
        out.append('    mean_slope = cache["mean_slope"]\n')
        out.append('    mean_elevation = cache["mean_elevation"]\n')
        out.append('    flow_acc_mean = cache["flow_acc_mean"]\n')
        out.append('    mean_curvature = cache["mean_curvature"]\n')
        out.append('    G = cache["G"]\n')
        out.append('    transform = cache["transform"]\n')
        out.append('    crs = cache["crs"]\n')
        
        out.append('else:\n')
        
        for b_line in block_lines:
            if b_line.strip() == '':
                out.append(b_line)
            else:
                out.append('    ' + b_line)
                
        out.append('\n    print(f"Saving terrain cache for {REGION}...")\n')
        out.append('    os.makedirs("outputs", exist_ok=True)\n')
        out.append('    with open(cache_path, "wb") as f:\n')
        out.append('        pickle.dump({\n')
        out.append('            "grid": grid,\n')
        out.append('            "mean_slope": mean_slope,\n')
        out.append('            "mean_elevation": mean_elevation,\n')
        out.append('            "flow_acc_mean": flow_acc_mean,\n')
        out.append('            "mean_curvature": mean_curvature,\n')
        out.append('            "G": G,\n')
        out.append('            "transform": transform,\n')
        out.append('            "crs": crs\n')
        out.append('        }, f)\n')
        
    idx += 1

with open('main.py', 'w', encoding='utf-8') as f:
    f.writelines(out)
