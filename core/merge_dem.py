import os
import rasterio
from rasterio.merge import merge

def merge_dem_tiles(input_folder, output_path):
    dem_files = []

    # Collect all .tif files
    for file in os.listdir(input_folder):
        if file.endswith(".tif"):
            dem_files.append(os.path.join(input_folder, file))

    src_files = []
    for file in dem_files:
        src = rasterio.open(file)
        src_files.append(src)

    mosaic, out_transform = merge(src_files)

    out_meta = src.meta.copy()
    out_meta.update({
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_transform
    })

    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(mosaic)

    for src in src_files:
        src.close()

    print("Merged DEM saved at:", output_path)
