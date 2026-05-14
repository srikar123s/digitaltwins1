import rasterio
from rasterio.enums import Resampling

input_file = "data/dem/uk_utm.tif"
output_file = "data/dem/uk_utm_small.tif"

scale_factor = 6   # you can change to 6 or 8 if needed

with rasterio.open(input_file) as src:

    print("Original size:", src.width, src.height)

    data = src.read(
        1,
        out_shape=(
            src.height // scale_factor,
            src.width // scale_factor
        ),
        resampling=Resampling.average
    )

    # Adjust transform
    transform = src.transform * src.transform.scale(
        (src.width / data.shape[-1]),
        (src.height / data.shape[-2])
    )

    profile = src.profile
    profile.update({
        "height": data.shape[0],
        "width": data.shape[1],
        "transform": transform
    })

    with rasterio.open(output_file, "w", **profile) as dst:
        dst.write(data, 1)

print("Small UTM created:", output_file)