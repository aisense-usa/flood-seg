import rasterio
from rasterio.windows import Window
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

tif_path = "/content/Hanumannagar_Postflood_Orthomosaic.tif"
output_folder = "/content/flood_data_jpg"
os.makedirs(output_folder, exist_ok=True)

tile_size = 1024

with rasterio.open(tif_path) as src:
    width = src.width
    height = src.height
    print(f"TIFF size: {width} x {height}, Bands: {src.count}")

    for i in tqdm(range(0, height, tile_size), desc="Rows"):
        for j in range(0, width, tile_size):
            window = Window(j, i, tile_size, tile_size)
            # Read first 3 bands (RGB)
            img = src.read([1,2,3], window=window)
            img = np.moveaxis(img, 0, -1)
            img = np.clip(img, 0, 255).astype(np.uint8)

            # Save tile
            tile_name = os.path.join(output_folder, f"tile_{i}_{j}.jpg")
            Image.fromarray(img).save(tile_name)

