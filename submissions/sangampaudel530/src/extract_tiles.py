import rasterio
from rasterio.windows import Window
from PIL import Image
import numpy as np
import os

# ---------- CONFIG ----------
PRE_TIFF = "data/Preflood.tif"
POST_TIFF = "data/Postflood.tif"
OUTPUT_DIR = "sample"
TILE_SIZE = 512  # or 256
NUM_TILES = 5    # number of tiles to extract
WHITE_THRESHOLD = 0.95  # ignore mostly white tiles
# ----------------------------

# Create folders
PRE_FOLDER = os.path.join(OUTPUT_DIR, "pre")
POST_FOLDER = os.path.join(OUTPUT_DIR, "post")
os.makedirs(PRE_FOLDER, exist_ok=True)
os.makedirs(POST_FOLDER, exist_ok=True)

def extract_tiles(tiff_path, out_folder, num_tiles=NUM_TILES):
    with rasterio.open(tiff_path) as src:
        width, height = src.width, src.height
        tiles_extracted = 0

        # Sample random positions
        np.random.seed(42)
        xs = np.random.randint(0, width-TILE_SIZE, size=num_tiles*5)
        ys = np.random.randint(0, height-TILE_SIZE, size=num_tiles*5)

        for x, y in zip(xs, ys):
            if tiles_extracted >= num_tiles:
                break
            window = Window(x, y, TILE_SIZE, TILE_SIZE)
            img = src.read([1,2,3], window=window)  # RGB
            img = np.moveaxis(img, 0, -1)

            # Check for mostly white tile
            white_frac = np.mean(np.all(img > 240, axis=-1))
            if white_frac > WHITE_THRESHOLD:
                continue

            tile_path = os.path.join(out_folder, f"tile_{tiles_extracted}.png")
            Image.fromarray(img).save(tile_path)
            tiles_extracted += 1
            print(f"Saved tile: {tile_path}")

# Extract demo tiles
print("Extracting pre-flood tiles...")
extract_tiles(PRE_TIFF, PRE_FOLDER)

print("\nExtracting post-flood tiles...")
extract_tiles(POST_TIFF, POST_FOLDER)

print("\n✅ Done! Sample tiles ready in 'sample/pre/' and 'sample/post/'")
