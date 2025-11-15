import os
import numpy as np
import pandas as pd
from PIL import Image

# --- CONFIG ---
PRE_FOLDER = "sample/pre"
POST_FOLDER = "sample/post"
CUTOUTS_FOLDER = "sample/cutouts"
CSV_PATH = "sample/affected.csv"
THRESHOLD_DIFF = 30   # pixel change threshold for "flood"
# ----------------

os.makedirs(CUTOUTS_FOLDER, exist_ok=True)

# Get sorted tile lists
pre_tiles = sorted(os.listdir(PRE_FOLDER))
post_tiles = sorted(os.listdir(POST_FOLDER))

results = []

for idx, (pre_name, post_name) in enumerate(zip(pre_tiles, post_tiles)):
    pre_path = os.path.join(PRE_FOLDER, pre_name)
    post_path = os.path.join(POST_FOLDER, post_name)

    pre_img = np.array(Image.open(pre_path))
    post_img = np.array(Image.open(post_path))

    # Simple difference to detect change
    diff = np.abs(post_img.astype(int) - pre_img.astype(int))
    flood_mask = np.any(diff > THRESHOLD_DIFF, axis=-1).astype(np.uint8)

    flood_pixels = np.sum(flood_mask)
    if flood_pixels == 0:
        continue  # skip tiles with no change

    # Save cutout images
    cutout_pre = os.path.join(CUTOUTS_FOLDER, f"tile_{idx}_pre.png")
    cutout_post = os.path.join(CUTOUTS_FOLDER, f"tile_{idx}_post.png")
    Image.fromarray(pre_img).save(cutout_pre)
    Image.fromarray(post_img).save(cutout_post)

    # Approximate area (assuming 1 pixel = 1 m² for demo)
    area_m2 = flood_pixels

    results.append({
        "tile_id": f"tile_{idx}",
        "center_x": pre_img.shape[1]//2,
        "center_y": pre_img.shape[0]//2,
        "area_m2": area_m2,
        "pre_flood_tile": cutout_pre,
        "post_flood_tile": cutout_post
    })

# Save CSV
df = pd.DataFrame(results)
df.to_csv(CSV_PATH, index=False)
print(f"\n✅ Done! CSV saved at {CSV_PATH}")
print(f"Cutouts saved in {CUTOUTS_FOLDER}")
