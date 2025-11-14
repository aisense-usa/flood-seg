from PIL import Image
import numpy as np
import os
from tqdm import tqdm

tile_folder = "flood_data_zip"
output_folder = "Filtered_Flood_data"
os.makedirs(output_folder, exist_ok=True)

threshold_white = 0.98  # 98% of pixels are white → discard

for file in tqdm(os.listdir(tile_folder)):
    if not file.endswith(".jpg"):
        continue
    path = os.path.join(tile_folder, file)
    img = Image.open(path).convert("RGB")
    arr = np.array(img)

    # Normalize and detect white pixels
    white_fraction = np.mean(np.all(arr > 240, axis=-1))

    if white_fraction < threshold_white:
        img.save(os.path.join(output_folder, file))
