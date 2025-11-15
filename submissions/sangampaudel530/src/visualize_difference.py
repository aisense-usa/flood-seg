import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

CUTOUT_FOLDER = "sample/cutouts"

# List all cutouts
tiles = sorted([f for f in os.listdir(CUTOUT_FOLDER) if f.endswith("_pre.png")])

for tile_pre in tiles:
    idx = tile_pre.split("_")[1]  # get tile number
    tile_post = f"tile_{idx}_post.png"

    pre_path = os.path.join(CUTOUT_FOLDER, tile_pre)
    post_path = os.path.join(CUTOUT_FOLDER, tile_post)

    pre_img = np.array(Image.open(pre_path))
    post_img = np.array(Image.open(post_path))

    # Difference mask (simple absolute difference)
    diff = np.abs(pre_img.astype(int) - post_img.astype(int))
    diff_mask = np.max(diff, axis=-1)  # max difference across channels

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(pre_img)
    axes[0].set_title("Pre-Flood")
    axes[0].axis("off")

    axes[1].imshow(post_img)
    axes[1].set_title("Post-Flood")
    axes[1].axis("off")

    axes[2].imshow(diff_mask, cmap="hot")
    axes[2].set_title("Difference")
    axes[2].axis("off")

    plt.suptitle(f"Tile {idx}")
    plt.show()

    # Optional: show only first few tiles
    if int(idx) >= 4:
        break
