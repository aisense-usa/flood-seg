import os
import rasterio
from rasterio.windows import from_bounds, Window
import math

def align_and_crop_to_overlap(pre_path, post_path, out_pre, out_post, block_size=1024):
    # Ensure output directories exist
    os.makedirs(os.path.dirname(out_pre), exist_ok=True)
    os.makedirs(os.path.dirname(out_post), exist_ok=True)

    with rasterio.open(pre_path) as pre, rasterio.open(post_path) as post:
        # Step 1: CRS check
        if pre.crs != post.crs:
            raise ValueError("❌ CRS mismatch! Please reproject first.")

        # Step 2: Compute overlap
        left = max(pre.bounds.left, post.bounds.left)
        right = min(pre.bounds.right, post.bounds.right)
        bottom = max(pre.bounds.bottom, post.bounds.bottom)
        top = min(pre.bounds.top, post.bounds.top)
        print(f"✅ Overlap bounds: {left:.2f}, {bottom:.2f}, {right:.2f}, {top:.2f}")

        # Step 3: Create read windows
        win_pre = from_bounds(left, bottom, right, top, pre.transform)
        win_post = from_bounds(left, bottom, right, top, post.transform)

        # Step 4: Output transform and shape
        transform = pre.window_transform(win_pre)
        height = int(win_pre.height)
        width = int(win_pre.width)
        print(f"✅ Overlap size: {width} x {height}")

        # Step 5: Update output profile
        profile = pre.profile
        profile.update({
            'height': height,
            'width': width,
            'transform': transform,
            'BIGTIFF': 'YES'  # ✅ Enable BigTIFF output
        })

        # Step 6: Write tile-by-tile
        with rasterio.open(out_pre, 'w', **profile) as dst_pre, \
             rasterio.open(out_post, 'w', **profile) as dst_post:

            num_tiles_x = math.ceil(width / block_size)
            num_tiles_y = math.ceil(height / block_size)
            print(f"🧩 Processing in {num_tiles_x} x {num_tiles_y} tiles...")

            for ty in range(num_tiles_y):
                for tx in range(num_tiles_x):
                    x_off = int(tx * block_size)
                    y_off = int(ty * block_size)
                    w = min(block_size, width - x_off)
                    h = min(block_size, height - y_off)

                    window_pre = Window(win_pre.col_off + x_off, win_pre.row_off + y_off, w, h)
                    window_post = Window(win_post.col_off + x_off, win_post.row_off + y_off, w, h)

                    pre_block = pre.read(window=window_pre)
                    post_block = post.read(window=window_post)

                    dst_pre.write(pre_block, window=Window(x_off, y_off, w, h))
                    dst_post.write(post_block, window=Window(x_off, y_off, w, h))

            print(f"✅ Finished writing aligned cropped TIFFs:\n - {out_pre}\n - {out_post}")
