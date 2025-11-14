import os
import torch
import numpy as np
import rasterio
from rasterio.windows import Window
from skimage.measure import label, regionprops
import cv2
import pandas as pd
from tqdm import tqdm
from model import SiameseUNetPP  


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
def load_model(model_class, checkpoint, device):
    model = model_class(n_channels=3, n_classes=1, base_ch=32)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------
# MODEL PREDICT (single tile)
# ---------------------------------------------------------
def model_predict(model, pre_tile, post_tile, device):

    pre_tile = pre_tile[:, :, :3]
    post_tile = post_tile[:, :, :3]

    pre = torch.tensor(pre_tile / 255., dtype=torch.float32).permute(2,0,1).unsqueeze(0).to(device)
    post = torch.tensor(post_tile / 255., dtype=torch.float32).permute(2,0,1).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(pre, post)
        pred = torch.sigmoid(out).squeeze().cpu().numpy()

    del pre, post, out
    torch.cuda.empty_cache()
    return pred


# ---------------------------------------------------------
# EXTRACT REGION CENTROIDS + AREA
# ---------------------------------------------------------
def extract_regions(mask, full_transform, pixel_size):

    labelled = label(mask)
    props = regionprops(labelled)

    results = []

    for region in props:
        if region.area < 20:
            continue

        cy, cx = region.centroid
        lon, lat = rasterio.transform.xy(full_transform, cy, cx)
        area_m2 = region.area * (pixel_size ** 2)

        results.append({
            "centroid_lon": lon,
            "centroid_lat": lat,
            "area_m2": area_m2,
        })

    return results


# ---------------------------------------------------------
# SAFE IMAGE WRITER (JPG ONLY — PREVENTS libpng ERRORS)
# ---------------------------------------------------------
def safe_write_image(path, image):

    # force JPG only
    path = path.replace(".png", ".jpg")

    # ensure valid uint8
    if image.dtype != np.uint8:
        image = np.nan_to_num(image)
        image = np.clip(image, 0, 255).astype(np.uint8)

    # skip too small tiles
    if image.shape[0] < 4 or image.shape[1] < 4:
        return False

    try:
        cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return True
    except:
        return False



# ---------------------------------------------------------
# MAIN INFERENCE
# ---------------------------------------------------------
def run_inference(
    model,
    pre_path,
    post_path,
    output_dir,
    tile_size=512,
    threshold=0.5,
):

    os.makedirs(output_dir, exist_ok=True)
    cutout_dir = os.path.join(output_dir, "cutouts")
    os.makedirs(cutout_dir, exist_ok=True)

    device = next(model.parameters()).device
    pre = rasterio.open(pre_path)
    post = rasterio.open(post_path)

    assert pre.shape == post.shape, "ERROR: Pre/Post must be aligned!"

    H, W = pre.height, pre.width
    base_transform = pre.transform
    pixel_size = pre.res[0]

    rows = (H + tile_size - 1) // tile_size
    cols = (W + tile_size - 1) // tile_size

    results = []
    tile_id_counter = 0

    for ty in tqdm(range(rows), desc="Rows"):
        for tx in range(cols):

            x_off = tx * tile_size
            y_off = ty * tile_size

            w = min(tile_size, W - x_off)
            h = min(tile_size, H - y_off)

            window = Window(x_off, y_off, w, h)

            # Read only required area
            pre_arr = pre.read(window=window)
            post_arr = post.read(window=window)

            pre_tile = np.moveaxis(pre_arr, 0, 2).astype(np.uint8)
            post_tile = np.moveaxis(post_arr, 0, 2).astype(np.uint8)

            del pre_arr, post_arr

            # Predict mask
            pred = model_predict(model, pre_tile, post_tile, device)
            mask = (pred > threshold).astype(np.uint8)

            tile_transform = base_transform * rasterio.Affine.translation(x_off, y_off)
            regions = extract_regions(mask, tile_transform, pixel_size)

            # Skip saving tile images if NO regions detected
            if len(regions) == 0:
                continue

            tile_id = f"tile_{tile_id_counter:05d}"
            tile_id_counter += 1

            pre_file = os.path.join(cutout_dir, f"{tile_id}_pre.jpg")
            post_file = os.path.join(cutout_dir, f"{tile_id}_post.jpg")

            # Make BGR for OpenCV
            pre_img = cv2.cvtColor(pre_tile[:,:,:3], cv2.COLOR_RGB2BGR)
            post_img = cv2.cvtColor(post_tile[:,:,:3], cv2.COLOR_RGB2BGR)

            safe_write_image(pre_file, pre_img)
            safe_write_image(post_file, post_img)

            for region in regions:
                results.append({
                    "tile_id": tile_id,
                    "center_longitude": region["centroid_lon"],
                    "center_latitude": region["centroid_lat"],
                    "area_m2": region["area_m2"],
                    "area_lost_m2": region["area_m2"],
                    "pre_flood_land_image": pre_file,
                    "post_flood_land_image": post_file,
                })

            del pre_tile, post_tile, mask, pred, regions
            torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "affected.csv"), index=False)

    print("\n✅ Inference Complete!")
    print("Output directory:", output_dir)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(
        SiameseUNetPP,
        "best_siamese_unetpp.pth",
        device
    )

    run_inference(
        model=model,
        pre_path="/kaggle/working/aligned/pre_aligned.tif",
        post_path="/kaggle/working/aligned/post_aligned.tif",
        output_dir="submissions/Dharay/",
        tile_size=512,
        threshold=0.45
    )
