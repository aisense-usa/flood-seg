import os
import torch
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio.features import shapes
from shapely.geometry import shape
import cv2
from torchvision import transforms
from torch import nn
import torch.nn.functional as F

PRE_FLOOD_PATH = "./data/Preflood.tif"
POST_FLOOD_PATH = "./data/Preflood.tif"
OUTPUT_DIR = "./submissions/Tulsee/"
CUTOUT_DIR = os.path.join(OUTPUT_DIR, "cutouts")
CSV_PATH = os.path.join(OUTPUT_DIR, "affected.csv")

TILE_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(CUTOUT_DIR, exist_ok=True)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(128, 256)
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv1 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv2 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        bn = self.bottleneck(p2)
        # ---- Upsample path ----
        up1 = self.up1(bn)
        # align d2 size with up1
        d2 = F.interpolate(d2, size=up1.shape[2:], mode="bilinear", align_corners=True)
        merge1 = torch.cat([up1, d2], dim=1)
        c1 = self.conv1(merge1)

        up2 = self.up2(c1)
        # align d1 size with up2
        d1 = F.interpolate(d1, size=up2.shape[2:], mode="bilinear", align_corners=True)
        merge2 = torch.cat([up2, d1], dim=1)
        c2 = self.conv2(merge2)

        return torch.sigmoid(self.outc(c2))


# load model (pretend pretrained)
model = UNet().to(DEVICE)
model.eval()

transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

# Flood Detection


def predict_mask(model, title):
    tile_tensor = transform(title).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = model(tile_tensor)
    mask = (pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
    return mask


def detect_flood(pre_path, post_path):
    results = []
    with rasterio.open(pre_path) as pre, rasterio.open(post_path) as post:
        transform = post.transform
        pixel_size = transform.a
        width, height = post.width, post.height

        print(f"Processing {width} x {height} ...")

        tile_id = 0
        for y in range(0, height, TILE_SIZE):
            for x in range(0, width, TILE_SIZE):
                window = Window(x, y, TILE_SIZE, TILE_SIZE)
                pre_tile = pre.read(window=window)
                post_tile = post.read(window=window)

                pre_rgb = np.moveaxis(pre_tile[:3], 0, -1)
                post_rgb = np.moveaxis(post_tile[:3], 0, -1)

                # Predict masks
                pre_mask = predict_mask(model, pre_rgb)
                post_mask = predict_mask(model, post_rgb)

                flood_mask = ((post_mask == 1) & (pre_mask == 0)).astype(np.uint8)

                if np.sum(flood_mask) == 0:
                    continue

                pre_path = os.path.join(CUTOUT_DIR, f"tile_{tile_id:03d}_pre.png")
                post_path = os.path.join(CUTOUT_DIR, f"tile_{tile_id:03d}_post.png")

                cv2.imwrite(pre_path, cv2.cvtColor(pre_rgb, cv2.COLOR_RGB2BGR))
                cv2.imwrite(post_path, cv2.cvtColor(post_rgb, cv2.COLOR_RGB2BGR))

                for geom, val in shapes(
                    flood_mask, transform=rasterio.windows.transform(window, transform)
                ):
                    if val == 1:
                        polygon = shape(geom)
                        centroid = polygon.centroid
                        area_m2 = polygon.area
                        results.append(
                            {
                                "tile_id": f"tile_{tile_id:03d}",
                                "center_longitude": centroid.x,
                                "center_latitude": centroid.y,
                                "area_m2": area_m2,
                                "area_lost_m2": area_m2,
                                "pre_flood_land_image": pre_path,
                                "post_flood_land_image": post_path,
                            }
                        )

                tile_id += 1
                print(f"Processed tile {tile_id}")

    return results


# Main
if __name__ == "__main__":
    print("Starting deep-learning flood detection...")
    results = detect_flood(PRE_FLOOD_PATH, POST_FLOOD_PATH)
    if results:
        df = pd.DataFrame(results)
        df.to_csv(CSV_PATH, index=False)
        print(f"\n✅ Results saved to: {CSV_PATH}")
    else:
        print("⚠️ No changes detected.")
