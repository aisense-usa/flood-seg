import cv2
import torch
import rasterio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset

from utility import split_flood_dataset

class FloodDataset(Dataset):
    def __init__(self, pre_path, post_path, tile_size=512, augment=None, use_ndwi=True):
        self.pre_path = pre_path
        self.post_path = post_path
        self.tile_size = tile_size
        self.augment = augment
        self.use_ndwi = use_ndwi

        with rasterio.open(pre_path) as src:
            self.width = src.width
            self.height = src.height
            self.geo_transform = src.transform
            self.crs = src.crs

        self.tiles_x = (self.width + tile_size - 1) // tile_size
        self.tiles_y = (self.height + tile_size - 1) // tile_size
        self.total_tiles = self.tiles_x * self.tiles_y

    def __len__(self):
        return self.total_tiles

    def _compute_ndwi(self, tile):
        green = tile[..., 1].astype('float32')
        nir = tile[..., 3].astype('float32')
        ndwi = (green - nir) / (green + nir + 1e-8)
        return ndwi

    def _compute_rgb_diff(self, pre, post):
        gray_pre = cv2.cvtColor(pre[..., :3], cv2.COLOR_RGB2GRAY).astype('float32')
        gray_post = cv2.cvtColor(post[..., :3], cv2.COLOR_RGB2GRAY).astype('float32')
        diff = cv2.absdiff(gray_post, gray_pre)
        diff = cv2.GaussianBlur(diff, (5, 5), 0)
        diff_norm = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
        return diff_norm

    def _binarize(self, image):
        img8 = np.uint8(image * 255)
        _, mask = cv2.threshold(img8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return mask // 255

    def __getitem__(self, idx):
        x_idx = idx % self.tiles_x
        y_idx = idx // self.tiles_x
        x = x_idx * self.tile_size
        y = y_idx * self.tile_size
        window = rasterio.windows.Window(x, y, self.tile_size, self.tile_size)
    
        with rasterio.open(self.pre_path) as pre_src, rasterio.open(self.post_path) as post_src:
            pre_tile = np.transpose(pre_src.read(window=window), (1,2,0))
            post_tile = np.transpose(post_src.read(window=window), (1,2,0))
    
        # --- pad to tile_size if needed ---
        def pad_to_tile(tile):
            h, w, c = tile.shape
            pad_h = max(0, self.tile_size - h)
            pad_w = max(0, self.tile_size - w)
            if pad_h > 0 or pad_w > 0:
                tile = np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            return tile[:self.tile_size, :self.tile_size, :]
    
        pre_tile = pad_to_tile(pre_tile)
        post_tile = pad_to_tile(post_tile)
    
        # Normalize
        pre_tile = pre_tile.astype("float32")
        post_tile = post_tile.astype("float32")
        pre_tile /= pre_tile.max() if pre_tile.max() > 1 else 1
        post_tile /= post_tile.max() if post_tile.max() > 1 else 1
    
        # Compute flood mask
        if self.use_ndwi and pre_tile.shape[-1] >= 4:
            ndwi_pre = self._compute_ndwi(pre_tile)
            ndwi_post = self._compute_ndwi(post_tile)
            mask = (self._binarize(ndwi_post) == 1) & (self._binarize(ndwi_pre) == 0)
        else:
            diff = self._compute_rgb_diff(pre_tile, post_tile)
            mask = self._binarize(diff)
    
        mask = mask.astype("float32")[None, :, :]
    
        pre_tile = torch.from_numpy(np.transpose(pre_tile[..., :3], (2, 0, 1)))
        post_tile = torch.from_numpy(np.transpose(post_tile[..., :3], (2, 0, 1)))
        mask = torch.from_numpy(mask)
    
        if self.augment:
            pre_tile, post_tile, mask = self.augment(pre_tile, post_tile, mask)
    
        center_x = x + self.tile_size // 2
        center_y = y + self.tile_size // 2
    
        return pre_tile, post_tile, mask, (center_x, center_y), (x_idx, y_idx)
    

def get_flood_dataloaders(
        pre_path,
        post_path,
        tile_size=512,
        use_ndwi=True,
        val_fraction=0.2,
        batch_size=4,
        num_workers=2,
        augment=None
    ):

    dataset = FloodDataset(
        pre_path=pre_path,
        post_path=post_path,
        tile_size=tile_size,
        use_ndwi=use_ndwi,
        augment=augment
    )

    # split
    train_idx, val_idx = split_flood_dataset(dataset, val_fraction=val_fraction)

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)

    # loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = get_flood_dataloaders(
        pre_path="path_to_pre_flood_image.tif",
        post_path="path_to_post_flood_image.tif",
        tile_size=512,
        use_ndwi=True,
        val_fraction=0.2,
        batch_size=4,
        num_workers=2,
    )

    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))
