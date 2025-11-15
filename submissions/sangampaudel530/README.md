# 🌊 Post–Flood Damage Cutouts Demo

## Overview

This project demonstrates a workflow for detecting flood-affected areas using pre- and post-flood orthomosaics.
The goal is to identify land that changed due to flooding, calculate affected areas, and provide a sample visualization.

This is a **demo submission** for the Nepal AI & Computer Vision Bootcamp challenge.

---

## Workflow

### 1️⃣ Extract Tiles

* Large TIFF orthomosaics (pre-flood and post-flood) were divided into smaller, manageable tiles using `src/extract_tiles.py`.
* Only **meaningful tiles** containing land/water were retained; mostly white or empty tiles were discarded using `src/filter_white_tiles.py`.
* Tile size: **512 × 512 pixels** (configurable).

### 2️⃣ Flood Detection & Visualization

* Pre- and post-flood tiles were compared using a **pixel difference approach** in `src/visualize_difference.py`.
* Changes between pre/post tiles are highlighted in **difference images**.
* This provides a simple visualization of flood-affected areas for the demo.

### 3️⃣ Area & Centroid Calculation

* Each tile is treated as a square region; area in m² is calculated using **tile size × pixel resolution**.
* Tile center coordinates (`center_x`, `center_y`) are computed and recorded in the CSV file.
* CSV columns:

  * `tile_id` – ID of the tile
  * `center_x`, `center_y` – center coordinates of the tile
  * `area_m2` – total area of the tile
  * `pre_flood_tile` – path to pre-flood tile
  * `post_flood_tile` – path to post-flood tile

### 4️⃣ Output

* `cutouts/` → extracted pre/post flood tiles
* `difference_visualization/` → sample images showing detected changes
* `affected.csv` → CSV listing tile IDs, center coordinates, area, and tile paths

---

## Assumptions

* Tiles with **>95% white pixels** are ignored (empty tiles).
* Each pixel represents a square area from orthomosaic metadata (assumes real-world resolution is known).
* This is a **demo workflow**; no advanced ML segmentation or NDWI detection is applied.
* Number of tiles extracted is configurable (demo uses 5 tiles per set).

---

## File Structure

```
submissions/<github_handle>/
├─ cutouts/
│   ├─ tile_0_pre.png
│   ├─ tile_0_post.png
│   └─ ...
├─ difference_visualization/
│   ├─ tile_0_diff.png
│   └─ ...
├─ src/
│   ├─ extract_tiles.py
│   ├─ filter_white_tiles.py
│   └─ visualize_difference.py
├─ affected.csv
└─ README.md
```

---

## How to Reproduce

1. Run `src/extract_tiles.py` to extract tiles from pre- and post-flood TIFFs.
2. Run `src/filter_white_tiles.py` to remove mostly empty tiles.
3. Run `src/visualize_difference.py` to create difference images and calculate area per tile.
4. Check `affected.csv` for tile information, centroids, and area values.

---

## Notes

* This demo focuses on showing the **workflow and methodology**, not precise flood segmentation.
* A real-world implementation can include **U-Net, DeepLabV3, SegFormer, or NDWI-based methods** for accurate flood detection.
* All images and CSV are prepared to match the submission format required for the bootcamp challenge.
