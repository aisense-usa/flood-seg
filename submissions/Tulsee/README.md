# 🌊 Post–Flood Damage Detection – Submission

**Contributor:** Shankar Ghimire
**GitHub Handle / Folder:** Tulsee

---

## 📝 Overview

This submission addresses the **Post–Flood Damage Cutouts Challenge**. Using pre-flood and post-flood orthomosaics, the goal was to detect flood-affected areas, generate cutouts, and compute the real-world area lost.

The approach combines:

1. **Tiling large orthomosaics** into manageable patches (512×512 pixels).
2. **UNet-based semantic segmentation** to predict flooded areas.
3. **Change detection** by comparing pre-flood and post-flood masks.
4. **Geospatial calculations** to compute centroids and area (m²) using `rasterio` and `shapely`.
5. **Output CSV and cutouts** for each affected region.

---

## ⚙️ Method

1. **Tile Extraction:**

   - Split both pre-flood and post-flood images into 512×512 tiles.
   - Tiles are converted to **3-channel RGB** and normalized to `[0,1]`.

2. **Flood Mask Prediction:**

   - A pre-trained **UNet** predicts pixel-level flood masks for each tile.
   - A **threshold of 0.25** is applied to include subtle flood regions.

3. **Change Detection:**

   - Flood is considered where **post-flood mask > 0.25** and **pre-flood mask ≤ 0.25**.
   - This isolates new flooded areas while ignoring permanent water bodies.

4. **Geospatial Calculations:**

   - Each detected region is converted from pixels to **real-world coordinates** using `rasterio` transform.
   - **Centroids (longitude, latitude)** and **area in m²** are computed using `shapely`.

5. **Output:**

   - **CSV:** `affected.csv` with one row per detected region.
   - **Cutouts:** Pre/post tile images stored in `cutouts/` folder.

---

---

## 📄 CSV Columns

| Column                | Description                    |
| --------------------- | ------------------------------ |
| tile_id               | Unique tile identifier         |
| center_longitude      | Longitude of region centroid   |
| center_latitude       | Latitude of region centroid    |
| area_m2               | Area of affected region (m²)   |
| area_lost_m2          | Area lost due to flood (m²)    |
| pre_flood_land_image  | Path to pre-flood tile cutout  |
| post_flood_land_image | Path to post-flood tile cutout |

**Example row:**

| tile_id  | center_longitude | center_latitude | area_m2 | area_lost_m2 | pre_flood_land_image     | post_flood_land_image     |
| -------- | ---------------- | --------------- | ------- | ------------ | ------------------------ | ------------------------- |
| tile_019 | 85.12345         | 26.54321        | 2500.0  | 2500.0       | cutouts/tile_019_pre.png | cutouts/tile_019_post.png |

---

## 📊 Notes & Assumptions

- **Pixel Size:** Taken from TIFF metadata.
- **Threshold:** 0.25 probability for flood detection.
- **Empty Tiles:** Tiles with no data are skipped.
- **Alignment:** Pre-flood and post-flood images assumed spatially aligned. Misaligned tiles may affect detection.
- **Dtype:** All tiles converted to 3-channel `uint8` RGB.

---

## ✅ Deliverables

1. `affected.csv` – all detected flood regions with coordinates and area.
2. `cutouts/` – pre/post tile images corresponding to affected areas.
3. `README.md` – this document.

---

## 📌 Tools & Libraries

- Python 3.9+
- PyTorch, Torchvision
- OpenCV (`opencv-python-headless`)
- Rasterio, Shapely, Numpy, Pandas

---

## 🔍 Remarks

- The current submission detected **flooded areas based on the UNet predictions and thresholding**.
- Any empty or misaligned tiles were ignored.
- Further improvements could include:
  - Fine-tuning UNet with domain-specific flood data.
  - Adjusting thresholds for different flood intensities.
  - Post-processing with morphological operations to clean masks.

---

**End of Submission**
