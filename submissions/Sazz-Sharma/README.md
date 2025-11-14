# Flood-Seg – `Sazz-Sharma`

This is a simple but geospatially-correct pipeline for the flood cutouts task.

**Goal:**  
Compare **pre-flood** and **post-flood** orthomosaics, find where **land turned into water**, and export:

- `affected.csv` – one row per affected tile (region),
- PNG cutouts – pre/post image patches for each affected tile.

I use **classical CV + geospatial tools**, not deep learning, to keep it lightweight and explainable in a Google Colab setting.

---

## 1. What I Did (Pipeline)

### 1.1 Load orthomosaics

- Pre-flood: `Hanumannagar_Preflood_Orthomosaic.tif`
- Post-flood: `Hanumannagar_Postflood_Orthomosaic.tif`
- Both are opened with `rasterio`.
- CRS: `EPSG:32645` (UTM zone 45N, meters).
- Pixel size: `0.04 m × 0.04 m` →  
  **Pixel area** = `0.04 × 0.04 = 0.0016 m²`.

### 1.2 Tiling (to avoid OOM in Colab)

- The pre-flood raster is split into tiles of **1024 × 1024** pixels.
- For each pre-flood tile:
  - I compute its **map bounds** (in meters) using the geotransform.
  - I use `rasterio.windows.from_bounds` to find the **corresponding window** in the post-flood raster.
  - I read both tiles and crop to their **common overlapping area** so that each pixel (i, j) refers to the same ground location.

### 1.3 Water detection (per tile)

Assuming band order: **R, G, B, NIR**.

For each tile:

- I compute an NDWI-like index:
  ```text
  NDWI = (G - NIR) / (G + NIR)
  ```
- Water pixels are identified where: `NDWI > 0.1` AND `intensity > 20` (to avoid dark noise).
- This creates binary masks: `pre_water` and `post_water`.

### 1.4 Change detection

- **Lost land** = pixels that were land (non-water) pre-flood but became water post-flood.
- Formula: `lost_land_mask = (~pre_water) & post_water`
- Only tiles with `area_lost_m2 ≥ 100 m²` are considered "affected".

### 1.5 Output generation

For each affected tile:
- **CSV row**: tile_id, center coordinates (lat/lon), total land area, lost land area.
- **PNG cutouts**: pre/post RGB images (normalized to 0-1 range).

---

## 2. Design Choices

### 2.1 Why NDWI instead of deep learning?

- **Lightweight**: Works in Colab without GPU requirements or large model downloads.
- **Interpretable**: Clear water detection logic based on spectral properties.
- **Fast**: Classical CV processes large orthomosaics efficiently.
- **Modular**: Easy to integrate with future ML approaches (SAM, DinoV3) for refinement.

### 2.2 Geospatial alignment

- Uses rasterio's `from_bounds()` to ensure pre/post tiles cover identical ground areas.
- Essential for accurate change detection at pixel level.

### 2.3 Tiling strategy

- 1024×1024 pixel tiles balance memory efficiency with processing overhead.
- Grid-based approach ensures complete coverage without gaps.

---

## 3. Future Extensions

This pipeline provides a solid foundation for:
- **SAM integration**: Use NDWI masks as prompts for Segment Anything Model refinement.
- **DinoV3 features**: Replace spectral indices with learned visual features for better water detection.
- **Multi-temporal analysis**: Extend to time series for flood progression tracking.
