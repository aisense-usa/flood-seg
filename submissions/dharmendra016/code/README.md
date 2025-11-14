# 🌊 Flood Damage Detection Using Siamese U-Net++

**Author:** Dharmendra Singh Chaudhary
**Challenge:** Post-Flood Damage Cutouts Challenge — Nepal AI & CV Bootcamp

---

## 📌 1. Problem Summary

The task is to determine which areas were affected by flooding using **two large orthomosaic GeoTIFF images**:

* **Pre-Flood Orthomosaic**
* **Post-Flood Orthomosaic**

These images cover several square kilometers and contain spatial metadata such as **CRS**, **bounds**, and **geotransform parameters**.

### 🎯 The goal of the challenge:

✔ Detect flood-affected land areas
✔ Segment flooded pixels tile-wise
✔ Compute centroid coordinates (latitude/longitude)
✔ Compute area lost (m²)
✔ Export a CSV report of affected tiles
✔ Save pre- and post-flood tile cutouts

A major challenge: **No ground-truth segmentation masks were provided**, so I designed a **self-supervised pipeline** using pseudo-label generation + Siamese U-Net++.

---

## 🧠 2. Overall Project Pipeline

This section follows the workflow implemented in **flood_detection.ipynb**.

---

### **Step 1 — Explore & Load Orthomosaics**

* Used **Rasterio** to inspect orthomosaic metadata.
* Images were extremely large (10k–40k pixels per dimension), so full loading was impossible.
* Used **streamed tile reading** with `rasterio.windows.Window`.

---

### **Step 2 — Align & Crop Pre/Post TIFFs**

A key challenge:

❗ **The pre- and post-flood TIFFs had different shapes**, causing mismatches in tile generation.

I implemented **aligned overlapping crop** using:

```python
aalign_and_crop_to_overlap(pre, post, out_pre, out_post)
```

This produced:

* `pre_aligned.tif`
* `post_aligned.tif`

Both having identical:

* CRS
* resolution
* spatial extent
* dimensions

---

### **Step 3 — Tile Extraction Using Raster Windows**

* Split images into **512 × 512 pixel tiles**.
* Raster windows avoid loading the entire TIFF.
* Each tile preserves geospatial metadata used later for computing:
  ✔ pixel area in m²
  ✔ centroid coordinates

---

### **Step 4 — Pseudo-Mask Generation (Self-Supervised Change Detection)**

Since no ground-truth mask exists, I generated **pseudo labels** using two strategies:

### ✔ If NIR channel exists — NDWI

```
NDWI = (Green - NIR) / (Green + NIR)
```

* Compute NDWI for both pre and post images.
* Apply **Otsu threshold** to detect water.
* Flood mask = `Water_post AND (NOT Water_pre)`.

### ✔ If only RGB — Difference Masking

* Convert pre/post tiles to grayscale.
* Compute **absolute intensity difference**.
* Apply Gaussian smoothing.
* Apply Otsu threshold.

### ✔ Noise removal

* Morphological opening
* Minimum connected component area filtering

Pseudo masks were saved as:

```
tile_x_pre.png
tile_x_post.png
tile_x_mask.png
```

These masks served as **training labels** for the Siamese U-Net++.

---

## 🧩 5. Dataset Loader (FloodDataset)

The custom PyTorch `FloodDataset` performs:

* Raster window extraction
* Padding of non-complete tiles
* Normalization to [0, 1]
* Pseudo-mask generation (NDWI or RGB diff)
* Returns:

  * Pre-flood tile
  * Post-flood tile
  * Pseudo-flood mask
  * Tile centroid
  * Tile grid index

This makes the pipeline memory-efficient and geospatially accurate.

---

## 🔥 6. Model — **Siamese U-Net++**

I implemented a full **nested U-Net++ architecture** adapted to a Siamese setting.

### 🔑 Key Features

✔ Shared encoder processes **pre-flood** and **post-flood** tiles using the same weights
✔ Feature maps are **concatenated** to highlight changes
✔ **Nested dense skip connections** (core of U-Net++)
✔ **Deep supervision** using four output heads
✔ Final segmentation = average of deep supervision outputs

This architecture is highly effective for **fine-grained change detection**.

---

## 🏋️ 7. Training Pipeline

### 📌 Train/Val Split

Used a **spatial split** to prevent leakage:

* First 80% image rows → **Training**
* Last 20% image rows → **Validation**

This prevents the model from seeing nearby regions during validation.

### 📌 Loss Function

A combination for stable segmentation:

* **BCEWithLogitsLoss**
* **Dice Loss**

### 📌 Optimizer

* **AdamW**
* Learning rate = `1e-4`

### 📌 Mixed Precision Training

Used:

```
torch.amp.autocast
GradScaler
```

for speed & lower memory usage.

### 📌 Early Stopping

* Patience = 3 epochs
* Automatically saves the best checkpoint.

---

## 🧊 8. Inference Pipeline (GeoTIFF → Mask → CSV)

The inference script performs:

1. Load `pre_aligned.tif` and `post_aligned.tif`.
2. Slide 512×512 window over full raster.
3. Predict mask using Siamese U-Net++.
4. Threshold mask.
5. Extract connected components.
6. Compute centroid in pixel coords.
7. Convert pixel coords → lat/lon using geotransform.
8. Compute area in m²:

```
area = positive_pixels × (pixel_width × pixel_height)
```

9. Save tile cutouts:

```
tile_x_pre.jpg
tile_x_post.jpg
```

10. Append to `affected.csv`:

| tile_id | longitude | latitude | area_m2 | pre_image | post_image |

---

## ⚠️ 9. Time Constraints — Full Inference Not Completed

Although the entire inference pipeline is correctly implemented, **I could not run full inference** due to:

* Orthomosaics containing tens of thousands of tiles
* Limited Kaggle GPU runtime
* Heavy Siamese Nested U-Net++ architecture
* Slow tile-by-tile prediction
* Strict submission deadline

➡️ Therefore, I prepared the full pipeline but **could not finish generating the final CSV**.

---

## 🧩 10. Key Challenges & Solutions

### **Challenge 1 — Mismatched Dimensions**

✔ Solved via aligned overlapping cropping

### **Challenge 2 — No Ground Truth**

✔ Solved by generating NDWI/RGB pseudo masks

### **Challenge 3 — Huge Raster Sizes**

✔ Solved with raster window streaming

### **Challenge 4 — Spatial Leakage**

✔ Avoided using spatial train/val split

### **Challenge 5 — Computing Geo-Area & Centroid**

✔ Used raster transform + pixel resolution

### **Challenge 6 — Kaggle Time Limit**

✔ Complete pipeline implemented but full inference incomplete

---

## 🏁 11. Conclusion

This project successfully integrates:

* Geospatial raster processing
* Pseudo-label generation
* Change detection
* Siamese U-Net++ model
* Tile-based segmentation
* Geo-coordinate conversion
* Area estimation
* CSV reporting

Even without ground-truth segmentation masks, the system learned to detect flooded land using **self-supervision**.

### 💡 What I Learned

* Handling large orthomosaic TIFFs
* Building geospatial ML pipelines
* Remote sensing change detection
* Siamese deep learning architectures
* Geographical coordinate and area computations

This challenge offered valuable experience building a **real-world disaster response solution**.

---

## 📚 References

* Siamese Networks: [https://www.geeksforgeeks.org/nlp/siamese-neural-network-in-deep-learning/](https://www.geeksforgeeks.org/nlp/siamese-neural-network-in-deep-learning/)
* U-Net Architecture Explained: [https://www.geeksforgeeks.org/machine-learning/unet-architecture-explained/](https://www.geeksforgeeks.org/machine-learning/unet-architecture-explained/)
