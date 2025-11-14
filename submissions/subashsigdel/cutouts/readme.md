# Hanumannagar Flood Mapping – Submission

## Overview

My initial approach, after reading the problem statement, was to convert the TIF images into smaller JPG tiles and manually annotate or mask them using Roboflow to fine-tune a YOLO segmentation model. I know about Instance segmentation and semantic segmentation and yolo will not work for this, then I thought of using U-Net instead of YOLO for segmentation. However, after converting the TIF images to JPG, I ended up with many white images. After removing these, there were still over 3,000 images remaining, making it impractical to manually annotate and fine-tune a model.  

Therefore, I decided to generate **NDWI-based water masks** for all pre-flood and post-flood satellite tiles.

### NDWI (Normalized Difference Water Index)


NDWI = Green - NIR / Green + NIR

**Pixel classification:**  
- Water pixel → NDWI > threshold (0.1)  
- Land pixel → NDWI ≤ threshold  

**Workflow:**  
1. Read each tile from pre-flood and post-flood folders.  
2. Compute NDWI using the Green (Band 2) and NIR (Band 4) bands.  
3. Convert NDWI to a binary mask (1 = water, 0 = land).  
4. Save the mask as a GeoTIFF while preserving georeferencing.  
5. Visualize results using Folium maps.  

> ⚠️ Note: This method is fully automatic and scalable but does not provide pixel-perfect accuracy.

## Area and Centroids Calculation

- **area_m2**: Computed from polygon geometry after reprojecting to UTM (EPSG:32645) for accurate measurement in meters.  
- **Centroids**: Geometric center of each polygon in WGS84 coordinates (longitude and latitude).  

## CSV Output

The exported CSV contains the following columns:

| Column          | Description |
|-----------------|-------------|
| `centroid_lat`  | Latitude of polygon centroid (WGS84) |
| `centroid_lon`  | Longitude of polygon centroid (WGS84) |
| `area_m2`       | Polygon area in square meters |
| `geometry_wkt`  | Polygon geometry in WKT format |

This CSV provides **flood polygon locations, areas, and geometry**, and the results can be visualized by running the generated HTML file.

### NDWI Flood Mask
![NDWI Mask](flood_overlay.png)