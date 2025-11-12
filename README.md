
# Post–Flood Damage Cutouts Challenge

**Why:**
This challenge is the entry step for our Nepal AI & Computer Vision Bootcamp. Using pre- and post-flood orthomosaics, you’ll highlight damaged land and measure affected area. The goal is simple: find flooded land so communities can plan lost land recovery, collect insurance evidence, and build future resilience. Demonstrating how AI models can support disaster recovery and agricultural resilience across the Valley.



## Challenge

**Task:** Find land that changed due to flooding and measure how much.

**Input:** Two image sets of the same place — **pre-flood** and **post-flood** orthomosaics.

**Output table:** For each affected region, give **coordinates** (the **center** of the land area) and **area lost (m²)**.

**Or, in short:** Use pre/post flood orthomosaics to segment flood-affected land and output a table of affected regions with **centroid coordinates + area (m²)**.

**Freedom:** Any method (classical CV, U-Net/DeepLab/SegFormer, SAM, thresholding). Any tools: Python, OpenCV, PyTorch/TF, QGIS—your choice. Pretrained or quick train—no restriction.


### If you have access to AI and can Vibe Code dont shy on using them, can give brownie points

---


## Data

* **Small sample (for quick testing):**



* **Full orthomosaics (Google Drive):**

  * Pre-flood: **[[Drive link – pre]](https://drive.google.com/file/d/1by9yXKye9QkaN9dAqWnMJkNglG9AD3rM/view?usp=sharing)**
  * Post-flood: **[[Drive link – post]](https://drive.google.com/file/d/1x8VdZBs25F9EkWnJw1XR4xnUfBHn0eyY/view?usp=sharing)**

> Tip: You can also use a small portion of both the orthomosiac

--

**How to submit (step-by-step, simple):**

1. On GitHub, **Fork** this repo (top-right ➜ Fork).
2. **Clone** your fork locally: `git clone https://github.com/<you>/<repo>.git && cd <repo>`
3. **Create branch:** `git checkout -b submit/<github_handle>`
4. **Make your folder:** `mkdir -p submissions/<github_handle>/cutouts`
5. Run your code to generate **cutouts** and **affected.csv** into that folder. Add a **README.md**, **requirements.txt**.
6. **Commit & push:**

   ```
   git add submissions/<github_handle>
   git commit -m "Submission"
   git push origin submit/<github_handle>
   ```
7. On GitHub, open a **Pull Request** from your branch to this repo.
8. **PR title:** `[SUBMISSION] <Full Name> (<github_handle>)`.

If you are not familiar with GitHub, in the Flood-seg (Google Drive) -> please make a new folder in the folder **[Submissions](https://drive.google.com/drive/folders/12szwpZg5XQ7ZnWQwGk8XrHqbOrhPqZzR?usp=sharing)** with your name — for example submissions/snehalraj_chugh/ — and upload your cutouts, affected.csv, README, and code files there.


--

## Deliverables to be judged on

### 1) CSV (one row per region)

**Columns (exact names):**
`tile_id, center_longitude, center_latitude, area_m2, area_lost_m2, pre_flood_land_image, post_flood_land_image`

**Notes:**

* `area_m2 = positive_pixels × (pixel_size_m)^2`
* `area_lost_m2` = area that turned from **land → water/flood** (or looks destroyed). Briefly explain your rule in your README.

**Example:**

| tile_id  | center_longitude | center_latitude | area_m2 | area_lost_m2 | pre_flood_land_image                                      | post_flood_land_image                                     |
|-----------|------------------|-----------------|----------|--------------|------------------------------------------------------------|------------------------------------------------------------|
| tile_019  | -76.749189       | 39.274509       | 312.4    | 280.7        | submissions/snehalraj/cutouts/tile_019_pre.png             | submissions/snehalraj/cutouts/tile_019_post.png            |


### 2) Cutout images (good to have)

* Folder: `submissions/<github_handle>/cutouts/`
* Filenames (recommended): `tile_<id>_pre.png` and `tile_<id>_post.png`
* Pre and post can be separate files (simpler) **or** a single combined image.

  * If combined, still fill both CSV columns with the same path.
 

### 3) Short README (≤1 page)

* What you did (1–2 short paragraphs), how you computed **area** and **centroids**, any assumptions (**pixel size**, thresholds).


## Repo Layout (simple)

```
flood-seg/
├─ submissions/
│  └─ <github_handle>/
│     ├─ cutouts/
│     ├─ affected.csv
│     └─ README.md
└─ README.md
```

---

## Nice-to-Have

* **Tiny Demo Website** to visualize results (local or hosted):

  * Streamlit or Gradio.
  * Show pre vs post with detected regions and area numbers.


---

## Scoring (lightweight)

* CSV + cutouts present & paths valid (40)
* Sensible centroids and areas (40)
* README clarity (20)

---

## Questions

Open an **Issue** in this repo with a clear title (e.g., “Download issue – pre-flood link”) and we’ll reply there.

---

**Good luck—and thank you for building tools that help flood-affected communities.**
