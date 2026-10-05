# Phase 1A: Dataset Feasibility Research & Methodology Report

**Project**: Explainable AI Based Deforestation Anomaly Detection Using Geospatial Satellite Images  
**Execution Context**: MacBook Air M3 (8 GB Unified Memory, macOS, Python 3.12, TensorFlow/Keras)  
**Status**: Research Only — Zero Downloads, Zero Package Installations, Zero Code Modifications Executed.

---

## 1. Audit of Current Project Documentation (Task 1)

A review of the foundation documentation (`README.md`, `docs/methodology.md`, `docs/dataset_plan.md`, `docs/evaluation_protocol.md`, `docs/decisions.md`) reveals four methodological assumptions that must be qualified prior to data acquisition:

1. **Task Formulation Ambiguity (Patch Classification vs. Dense Segmentation)**:
   - *Current text*: Documentation discusses generating $256 \times 256$ pixel change masks alongside Grad-CAM explainability interchangeably.
   - *Correction*: Standard Grad-CAM requires a scalar classification logit (e.g., $y \in [0, 1]$ for patch-level change probability). Computing Grad-CAM on dense pixel-level segmentation requires aggregating pixel logits over an arbitrary region of interest (e.g., Seg-Grad-CAM). The project must explicitly adopt either **patch-level classification with Grad-CAM attribution** as the primary baseline, or formalize an aggregated loss for segmentation explainability.

2. **PRODES Temporal Precision and Resolution Limitations**:
   - *Current text*: Implies PRODES polygons can serve as exact temporal ground truth for any arbitrary $T1$ and $T2$ observation pair.
   - *Correction*: PRODES is an **annual cumulative accounting product** monitoring the period from **August 1 of year $Y-1$ to July 31 of year $Y$**. It does not record the exact day or week trees were felled. Furthermore, PRODES has a **Minimum Mapping Unit (MMU) of 6.25 hectares** (approx. 625 pixels at 10 m resolution). Forest clearings smaller than 6.25 ha are omitted by INPE interpreters, creating unavoidable label noise if evaluated at the single-pixel level.

3. **Deforestation Sparse Anomaly vs. Filtered Benchmarks**:
   - *Current text*: Mentions 90–95% accuracy as an aspirational benchmark.
   - *Correction*: On realistic, uncurated satellite landscapes, deforestation accounts for $<2\%$ of pixels. A trivial constant model predicting "no change" achieves $>98\%$ accuracy. High accuracy numbers reported in academic literature (e.g., Alshehri et al.) were achieved only after filtering out patches with $<10\%$ deforestation. The evaluation protocol must treat change as a rare anomaly and prioritize positive-class Precision, Recall, and IoU.

4. **Remote Sensing Alignment Constraints**:
   - *Current text*: Recommends evaluating 6 spectral bands (including 20 m SWIR bands B11 and B12).
   - *Correction*: Ingesting 20 m bands requires spatial resampling and multi-resolution grid interpolation, which increases preprocessing memory overhead. For an 8 GB RAM Mac, native 10 m bands (B02, B03, B04, B08) provide superior computational simplicity and eliminate resampling artifacts.

---

## 2. Investigation of Research References (Task 2)

### Reference 1: Alshehri et al. (IEEE GRSL 2024)
- **Citation**: M. S. Alshehri, A. Ouadou, and G. J. Scott, *"Deep Transformer-Based Network Deforestation Detection in the Brazilian Amazon Using Sentinel-2 Imagery,"* in *IEEE Geoscience and Remote Sensing Letters*, vol. 21, pp. 1–5, 2024. DOI: [10.1109/LGRS.2024.3355104](https://doi.org/10.1109/LGRS.2024.3355104).
- **Core Methodology**: Adapted the ChangeFormer transformer architecture (originally designed for urban change detection) to bi-temporal Sentinel-2 optical imagery paired with PRODES ground reference masks in the Brazilian Amazon.
- **Patch Dimensions & Bands**:
  - Evaluated both $256 \times 256$ and $512 \times 512$ chips; determined that **$256 \times 256$ patches were superior** for capturing smaller deforestation events without overwhelming background class imbalance.
  - Incorporated spectral vegetation features (NDVI, Enhanced Vegetation Index / EVI, and Color-Shifted Infrared band combinations).
- **Dataset Construction & Filtering**:
  - **Critical Finding**: The authors explicitly filtered out patches where deforestation accounted for less than $10\%$ of the chip area. This artificially balanced the training dataset and eliminated difficult sparse-change samples, directly inflating reported metrics.
- **Reported Performance**:
  - Self-reported Overall Accuracy: 93%, F1-score: 84.62%–90%, IoU: 82%.
  - Note: Independent audit by Konishi & Pedrini (arXiv:2512.08075, 2025) confirmed Alshehri's reported F1 of 84.62% on their specific filtered benchmark, but noted that such performance does not generalize directly to uncurated landscapes.
- **Public Artifact Availability**:
  - The authors did **not** publish a downloadable, pre-packaged patch dataset. Researchers must regenerate patches directly from Sentinel-2 L2A tiles and INPE PRODES vector shapefiles.

### Reference 2: Manuel Horvey Daniel et al. (ChangeFormer Hugging Face)
- **Source**: [https://huggingface.co/manuelhorveydaniel/ChangeFormer](https://huggingface.co/manuelhorveydaniel/ChangeFormer) (Hugging Face model card and repository).
- **Context**: Undergraduate / final-year project by Emmanuel Amey, Sammuel Young Appiah, Asare Prince Owusu, and Yaaya Pearl Apenu, supervised / adapted from Alshehri et al.
- **Data Configuration**:
  - Bi-temporal pairs: 2020 vs. 2021 Sentinel-2 Level-2A imagery.
  - Bands: 4 native 10 m bands — **B02 (Blue), B03 (Green), B04 (Red), B08 (NIR)**.
  - Region: Brazilian Amazon (specifically citing APA Triunfo do Xingu, Pará).
  - Patch size: $256 \times 256$ pixels.
  - Stated dataset volume: 19,560 patches.
- **Performance Claims & Forensic Red Flags**:
  - Self-reported validation metrics: **F1-score of 0.9986**, **IoU of 0.9972**.
  - **Scientific Critique**: An F1 of 0.9986 on satellite change detection is an extreme statistical anomaly. In remote sensing, atmospheric variations, cloud shadow edges, phenological shifts, and PRODES polygon delineation errors create a natural noise floor. An IoU of 0.9972 strongly indicates **catastrophic spatial data leakage** (e.g., random patch splitting across contiguous, overlapping scene tiles where adjacent patches appear in both train and validation sets) or metric calculation dominated by the background no-change class.
  - **Artifact Availability**: The underlying 19,560 patch files are **not** hosted on Hugging Face or public storage. The referenced GitHub repository (`github.com/manuelhorvey/ChangeFormer`) does not host the code under that user path.
  - **Framework Conflict**: The model implementation relies on PyTorch (`torch`, `torchvision`, `timm`), which is strictly forbidden under our project constraints.

---

## 3. Comparative Evaluation of Acquisition Providers (Task 3)

| Criterion | A. Microsoft Planetary Computer | B. Copernicus Data Space (CDSE) | C. Google Earth Engine (GEE) |
|---|---|---|---|
| **Sentinel-2 L2A Access** | **Native STAC API & Cloud-Optimized GeoTIFFs (COGs)**. Pre-computed BOA surface reflectance. | Available via OData & STAC APIs. Full product downloads standard. | Available via Earth Engine image collections (`COPERNICUS/S2_SR_HARMONIZED`). |
| **Windowed / Sub-Scene Retrieval** | **Yes (Exceptional)**. Using HTTP range requests (`rasterio` / `GDAL` `/vsicurl/`), individual $256 \times 256$ or $2500 \times 2500$ bounding boxes can be read without downloading full 800 MB granules. | Limited. Most workflows require downloading full SAFE zip archives (500–1000 MB per scene). | Yes, but requires server-side reduction and asynchronous task export. |
| **PRODES Reference Availability** | Not hosted as a native managed STAC collection. Ingested via INPE TerraBrasilis WFS/shapefile. | Not hosted natively. | Available via community datasets or user table assets. |
| **Authentication & Accounts** | **No account or API key required** for basic STAC queries; free SAS signing via public endpoint. | Mandatory user account registration and OAuth2 token workflow. | Mandatory Google Cloud project registration, account approval, and cloud credentials. |
| **Storage Impact on 8 GB Mac** | **Minimal**. Only requested bounding-box tiles/bands are fetched to disk (e.g., $<500$ MB total). | High risk of filling local disk if downloading multiple 800 MB SAFE scenes. | Low local storage, but high cloud configuration overhead. |
| **Provider Recommendation** | **RECOMMENDED PRIMARY ROUTE** | Secondary / Fallback only | Rejected (unnecessary credential/cloud barrier) |

---

## 4. Study Area Recommendation and Empirical Evidence (Task 4)

### Evaluated Candidates
1. **Novo Progresso Region, Pará** (BR-163 Corridor):
   - Heavily deforested frontier; however, the area suffers from persistent heavy atmospheric smoke during the July–September fire season and frequent convective cloud clusters.
2. **Jamari / Candeias do Jamari / Porto Velho Region, Rondônia**:
   - **Recommended Choice**.
   - **Geographic Bounds**: Centered around the boundary of the Jamari National Forest (FLONA de Jamari) and surrounding colonization settlements (Candeias do Jamari and Itapuã do Oeste).
   - **Sentinel-2 MGRS Tile**: **`20LMR`** (UTM Zone 20L).
   - **Bounding Box (EPSG:4326)**:
     - West: `-63.30°`, South: `-9.40°`
     - East: `-62.70°`, North: `-8.90°`
     - Approximate spatial extent: $pprox 66 	ext{ km} 	imes 55 	ext{ km}$ ($3,630 	ext{ km}^2$), easily fitting within a single MGRS tile (`20LMR`).

### Supporting Empirical Evidence for Jamari (`20LMR`)
- **PRODES Activity**: Rondônia is one of the classic "fishbone" deforestation landscapes in the Brazilian Amazon. INPE TerraBrasilis records consistent, substantial annual clear-cut polygons in the municipalities of Candeias do Jamari and Porto Velho across 2020, 2021, and 2022.
- **Topological Contrast**: FLONA de Jamari provides an intact primary rainforest control area, while the adjacent agrarian settlements provide clear-cut deforestation events, guaranteeing strong positive and negative examples within identical atmospheric conditions.
- **Atmospheric Clarity in Dry Season**: Rondônia experiences a pronounced dry season from **late July through early September**, with historical Sentinel-2 acquisitions regularly achieving $<5\%$ scene-wide cloud cover.
- **Academic Benchmark Precedence**: Tile `20LMR` is extensively utilized in Brazilian remote sensing research (e.g., INPE's `sits` package documentation and validation benchmarks).
- **Hardware Compatibility**: A $66 	imes 55$ km sub-region at 10 m resolution comprises approximately $6,600 	imes 5,500$ pixels. Extracted as $256 	imes 256$ non-overlapping patches, it produces $pprox 500$ patches, perfectly sized for 8 GB RAM exploration.

---

## 5. Defensible Temporal-Label Strategy (Task 5)

### PRODES Accounting Period Alignment
- INPE defines the PRODES deforestation year $Y$ as running from **August 1 of year $Y-1$ to July 31 of year $Y$**.
- To ensure astronomical and methodological alignment, the bi-temporal Sentinel-2 pair must match this annual boundary:
  - **$T1$ Baseline Observation**: **August 2020** (dry season, start of PRODES 2021 cycle).
  - **$T2$ Target Observation**: **August 2021** (dry season, end of PRODES 2021 cycle).
  - **Reference Deforestation Layer**: **PRODES Year 2021** (`PRODES_2021`).

### Mitigation of Date Mismatches and Temporal Noise
1. **Pre-Existing Deforestation**: All clearings that occurred prior to August 1, 2020 (PRODES years $\le 2020$) must be rasterized as an **exclusion mask** (`mask_prior_loss`). These pixels are neither intact forest nor new change, and must be ignored during loss computation and metric evaluation.
2. **Post-$T2$ Deforestation**: Because $T2$ is acquired in August 2021, any clearing that took place after the $T2$ acquisition timestamp belongs to PRODES 2022 and will not appear in the PRODES 2021 reference layer.
3. **Non-Forest and Hydrography**: Natural savannahs, cerrado formations, and rivers (INPE layers `nao_floresta` and `hidrografia`) must be masked out to prevent false positive spectral confusion.
4. **Cloud / Shadow Masking**: Sentinel-2 Scene Classification Layer (SCL) pixels flagged as cloud high/medium probability (values 8, 9) and cloud shadow (value 3) must be dynamically masked for both $T1$ and $T2$.

---

## 6. Spectral Band Recommendation (Task 6)

### Comparison: 4 Native Bands vs. 6 Multi-Resolution Bands

| Feature | Option A: 4 Native 10 m Bands (`B02, B03, B04, B08`) | Option B: 6 Bands (`B02, B03, B04, B08, B11, B12`) |
|---|---|---|
| **Bands Included** | Blue (B2), Green (B3), Red (B4), NIR (B8) | Blue, Green, Red, NIR, SWIR-1 (B11), SWIR-2 (B12) |
| **Native Resolution** | 10 m for all 4 bands | 10 m (B2–B8); 20 m (B11, B12) |
| **Spatial Resampling** | **None required**. True native pixel geometry. | Bilinear or bicubic interpolation required to resample 20 m to 10 m. |
| **Input Tensor Size ($256 	imes 256$)** | Bi-temporal pair: $256 	imes 256 	imes 8$ channels | Bi-temporal pair: $256 	imes 256 	imes 12$ channels |
| **RAM Footprint (8 GB Mac)** | **Low**. Fast generator streaming; minimal memory pressure. | 50% larger tensor in RAM; increased risk of memory pressure. |
| **Reference Alignment** | Directly matches Manuel Horvey Daniel / ChangeFormer 4-band setup. | Common in advanced forestry literature, but adds preprocessing friction. |

### Recommendation
**Adopt Option A (4 native 10 m bands: B02, B03, B04, B08) for the initial baseline.**  
*Justification*: Eliminates resampling artifacts, halves storage requirements, maintains strict 10 m spatial fidelity, and guarantees smooth execution within the 8 GB unified memory limit of the MacBook Air M3. Bands B11 and B12 are reserved as an ablation study for future phases.

---

## 7. Practical Dataset Size and Storage Budget (Task 7)

### Target Patch Volume
- **Conservative Initial Benchmark**: **500 to 1,000 bi-temporal patch pairs** of size $256 	imes 256$ pixels.
- Extracted from the bounded Jamari sub-AOI ($66 	imes 55$ km) using a stride of 256 (non-overlapping) or 192 (modest overlap).

### Storage Calculations (Verified Arithmetic)

$$	ext{Bytes per single patch} = 256 	imes 256 	imes 4 	ext{ bands} 	imes 2 	ext{ bytes (uint16)} = 524,288 	ext{ bytes} pprox 0.524 	ext{ MB}$$

- **Bi-Temporal Pair ($T1 + T2$)**: $0.524 	ext{ MB} 	imes 2 = 1.05 	ext{ MB}$.
- **Binary Ground Truth Mask ($256 	imes 256 	imes 1$ uint8)**: $65.5 	ext{ KB} pprox 0.065 	ext{ MB}$.
- **Total per sample**: $pprox 1.12 	ext{ MB}$ (uncompressed uint16 raw reflectance).

| Dataset Size | Raw Patch Storage (uint16) | Processed float32 / TFRecord | Total Disk Footprint |
|---|---|---|---|
| **500 pairs** | $pprox 560 	ext{ MB}$ | $pprox 1.1 	ext{ GB}$ | $pprox 1.7 	ext{ GB}$ |
| **1,000 pairs** | $pprox 1.12 	ext{ GB}$ | $pprox 2.2 	ext{ GB}$ | $pprox 3.3 	ext{ GB}$ |

- **Temporary Download Buffer**: Retrieving the 4 bands for $T1$ and $T2$ over the $66 	imes 55$ km AOI requires approximately **$400 	ext{ MB}$** of windowed GeoTIFFs.
- **PRODES Vector GeoJSON**: $pprox 15 	ext{ MB}$.
- **Safe Upper Limit for Phase 1 Download**: **$\le 2.0 	ext{ GB}$ total disk consumption** (well within the 88 GiB available disk headroom).

---

## 8. Spatial Split and Evaluation Strategy (Task 8)

### Eliminating Spatial Autocorrelation
- **Spatial Block Partitioning**: The Jamari study region will be split into three contiguous geographic blocks:
  - **Training Block (West / Central)**: 60% of spatial area.
  - **Validation Block (North / Central)**: 20% of spatial area.
  - **Test Block (East / South)**: 20% of spatial area.
- **Buffer Zones**: A physical buffer zone of **at least 2.5 km (10 patch widths)** will separate the training block from validation and test blocks. Patches falling within the buffer corridor are discarded.
- **Zero Test Leakage**: The test block will remain completely unread and unvisited during preprocessing parameter tuning, model selection, and hyperparameter search.

### Task Formulation: Patch-Level Classification vs. Pixel Segmentation
- **Scientific Decision**: We formulate the primary baseline as **Patch-Level Binary Deforestation Classification**:
  - *Input*: Bi-temporal pair $(X_{T1}, X_{T2}) \in \mathbb{R}^{256 	imes 256 	imes 8}$.
  - *Label*: $y \in \{0, 1\}$, where $y = 1$ if the cumulative PRODES deforestation within the $256 	imes 256$ patch exceeds a verified minimum threshold ($\ge 1.0 	ext{ hectare}$, or $\ge 100 	ext{ pixels}$ of genuine forest loss), and $y = 0$ if the patch contains $< 0.1 	ext{ ha}$ of change. Patches with marginal, ambiguous change ($0.1–1.0 	ext{ ha}$) are excluded from the binary classification test set.
- **Why This Formulation Fits the Project Best**:
  1. Standard Grad-CAM computes gradients $rac{\partial y_c}{\partial A^k}$ with respect to a **scalar class score** $y_c$. It operates naturally and mathematically cleanly on patch classification.
  2. The resulting Grad-CAM heatmap ($256 	imes 256$) produces a dense spatial localization of where the network perceived deforestation, which can then be directly compared against the PRODES pixel mask using IoU and Dice metrics!
  3. It circumvents the severe 6.25 ha MMU delineation boundary noise inherent in PRODES polygon boundaries.

### Evaluation Metrics Suite
- **Mandatory Metrics**:
  - Positive-Class Precision ($rac{TP}{TP + FP}$)
  - Positive-Class Recall ($rac{TP}{TP + FN}$)
  - Positive-Class F1-Score
  - Bounding / Localization IoU between thresholded Grad-CAM heatmap and PRODES mask
  - Confusion Matrix ($TP, FP, TN, FN$)
  - Overall Accuracy (reported with explicit disclaimers regarding class imbalance)
- **Baseline Comparison**: A naive "No-Change Predictor" baseline ($y = 0$ everywhere) will be evaluated alongside the deep learning model to prove that high accuracy is not merely an artifact of class imbalance.

---

## 9. Baseline Architecture Recommendation (Task 9)

### Architecture Comparison

| Dimension | Option A: Early-Fusion CNN | Option B: Compact Siamese CNN with Difference Head |
|---|---|---|
| **Input Structure** | Concatenated tensor: $[256, 256, 8]$ ($T1$ and $T2$ stacked) | Two parallel tensors: $[256, 256, 4]$ ($T1$) and $[256, 256, 4]$ ($T2$) |
| **Feature Extraction** | Single standard CNN backbone. Cross-temporal interactions occur in layer 1. | Shared-weight CNN backbone ($W_{T1} \equiv W_{T2}$). Extracts independent feature representations. |
| **Temporal Inductive Bias** | Weak. Model must learn from scratch that channels $0..3$ correspond to channels $4..7$. | **Strong**. Weight sharing enforces identical representation learning on both dates. |
| **Change Mechanism** | Implicit convolution across stacked channels. | **Explicit Feature Differencing**: $D = |F(X_{T2}) - F(X_{T1})|$. |
| **Explainability (Grad-CAM)** | Computes gradients on single trunk; heatmaps cannot easily disentangle $T1$ vs. $T2$ contributions. | **Direct Attribution**: Gradients can be projected back through both Siamese branches to verify what changed between $T1$ and $T2$. |
| **RAM & Compute on M3** | Very low ($pprox 2.5 	ext{M}$ parameters). | Low ($pprox 3.0 	ext{M}$ parameters with shared weights). Highly memory-safe on 8 GB RAM. |

### Recommendation
**Adopt Option B: A Compact Siamese CNN with Feature Differencing Head in TensorFlow / Keras.**  
*Justification*:
- Weight sharing preserves the physical symmetry of optical reflectance.
- Explicit feature subtraction ($|F_{T2} - F_{T1}|$) provides a direct remote sensing inductive bias for spectral change.
- Native Keras Functional API (`tf.keras.Model`) implementation requires $<15$ MB model memory footprint.
- Enables clear, mathematically rigorous Grad-CAM backpropagation from the scalar change logit back into the convolutional feature representations.

---

## 10. Summary of Risks and Unresolved Questions

1. **PRODES Polygon Vector Availability for Target Bounding Box**:
   - *Risk*: INPE TerraBrasilis WFS endpoint availability can experience transient downtime or GeoServer schema updates.
   - *Mitigation*: Verify WFS query response for tile `20LMR` before running download pipelines; fallback to direct download of the official 2021 Rondônia shapefile.
2. **Cloud Occlusion in Target Acquisition Windows**:
   - *Risk*: A specific day in August 2020 or 2021 may contain localized cloud clusters over Jamari.
   - *Mitigation*: Planetary Computer STAC search must query a 30-day temporal window (July 15 – August 31) and sort by `eo:cloud_cover` to pick the single cleanest granule pair.
3. **Class Imbalance in Extracted Patches**:
   - *Risk*: Deforestation may be concentrated in a small fraction of patches, leaving 90% of patches with zero change.
   - *Mitigation*: Report metrics with stratification; do not artificially discard all zero-change patches, but ensure balanced mini-batch sampling during training.

---

## 11. GO / NO-GO Recommendation

### **Recommendation: GO (With Strict Constraints)**

The feasibility investigation confirms that a bi-temporal Sentinel-2 (`20LMR`) + PRODES (2020 vs. 2021) workflow is **scientifically defensible, computationally feasible on an 8 GB MacBook Air M3, and executable via open Microsoft Planetary Computer STAC APIs without upfront account creation or cost.**

**Strict Conditions for Phase 1B (Data Acquisition)**:
1. Limit download to a small bounded sub-AOI within MGRS tile `20LMR` (Jamari, Rondônia).
2. Download only 4 native 10 m bands (B02, B03, B04, B08) for two selected clear dates (2020 and 2021).
3. Enforce a strict disk storage cap of $\le 2.0 	ext{ GB}$.
4. Do not install PyTorch or third-party bloated frameworks; use lightweight, standard geospatial tools (`pystac-client`, `rasterio`, `shapely`, `geopandas`) and TensorFlow 2.x / Keras.

---

*Report prepared independently without downloading imagery, creating environments, installing packages, or modifying existing codebase.*
