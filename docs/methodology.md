# Methodology Specification

## 1. Problem Formulation: Bi-Temporal Change Detection

Deforestation is inherently a dynamic process of land-use transition over time. Conventional static land-cover classification models (such as single-date scene classification on EuroSAT or ForestNet) categorize the dominant surface type at a single snapshot. They do not observe change and cannot distinguish between long-standing agricultural clearings and recent deforestation events.

This project formulates deforestation detection as **binary change detection on paired satellite observations**:

$$\Delta Y = f(X_{T1}, X_{T2})$$

where:
- $X_{T1} \in \mathbb{R}^{H \times W \times C}$: Co-registered multispectral satellite observation at baseline time $T1$ (pre-event forest state).
- $X_{T2} \in \mathbb{R}^{H \times W \times C}$: Co-registered multispectral satellite observation at observation time $T2$ (post-event state).
- $\Delta Y \in \{0, 1\}^{H \times W}$: Binary forest-loss label indicating where intact forest transitioned to non-forest between $T1$ and $T2$.

## 2. Planned 10-Step Workflow

The end-to-end research and engineering pipeline consists of the following 10 steps. **Note: None of these steps have been executed in Phase 0; this document establishes the protocol to be followed in subsequent phases.**

### Step 1: Study Area and Temporal Window Selection
- Identify a bounded Area of Interest (AOI) in the Brazilian Legal Amazon (e.g., hotspot regions in Rondônia, Pará, or Mato Grosso) exhibiting documented, verified deforestation events.
- Select two temporal windows, $T1$ and $T2$, matched to the regional dry season (typically July–September) to minimize cloud cover and phenological variance.

### Step 2: Sentinel-2 Observation Acquisition
- Query Sentinel-2 Level-2A (Bottom-of-Atmosphere surface reflectance) granules covering the AOI for both $T1$ and $T2$.
- Select spectral bands relevant to vegetation and soil contrast:
  - 10 m resolution: Blue (B02), Green (B03), Red (B04), Near-Infrared (B08).
  - 20 m resolution (resampled to 10 m): Short-Wave Infrared (B11, B12) and Red Edge.

### Step 3: PRODES Reference Data Acquisition & Inspection
- Ingest official PRODES (INPE) deforestation monitoring data covering the target AOI and temporal span.
- Inspect polygon geometries, attribution years, and exclusion masks (hydrography, unmonitored clouds, previous deforestation prior to $T1$).

### Step 4: Temporal Alignment, Spatial Registration, Cloud Masking, & Label Construction
- Verify coordinate reference systems (CRS) and ensure identical georeferencing between $T1$ and $T2$ rasters (sub-pixel alignment).
- Apply cloud and shadow masking using Sentinel-2 Scene Classification Layer (SCL) or QA bands.
- Rasterize PRODES deforestation vectors onto the exact 10 m pixel grid of the Sentinel-2 imagery to produce clean binary ground reference masks.

### Step 5: Patch Extraction and Tiling (256 × 256)
- Tile co-registered imagery and reference masks into non-overlapping or controlled-stride $256 \times 256$ pixel patches.
- Filter out patches dominated by unmasked clouds, sensor nodata, or invalid reference areas.
- Export standardized patch pairs with metadata indexing spatial coordinates, acquisition timestamps, and valid pixel counts.

### Step 6: Spatial Partitioning (Leakage Prevention)
- Partition the dataset into geographically separated training, validation, and test splits (spatial block holdout).
- Enforce buffer zones between spatial blocks to eliminate spatial autocorrelation and prevent patch leakage.

### Step 7: TensorFlow/Keras Baseline Model Development
- Construct a deep neural network baseline using TensorFlow / Keras (e.g., Siamese feature extractor with difference head, or concatenated multi-channel convolutional architecture).
- Design input pipelines using `tf.data.Dataset` with streaming and prefetching to operate strictly within the 8 GB unified memory constraint of the Apple Silicon M3.

### Step 8: Comprehensive Evaluation Protocol
- Evaluate predictions on the isolated spatial test set using:
  - Pixel-level and patch-level Accuracy.
  - Precision, Recall, and F1-Score on the positive (forest-loss) class.
  - Intersection-over-Union (IoU / Jaccard Index) for change masks.
  - Confusion matrices and per-class classification reports.

### Step 9: Explainability via Grad-CAM
- Implement Gradient-weighted Class Activation Mapping (Grad-CAM) to inspect the spatial activation maps of the final convolutional layers.
- Generate visual heatmaps demonstrating which regions within the bi-temporal input pair drove the network's deforestation classification.
- Qualitatively and quantitatively audit heatmaps against PRODES boundaries (verifying feature attribution without mistaking Grad-CAM for ground truth).

### Step 10: Interactive Streamlit Interface
- Once the model and explainability pipeline are empirically validated, build a Streamlit web application.
- The interface will allow users to inspect bi-temporal image pairs ($T1$ and $T2$), view model predictions, toggle the PRODES ground reference, and overlay Grad-CAM attribution heatmaps.

## 3. Methodological Integrity Disclaimers

1. **No Retrospective Claim**: This document establishes the planned methodology. No data has been downloaded, and no model has been trained.
2. **Epistemic Boundaries of XAI**: Grad-CAM visualizes where network gradients are concentrated for a given output neuron. It demonstrates what internal representations the model utilized; it does not constitute physical proof of deforestation.
3. **Class Imbalance Reality**: In real-world remote sensing, deforestation is a sparse event. A model predicting "no change" can achieve >95% accuracy while failing completely. Evaluation must prioritize Precision, Recall, and IoU on the positive class.
