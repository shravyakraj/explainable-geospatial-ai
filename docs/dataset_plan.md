# Dataset Plan: Bi-Temporal Satellite Imagery and Ground Reference

## 1. Rejection of Static Land-Cover Classification Datasets

A central scientific requirement of this project is to model **temporal forest loss**, not static land cover.

- **EuroSAT**: A single-date dataset containing 27,000 $64 \times 64$ Sentinel-2 image patches categorized into 10 static land-cover classes (e.g., Forest, Highway, Industrial). EuroSAT contains no temporal dimension, no paired observations, and no deforestation labels. It must **not** be used as ground truth for deforestation detection.
- **ForestNet**: While containing annotated deforestation drivers, ForestNet was constructed as a single-date post-event driver classification benchmark. 

Therefore, this project transitions to an authentic **bi-temporal change detection pair** methodology.

## 2. Satellite Sensor: Sentinel-2 Multispectral Instrument (MSI)

- **Source**: European Space Agency (ESA) Copernicus Program.
- **Product Level**: Level-2A (L2A) — Bottom of Atmosphere (BOA) Surface Reflectance with geometric and atmospheric corrections pre-applied.
- **Spatial Resolution**:
  - 10 m: Blue (B02), Green (B03), Red (B04), Near-Infrared (B08).
  - 20 m: Red Edge (B05, B06, B07, B8A) and Short-Wave Infrared (B11, B12).
- **Temporal Pairing**:
  - $T1$: Baseline acquisition during dry season (intact forest canopy state).
  - $T2$: Target observation during subsequent dry season (post-clearing state).
- **Cloud Masking**:
  - Sentinel-2 Scene Classification Layer (SCL) provides per-pixel classifications for cloud, cirrus, and cloud shadows. Pixels with cloud/shadow flags will be masked or excluded.

## 3. Reference Data: PRODES (INPE)

- **Source**: Brazilian National Institute for Space Research (INPE) — Project for Monitoring Deforestation in the Brazilian Amazon.
- **Nature of Data**: Annual vector polygons identifying clear-cut deforestation (suppression of primary forest) in the Legal Amazon.
- **Minimum Mapping Unit (MMU)**: 6.25 hectares (approx. 625 Sentinel-2 pixels at 10 m resolution).
- **Label Generation**:
  - PRODES polygons attributed to the interval between $T1$ and $T2$ will be extracted.
  - Polygons will be rasterized onto the exact coordinate grid, CRS, and pixel dimensions of the Sentinel-2 image tiles.
  - Previous deforestation (occurring prior to $T1$) and non-forest/water bodies will be masked out so the model focuses strictly on new change during the observation window.

## 4. Planned Study Area Selection Criteria

The candidate study area will be selected during Phase 1 according to the following strict criteria:
1. **Documented Deforestation Dynamics**: High-confidence annual PRODES activity within the target year.
2. **Minimal Cloud Interference**: Consistent dry-season acquisition dates with scene-level cloud cover < 10%.
3. **Data Volume Feasibility**: Contained within 1–2 Sentinel-2 MGRS tiles to respect the 8 GB RAM and local storage constraints of the MacBook Air M3.
4. **Candidate Regions**: Hotspots in the state of Rondônia (e.g., Porto Velho / Jamari) or Pará (e.g., Novo Progresso / Altamira).

## 5. Potential Acquisition Mechanisms Under Investigation

In Phase 1, acquisition options will be evaluated for accessibility, rate limits, licensing, and storage:
- **Microsoft Planetary Computer STAC API**: Provides open, authenticated access to cloud-optimized GeoTIFFs (COGs) of Sentinel-2 L2A without downloading entire scenes.
- **Copernicus Data Space Ecosystem (CDSE)**: Official ESA distribution platform with open API access.
- **Google Earth Engine (GEE) Export**: Cloud-filtered composite export clipped to exact study bounds.

## 6. Preprocessing and Patch Generation Plan

1. **Tile Size**: $256 \times 256$ pixels (approx. $2.56 \times 2.56$ km at 10 m resolution).
2. **Channel Stacking**:
   - Option A: Concatenated bi-temporal tensor $[256, 256, 2 \times C]$.
   - Option B: Siamese pair $[256, 256, C]$ for $T1$ and $[256, 256, C]$ for $T2$.
3. **Normalization**: Per-band reflectance scaling (e.g., dividing raw DN by 10,000 to map surface reflectance to $[0, 1]$).
4. **Data Integrity**: Preprocessed patches stored in efficient array formats (e.g., compressed NumPy `.npz` or TFRecords), maintaining spatial split boundaries.

## 7. Phase 0 Status

- **Zero Datasets Downloaded**: No satellite imagery or PRODES vector files have been acquired in this phase.
- **Verification Required**: Formal verification of API access, spatial bounds, and storage quotas must occur in Phase 1 before initiating data retrieval.
