# Phase 3: Bi-Temporal Grad-CAM Explainability & Interactive Streamlit Dashboard

**Project Title**: Explainable AI Based Deforestation Anomaly Detection Using Geospatial Satellite Images  
**Author**: Shravya  
**Date**: October 4, 2026  
**Repository**: `/Users/shravya/explainable-geospatial-ai`  
**Execution Environment**: Apple MacBook Air M3 (8 GB Unified Memory, macOS, Apple Silicon Metal GPU acceleration)  
**Deep Learning Framework**: TensorFlow 2.16.2 / Keras 3.15.1 (`tensorflow-metal 1.2.0`)  
**Web Framework**: Streamlit 1.36.0  

---

## 1. Phase 3 Objective & Scope

The objective of Phase 3 is to translate the empirically validated, locked **Phase 2D Champion Model** into an operational, explainable, scientific research application:
1. **Model Freezing & Immutability**: Package the 93.33% accuracy, 100% recall `MobileNetV3-Small Siamese` model into an immutable inference contract (`configs/production_model.yaml`).
2. **Bi-Temporal Grad-CAM Explainability**: Formulate, validate, and test gradient-weighted class activation mapping (Grad-CAM) across both baseline ($T_1$) and monitoring ($T_2$) Sentinel-2 scenes, alongside a derived relative salience map.
3. **Evidence-Based Scientific Text Generation**: Generate factual explanations strictly tied to measured spectral variations ($\Delta\text{NDVI}$) and model activations without fabricating causal narratives (e.g. logging, road construction, fire).
4. **Interactive Research Dashboard**: Deliver a clean, responsive, academic Streamlit application (`app/app.py`) allowing interactive patch selection, metadata inspection, bi-temporal visual comparisons, and attribution heatmaps.

---

## 2. Production Model Architecture & Provenance

The production model is strictly locked to the **Phase 2D Champion**:
- **Model Identifier**: `FAM-A2: Pretrained_MobileNetV3_Frozen`
- **Model Version**: `2.4.0-phase2d`
- **Checkpoint Location**: `models/champion_phase2d_mobilenetv3.keras`
- **Architecture Family**: Siamese Convolutional Neural Network with an Adapted Pretrained ImageNet Backbone
- **Total Parameters**: 978,429 (38,157 trainable, 940,272 non-trainable frozen weights)
- **Spectral Adapter**: Trainable $1\times 1$ Conv2D mapping 4 Sentinel-2 bands $[B02, B03, B04, B08]$ to 3 RGB channels with physical NIR initialization:
  $$R_{\text{in}} = 1.0 \cdot B_{04} + 0.3 \cdot B_{08}, \quad G_{\text{in}} = 1.0 \cdot B_{03} + 0.3 \cdot B_{08}, \quad B_{\text{in}} = 1.0 \cdot B_{02}$$
- **Fusion Layer**: Elementwise absolute feature difference $|f(T_1) - f(T_2)|$ in 576-dimensional latent space.
- **Classification Head**: BatchNormalization $\to$ Dense(64, ReLU) $\to$ Dropout(0.4) $\to$ Dense(1, Sigmoid).
- **Decision Threshold**: $\tau^* = 0.2700$ (optimized strictly on validation data).

### Locked Empirical Performance (Held-Out Test Set, $N=60$):
- **Test Accuracy**: **93.33% (56/60 patches correct)** [95% Wilson CI: 84.07% – 97.38%]
- **Deforestation Recall**: **100.00% (26/26 events detected, ZERO false negatives)**
- **Intact Forest Specificity**: **88.24% (30/34 intact patches, 4 false alarms)**
- **ROC-AUC**: **0.9666** | **PR-AUC**: **0.9354** | **F1-Score**: **0.9286**

---

## 3. Inference Contract

The inference API (`src/geospatial_ai/inference.py`) defines a structured, type-safe contract:

### 3.1 Input Specifications
- **Format**: Pair of NumPy float32 arrays $(T_1, T_2)$.
- **Spatial Dimensions**: $(256, 256, 4)$ per patch ($2.56\text{ km} \times 2.56\text{ km} = 655.36\text{ ha}$ at 10m spatial resolution).
- **Band Ordering**: `[B02 (Blue), B03 (Green), B04 (Red), B08 (NIR)]`.
- **Normalization**: Surface reflectance divided by $10,000.0$, bounded in $[0.0, 1.0]$.
- **Temporal Alignment**: $T_1$ = earlier acquisition (`2020-08-08`), $T_2$ = later acquisition (`2021-08-08`).

### 3.2 Output Contract
Calling `ProductionDeforestationClassifier.predict(t1, t2)` returns:
```python
{
    "prediction": "Deforestation" | "Intact Forest",
    "predicted_class": 1 | 0,
    "probability": float,          # raw predicted sigmoid output [0.0, 1.0]
    "decision_margin": float,      # normalized distance from decision threshold [0.0, 1.0]
    "threshold": 0.2700,           # locked decision threshold
    "model_name": "Pretrained_MobileNetV3_Frozen",
    "model_version": "2.4.0-phase2d",
    "input_shape": [256, 256, 4],
    "bands": ["B02", "B03", "B04", "B08"]
}
```

*Note on Decision Margin*: The `decision_margin` is a derived normalized distance from the classification threshold ($|p - \tau| / \text{norm}$) indicating proximity to the decision boundary. It is **not** a calibrated probability of prediction correctness, nor is the raw sigmoid output a certainty metric.

---

## 4. Bi-Temporal Grad-CAM Methodology

Grad-CAM (Selvaraju et al., 2017) is formulated to preserve the temporal duality of the Siamese network while avoiding post-normalization subtraction artifacts:

### 4.1 Mathematical Formulation
1. **Target Feature Tensor**: Extracted at the last convolutional activation layer of the shared MobileNetV3 backbone (`target_relu`), yielding spatial feature map $A \in \mathbb{R}^{8 \times 8 \times 576}$.
2. **Gradient Computation via `tf.GradientTape`**:
   The gradient of the output probability $y$ with respect to each temporal feature activation map is computed:
   $$G_1 = \frac{\partial y}{\partial A(T_1)}, \quad G_2 = \frac{\partial y}{\partial A(T_2)}$$
3. **Channel Importance Weights**:
   Gradients are pooled spatially across the $8\times 8$ grid:
   $$\alpha_{k}^{(t)} = \frac{1}{64} \sum_{i=1}^8 \sum_{j=1}^8 G_{i, j, k}^{(t)}, \quad t \in \{1, 2\}$$
4. **Pre-Normalization Raw Activations**:
   $$\text{CAM}_{\text{raw}}^{(t)} = \text{ReLU}\left( \sum_{k=1}^{576} \alpha_k^{(t)} A_k(T_t) \right)$$
5. **Per-Scene Normalization**:
   For individual scene visualization, each raw map is safely normalized and bilinearly upsampled to native $(256, 256)$ resolution:
   $$\text{CAM}_{\text{norm}}^{(t)} = \frac{\text{CAM}_{\text{raw}}^{(t)}}{\max(\text{CAM}_{\text{raw}}^{(t)}) + 10^{-8}}$$

### 4.2 Derived Relative Salience Map (Pre-Normalization Difference)
To answer *where did positive model activation increase between dates?*, the temporal comparison is computed directly from **pre-normalization CAM activations** before any scene-level scaling:
$$\Delta\text{CAM}_{\text{raw}} = \text{ReLU}\left( \text{CAM}_{\text{raw}}^{(2)} - \text{CAM}_{\text{raw}}^{(1)} \right)$$
Only then is the resulting difference map normalized to $[0, 1]$ and bilinearly upsampled to $(256, 256)$:
$$\Delta\text{CAM}_{\text{norm}} = \frac{\Delta\text{CAM}_{\text{raw}}}{\max(\Delta\text{CAM}_{\text{raw}}) + 10^{-8}}$$

*Scientific Interpretation Note*: This map is explicitly defined as a **Derived Relative Salience Map**. It highlights spatial zones where positive model activation increased from $T_1$ to $T_2$. It is a qualitative spatial attribution tool, **not** an absolute decision weight, calibrated posterior probability, or physical deforestation boundary.

### 4.3 Separation from Direct Spectral Products
The system strictly distinguishes between:
1. **Model Activations**: $T_1$ Grad-CAM, $T_2$ Grad-CAM (what the neural network utilized).
2. **Derived Relative Salience**: Pre-normalization directional shift in model attention.
3. **Direct Biophysical Spectral Maps**: $\Delta\text{NDVI} = \text{NDVI}(T_2) - \text{NDVI}(T_1)$ and RGB difference $|T_2 - T_1|$ (objective physical measurements independent of model weights).

---

## 5. Streamlit Application Architecture

The application is structured cleanly under `app/`:

```text
app/
├── app.py                      # Main entrypoint and UI orchestration
├── components/
│   ├── header.py               # Title, badges, and project synopsis
│   ├── patch_selector.py       # Sidebar scenario selector and geographic metadata
│   ├── prediction_card.py      # Decision banner, probability gauge, threshold indicator
│   └── explainability_view.py  # Tabbed views for Grad-CAM, CIR, NDVI, and 3x3 grid
├── inference/
│   └── predictor.py            # Streamlit-cached singleton wrapper for classifier
├── visualization/
│   └── image_renderers.py      # PIL and Matplotlib composite generators
└── utils/
    └── data_loader.py          # Cached patch array and manifest reader
```

### 5.1 Optimization & Memory Management
- **Model Caching**: `@st.cache_resource` ensures the 978k parameter model is loaded once into Apple Silicon Metal memory and reused across all UI interactions.
- **Data Caching**: `@st.cache_data` caches patch files and manifest records.
- **Interactive Performance**: Total bi-temporal inference, Grad-CAM generation, and overlay rendering execute interactively without requiring external server calls.

---

## 6. How to Launch and Test the Application

### 6.1 Prerequisites
Activate the project virtual environment:
```bash
cd /Users/shravya/explainable-geospatial-ai
source .venv/bin/activate
```

### 6.2 Launch Streamlit App
Run:
```bash
streamlit run app/app.py
```
Open your browser at `http://localhost:8501`.

### 6.3 Run the Automated Test Suite
Execute the full test suite (33 unit tests covering modeling, dataset, Grad-CAM properties, and app components):
```bash
PYTHONPATH=src:. pytest tests/
```

---

## 7. Example Workflow

1. **Launch**: Start `streamlit run app/app.py`.
2. **Select Scenario**: In the sidebar, select a curated scenario (e.g. `[Test] patch_r19_c10 — Deforestation`).
3. **Inspect Geospatial Metadata**: Review patch UTM Zone 20S coordinates, grid location, patch surface area ($655.36\text{ ha}$ per patch, $2.56\text{ km} \times 2.56\text{ km}$), PRODES reference MMU ($6.25\text{ ha}$), and acquisition dates (`2020-08-08` and `2021-08-08`).
4. **Examine Prediction**: Review the predicted class badge, output probability, and decision margin.
5. **Inspect Bi-Temporal Grad-CAM**: In the *Bi-Temporal Grad-CAM* tab, observe how the $ heatmap highlights spatial features in the monitoring scene, while the *Derived Relative Salience Map* displays the model-highlighted region of increased relative salience ($\text{ReLU}(\text{CAM}_{T2}^{\text{raw}} - \text{CAM}_{T1}^{\text{raw}})$). Note that Grad-CAM salience reflects qualitative neural attribution and is not physically or causally equivalent to vegetation-index delta maps.
6. **Verify Biophysical Properties**: Switch to the *Multi-Spectral & NDVI Analysis* tab to confirm the drop in Near-Infrared reflectance (B08) and the corresponding negative $\Delta\text{NDVI}$ shift.
7. **Read Scientific Explanation**: Review the factual automated text summary synthesized from measured spectral data.

---

## 8. Scientific Limitations & Ethical Disclaimers

1. **Patch-Level Classification Scope**: The model performs binary patch-level anomaly detection ($256\times 256$ pixels, $2.56\text{ km} \times 2.56\text{ km} = 655.36\text{ ha}$). It is not a pixel-level boundary segmentation model.
2. **Grad-CAM Attribution Limits**: Grad-CAM heatmaps highlight regions influencing the model's global feature representation; they should not be interpreted as legal deforestation boundaries or cadastre survey lines.
3. **Decision Score Calibration**: The output probability is a neural network decision score; it should not be interpreted as a frequentist real-world probability.
4. **Geographic Generalization**: The training dataset is derived from Sentinel-2 tile `20LMR` in the Brazilian Amazon. Deployment across distinct biomes (e.g. Cerrado, Chaco) requires domain adaptation.
5. **False Positive Management**: The model maintains an 88.24% specificity (4 false alarms out of 34 intact test patches), primarily triggered by seasonal canopy phenology variations in dense rain forest. Ground verification remains mandatory for operational enforcement.
