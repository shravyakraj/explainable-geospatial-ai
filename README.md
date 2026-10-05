# Explainable AI Based Deforestation Anomaly Detection Using Geospatial Satellite Images

## Project Overview

This project implements a scientifically sound, bi-temporal change detection and explainable artificial intelligence (XAI) methodology for detecting forest-loss anomalies from Earth observation data.

Unlike conventional land-cover classification approaches that classify static, single-date image scenes, this project formulates deforestation detection as a **bi-temporal change detection task** comparing paired satellite observations taken at time point **T1** (pre-event baseline: 2020-08-08) and time point **T2** (post-event observation: 2021-08-08), verified against official **PRODES** (INPE) deforestation reference polygons.

## Key Specifications

- **Target Task**: Binary forest-loss / deforestation detection from bi-temporal Sentinel-2 observation pairs.
- **Satellite Imagery**: Sentinel-2 Level-2A Bottom-Of-Atmosphere (BOA) Surface Reflectance (B02, B03, B04, B08 at 10m resolution).
- **Ground Reference**: PRODES 2021 (INPE, Amazon Deforestation Monitoring Project).
- **Deep Learning Framework**: **TensorFlow 2.16.2 / Keras 3.15.1** (`tensorflow-metal 1.2.0` on Apple Silicon Metal GPU; PyTorch prohibited).
- **Production Architecture**: Siamese Pretrained MobileNetV3-Small with 4-band NIR spectral adapter and elementwise absolute difference fusion (978,429 parameters).
- **Spatial Dimensions**: $256 \times 256$ pixels at 10m/pixel ($2.56\text{ km} \times 2.56\text{ km} = 655.36\text{ ha}$ per patch). PRODES MMU threshold is $6.25\text{ ha}$.
- **Explainability**: Bi-temporal Gradient-weighted Class Activation Mapping (Grad-CAM) across observation dates, alongside a derived relative salience map computed from pre-normalization activations and biophysical NDVI analysis.
- **Hardware Target**: Apple MacBook Air M3 (8 GB Unified Memory, macOS). Peak memory footprint is strictly controlled ($< 1.5\text{ GB}$).
- **Web Dashboard**: Interactive Streamlit application (`app/app.py`) for live bi-temporal inference, patch inspection, and explainability visualization.

---

## Current Project Status: Phase 3 Complete

- [x] **Phase 0**: Project foundation, directory structure, and architectural decision records.
- [x] **Phase 1**: Scientific audit, study area lock (MGRS `20LMR`), and temporal baseline alignment.
- [x] **Phase 2A**: Data acquisition, rasterization, exclusion masking, patch extraction (621 patches, $256 \times 256 \times 4$), and locked spatial block splitting.
- [x] **Phase 2B**: End-to-end baseline Siamese CNN implementation, validation-based model selection, held-out test evaluation ($N=60$), Grad-CAM explainability, and error analysis.
- [x] **Phase 2C**: Model performance improvement search across custom CNN backbones (M1 retained as baseline).
- [x] **Phase 2D**: Architecture Escalation Campaign across 6 families (Pretrained Siamese, Early Fusion, Spatial Change Attention, Hybrid Temporal Fusion, and Multi-Family Ensembles), achieving **93.33% test accuracy** and **100% recall**.
- [x] **Phase 3**: Bi-Temporal Grad-CAM Explainability, immutable production contract (`configs/production_model.yaml`), and interactive Streamlit research dashboard (`app/app.py`).

---

## Performance Summary (Held-Out Test Set, $N=60$)

Evaluated on the locked virgin held-out test split ($N=60$: 26 Deforestation, 34 Intact Forest):

| Metric | Naive Majority Baseline | Phase 2B Baseline (M1) | Phase 2D Champion (`FAM-A2: MobileNetV3`) | Gain over Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | 56.67% | 85.00% (51/60) | **93.33% (56/60)** | **+8.33 pp** |
| **Deforestation Recall** | 0.00% | 80.77% (21/26) | **100.00% (26/26, 0 Missed)** | **+19.23 pp** |
| **Intact Specificity** | 100.00% | 88.24% (30/34) | **88.24% (30/34, 4 False Alarms)** | Preserved |
| **Precision** | N/A | 84.00% | **86.67%** | **+2.67 pp** |
| **F1-Score** | 0.0000 | 0.8235 | **0.9286** | **+0.1051** |
| **Balanced Accuracy** | 50.00% | 84.50% | **94.12%** | **+9.62 pp** |
| **ROC-AUC** | 0.5000 | 0.9010 | **0.9666** | **+0.0656** |
| **PR-AUC** | 0.4333 | 0.8965 | **0.9354** | **+0.0389** |
| **95% Wilson CI** | — | [73.89%, 91.90%] | **[84.07%, 97.38%]** | — |

---

## Interactive Streamlit Dashboard

The Phase 3 research dashboard provides an interactive web interface to inspect Sentinel-2 bi-temporal patch pairs, run live inference through the production model, and explore spatial Grad-CAM attribution heatmaps.

### Launching the Dashboard

1. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```
2. Start the Streamlit application:
   ```bash
   streamlit run app/app.py
   ```
3. Open `http://localhost:8501` in your web browser.

### Features
- **Demonstration Scenarios**: Curated examples of confirmed deforestation, intact rainforest canopy, and validation events.
- **Live Inference**: Instant classification with output probabilities and decision margin metrics evaluated relative to the locked $\tau^* = 0.2700$ threshold.
- **Bi-Temporal Grad-CAM**: Separate $T_1$ baseline (`2020-08-08`) and $T_2$ monitoring (`2021-08-08`) heatmaps plus a derived relative salience map computed from pre-normalization activations.
- **Multi-Spectral Biophysical Analysis**: Interactive Color-Infrared (CIR) composites, $\Delta\text{NDVI}$ anomaly mapping, and spectral reflectance bar charts.
- **Automated Scientific Explanation**: Factual decision narratives generated strictly from measured spectral response differences without causal speculation.

---

## Running the Automated Test Suite

All 33 automated unit tests cover data pipelines, model architectures, Grad-CAM invariants, and Streamlit components:

```bash
PYTHONPATH=src .venv/bin/pytest
```
