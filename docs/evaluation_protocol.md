# Evaluation Protocol and Safeguards

## 1. Spatial Autocorrelation and Data Leakage Prevention

In geospatial machine learning, spatial autocorrelation violates the standard independent and identically distributed (i.i.d.) assumption. Adjacent satellite image patches share nearly identical atmospheric, solar, soil, and vegetation characteristics.

### The Leakage Pitfall
If satellite scenes are sliced into $256 \times 256$ patches and randomly assigned to training and test sets, patches immediately adjacent to each other will appear in both sets. This causes severe **spatial data leakage**, resulting in artificially inflated test metrics that fail catastrophically on new regions.

### Mandatory Safeguards
1. **Geographic Block Partitioning**: The study area will be divided into geographically contiguous regional blocks (e.g., Western block for training, Central block for validation, Eastern block for testing).
2. **Buffer Zones**: Spatial buffer margins (at least 2–5 km) will be established between training, validation, and test blocks to eliminate boundary patch correlation.
3. **Strict Model Selection Isolation**: Hyperparameter tuning, architecture selection, and early stopping checkpoints will be guided exclusively by the validation set. The test partition will remain unexamined until final model benchmarking.

## 2. Evaluation Metric Suite

Deforestation is a sparse, anomalous event across regional landscapes. Reporting only overall accuracy produces a deceptive evaluation:

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

If 96% of the landscape remains unchanged ($TN$), a naive baseline that predicts "no change" everywhere achieves 96% accuracy while identifying zero deforestation.

### Required Metrics
To provide rigorous and honest evaluation, the pipeline will compute:
- **Positive-Class Precision**: $\frac{TP}{TP + FP}$ — Proportion of predicted deforestation events that are genuine.
- **Positive-Class Recall (Sensitivity)**: $\frac{TP}{TP + FN}$ — Proportion of actual deforestation events correctly detected.
- **F1-Score**: Harmonic mean of Precision and Recall on the change class.
- **Intersection over Union (IoU / Jaccard Index)**: $\frac{TP}{TP + FP + FN}$ — Area of overlap between predicted change and PRODES reference mask.
- **Confusion Matrix**: Full breakdown of $TP$, $FP$, $TN$, and $FN$ at both pixel and patch levels.
- **Overall Accuracy**: Reported alongside class-level metrics for completeness, not as a standalone claim.

## 3. Performance Targets and Realistic Expectations

- **Aspirational Benchmark**: An accuracy target of 90–95% on balanced evaluation subsets or an F1-score of 0.80+ on sparse change detection represents a research aspiration, not a guaranteed initial outcome.
- **No Fabricated Metrics**: No metric values will be asserted or reported without empirical execution on verified test partitions.
- **Limitation Reporting**: Every experimental report must explicitly document failure modes (e.g., false positives from seasonal agricultural harvesting, cloud shadow artifacts, or sub-pixel edge mismatches).

## 4. Explainability (Grad-CAM) Validation Protocol

### Epistemic Boundaries
- Grad-CAM highlights regions in the feature maps that exerted the largest positive gradient influence on the target class prediction.
- **Grad-CAM does not prove deforestation**: It reveals what spatial patterns the neural network attended to; it does not replace scientific reference data.

### Auditing Procedure
1. Compute Grad-CAM heatmaps on test patches where the model predicts deforestation.
2. Overlay heatmaps onto both $T1$ and $T2$ visual representations (RGB and NIR false-color).
3. Evaluate spatial alignment between the high-attribution heatmap regions and the actual PRODES deforestation boundaries.
4. Categorize attributions into:
   - **True Attributions**: Model attends directly to cleared forest patches.
   - **Spurious Attributions**: Model attends to cloud edges, river borders, or road artifacts.
