# Architecture and Scientific Decision Records (ADRs)

## ADR-001: Transition from ForestNet/EuroSAT to Bi-Temporal Sentinel-2 + PRODES

- **Status**: Accepted
- **Context**: Previous work explored static single-image classification (e.g., EuroSAT land-cover classes and ForestNet driver taxonomy). However, static classification cannot capture land-cover transitions over time and cannot rigorously detect deforestation events as they occur.
- **Decision**: Adopt a bi-temporal change detection methodology. The pipeline pairs two temporal observations ($T1$ baseline, $T2$ observation) from Sentinel-2 and uses INPE PRODES annual deforestation data as the ground truth reference.
- **Consequences**: Eliminates false equivalence between land cover classification and change detection; requires spatial co-registration and temporal pairing pipelines.

## ADR-002: Framework Mandate — TensorFlow / Keras 3 with Apple Silicon Optimization

- **Status**: Accepted
- **Context**: The execution environment is a macOS machine with an Apple Silicon M3 processor and 8 GB unified memory. The project requires a clean, robust, and reproducible modeling stack.
- **Decision**: TensorFlow / Keras is mandatory. PyTorch is strictly prohibited.
- **Consequences**: All model architectures, data loaders, custom layers, loss functions, and Grad-CAM implementations must be written using TensorFlow / Keras APIs (`tf.keras`, `tf.data`).

## ADR-003: Hardware Budgeting — Strict 8 GB RAM Architecture

- **Status**: Accepted
- **Context**: The development machine has 8 GB of unified memory. Loading full satellite scenes ($10,980 \times 10,980$ pixels $\times 12$ bands $\approx 2.8$ GB uncompressed float32 per scene) into memory would immediately cause out-of-memory (OOM) crashes and system instability.
- **Decision**:
  1. Patch size fixed to $256 \times 256$ pixels.
  2. Data ingestion and preprocessing must process raster tiles in streaming chunks.
  3. Training pipelines must use `tf.data.Dataset` with disk caching or on-the-fly generator streaming.
  4. Batch sizes restricted to $8$–$16$.
  5. In-memory arrays of entire datasets are strictly prohibited.
- **Consequences**: Ensures smooth execution on Apple Silicon M3 without exceeding memory limits.

## ADR-004: Spatial Partitioning Strategy (Eliminating Spatial Leakage)

- **Status**: Accepted
- **Context**: Random pixel- or patch-level splitting across a satellite scene results in high spatial autocorrelation between training and test sets, artificially inflating performance metrics.
- **Decision**: All dataset partitions (train, validation, test) must be split into distinct geographic blocks separated by buffer zones.
- **Consequences**: Test set performance will reflect true generalization to unseen geographic territory.

## ADR-005: Explainability Pipeline and Epistemic Limits

- **Status**: Accepted
- **Context**: Stakeholders require insight into the spatial features driving model predictions.
- **Decision**: Implement Grad-CAM targeting the penultimate convolutional feature maps of the change detection network. Document explicitly that Grad-CAM reflects model activation gradients and does not constitute ground-truth physical verification.
- **Consequences**: Provides interpretable spatial visual heatmaps while maintaining scientific rigor in documentation and reporting.

## ADR-006: Staged Execution and Zero Premature Actions in Phase 0

- **Status**: Accepted
- **Context**: Premature package installation, environment resolution, and bulk data downloading can clutter local environments and violate architectural planning.
- **Decision**: Phase 0 is strictly limited to project initialization, file structure setup, configuration templates, and comprehensive documentation. No packages installed, no virtual environments created, no models trained, no external git remotes attached, and no datasets downloaded.
- **Consequences**: Guarantees a clean, deliberate transition to Phase 1 upon user approval.
