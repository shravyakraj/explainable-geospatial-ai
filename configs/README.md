# Configuration Management

This directory contains configuration files for study areas, data ingestion parameters, preprocessing parameters, model architectures, and evaluation protocols.

## Principles

1. **Declarative Experiments**: Every experiment run must be fully reproducible from a version-controlled configuration file.
2. **Decoupled Architecture**: Code in `src/geospatial_ai/` must consume parameter objects rather than hardcoded file paths or spatial bounds.
3. **Hardware Constraint Awareness**: Configuration files define memory-safe batch sizes, tile sizes (256x256), and channel selections compatible with 8 GB unified RAM.

## Configuration Schemas (Planned)

### Data Configuration (`data_config.yaml`)
- `study_area`: Target geographic bounding box (GeoJSON or EPSG:4326 coordinates)
- `temporal_windows`:
  - `t1`: Pre-event observation window (baseline dry season)
  - `t2`: Post-event observation window (target dry season)
- `bands`: Selected spectral bands (e.g., B02, B03, B04, B08, B11, B12)
- `patch_size`: `[256, 256]`
- `max_cloud_cover_percent`: Maximum allowable scene/tile cloud percentage (e.g., 10%)
- `spatial_split`: Block definitions for train, validation, and test partitions

### Model Configuration (`model_config.yaml`)
- `framework`: `tensorflow` / `keras`
- `architecture`: Bi-temporal change detection backbone (e.g., Siamese feature extractor or early-fusion CNN)
- `input_shape`: `[256, 256, 2 * num_bands]`
- `batch_size`: Sized for 8 GB RAM (e.g., 8 to 16)
- `learning_rate`: Initial optimizer learning rate
- `loss`: Binary cross-entropy / Focal loss / Dice loss (tailored for class imbalance)

### Explainability Configuration (`explainability_config.yaml`)
- `method`: `grad_cam`
- `target_layers`: Convolutional feature layers targeted for gradient extraction
- `attribution_threshold`: Post-processing thresholds for visual mask comparison
