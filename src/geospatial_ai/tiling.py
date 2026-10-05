"""Patch extraction, spatial splitting, and labeling."""

import json
import logging
from pathlib import Path

import numpy as np
import rasterio

from geospatial_ai.geo import StudyGrid, LOCKED_GRID

logger = logging.getLogger(__name__)


def get_spatial_split(col: int) -> str:
    """Assign locked spatial split based on 1-indexed column number."""
    if 1 <= col <= 15:
        return "train"
    elif col == 16:
        return "buffer"
    elif 17 <= col <= 21:
        return "val"
    elif col == 22:
        return "buffer"
    elif 23 <= col <= 27:
        return "test"
    else:
        raise ValueError(f"Column index {col} outside locked range [1, 27]")


def extract_and_label_patches(
    t1_paths: dict[str, Path],
    t2_paths: dict[str, Path],
    prodes_mask_path: Path,
    exclusion_mask_path: Path,
    output_dir: Path,
    grid: StudyGrid = LOCKED_GRID,
) -> list[dict]:
    """Extract 621 patches, assign ground-truth labels and spatial splits."""
    output_dir.mkdir(parents=True, exist_ok=True)
    patches_dir = output_dir / "patches"
    patches_dir.mkdir(parents=True, exist_ok=True)

    # Load 4-band stacks into memory as uint16
    band_order = ["B02", "B03", "B04", "B08"]
    t1_stack = np.zeros((grid.height_pixels, grid.width_pixels, 4), dtype=np.uint16)
    t2_stack = np.zeros((grid.height_pixels, grid.width_pixels, 4), dtype=np.uint16)

    for i, b in enumerate(band_order):
        with rasterio.open(t1_paths[b]) as src:
            t1_stack[:, :, i] = src.read(1)
        with rasterio.open(t2_paths[b]) as src:
            t2_stack[:, :, i] = src.read(1)

    with rasterio.open(prodes_mask_path) as src:
        prodes_mask = src.read(1).astype(np.uint8)

    with rasterio.open(exclusion_mask_path) as src:
        exclusion_mask = src.read(1).astype(np.uint8)

    manifest_records = []
    patch_size = grid.patch_size
    total_pixels_per_patch = patch_size * patch_size  # 65,536

    for r in range(grid.rows_patches):
        row_num = r + 1
        py0 = r * patch_size
        py1 = py0 + patch_size

        for c in range(grid.cols_patches):
            col_num = c + 1
            px0 = c * patch_size
            px1 = px0 + patch_size

            patch_id = f"patch_r{row_num:02d}_c{col_num:02d}"
            split = get_spatial_split(col_num)

            # Slice patch arrays
            p_t1 = t1_stack[py0:py1, px0:px1, :]
            p_t2 = t2_stack[py0:py1, px0:px1, :]
            p_prodes = prodes_mask[py0:py1, px0:px1]
            p_exclude = exclusion_mask[py0:py1, px0:px1]

            # Projected bounds for patch (EPSG:32720)
            p_minx = grid.origin_x + px0 * grid.pixel_size
            p_maxx = grid.origin_x + px1 * grid.pixel_size
            p_maxy = grid.origin_y - py0 * grid.pixel_size
            p_miny = grid.origin_y - py1 * grid.pixel_size

            # Pixel counts & statistics
            p_excluded_px = int(np.count_nonzero(p_exclude == 1))
            p_valid_px = total_pixels_per_patch - p_excluded_px
            valid_frac = float(p_valid_px / total_pixels_per_patch)

            # Unmasked PRODES loss
            p_clean_loss = (p_prodes == 1) & (p_exclude == 0)
            p_loss_px = int(np.count_nonzero(p_clean_loss))
            p_loss_ha = float(p_loss_px * (grid.pixel_size ** 2) / 10000.0)
            loss_frac = float(p_loss_px / total_pixels_per_patch)

            # Ground truth labeling logic
            if split == "buffer":
                label = -1
                exclusion_reason = "buffer_column_discarded"
            elif valid_frac < 0.50:
                label = -1
                exclusion_reason = "excessive_exclusion_over_50pct"
            elif p_loss_ha >= 6.25 or loss_frac >= 0.03:
                label = 1
                exclusion_reason = ""
            elif p_loss_px < 10 and valid_frac >= 0.70:
                label = 0
                exclusion_reason = ""
            else:
                label = -1
                exclusion_reason = "marginal_prodes_under_6.25ha"

            # Save compressed patch file
            patch_file = patches_dir / f"{patch_id}.npz"
            np.savez_compressed(
                patch_file,
                t1=p_t1,
                t2=p_t2,
                mask=p_prodes,
                exclude=p_exclude,
                label=label,
            )

            record = {
                "patch_id": patch_id,
                "row": row_num,
                "col": col_num,
                "split": split,
                "label": label,
                "exclusion_reason": exclusion_reason,
                "prodes_positive_pixels": p_loss_px,
                "prodes_positive_ha": round(p_loss_ha, 4),
                "prodes_fraction": round(loss_frac, 6),
                "excluded_pixels": p_excluded_px,
                "valid_pixels": p_valid_px,
                "valid_fraction": round(valid_frac, 4),
                "patch_bounds_epsg32720": [round(p_minx, 2), round(p_miny, 2), round(p_maxx, 2), round(p_maxy, 2)],
                "patch_file": str(patch_file.relative_to(output_dir.parent.parent)),
            }
            manifest_records.append(record)

    logger.info("Extracted %d patches to %s", len(manifest_records), patches_dir)
    return manifest_records
