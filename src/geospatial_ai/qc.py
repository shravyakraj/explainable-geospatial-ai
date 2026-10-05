"""Visual quality control and diagnostic figure generation."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio

from geospatial_ai.geo import StudyGrid, LOCKED_GRID

logger = logging.getLogger(__name__)


def generate_spatial_split_layout(records: list[dict], output_png: Path, grid: StudyGrid = LOCKED_GRID) -> Path:
    """Render the 27x23 patch grid with spatial split assignments and deforestation indicators."""
    output_png.parent.mkdir(parents=True, exist_ok=True)
    grid_matrix = np.zeros((grid.rows_patches, grid.cols_patches), dtype=int)
    loss_matrix = np.zeros((grid.rows_patches, grid.cols_patches), dtype=int)

    # 1: Train, 2: Buffer, 3: Val, 4: Test
    split_map = {"train": 1, "buffer": 2, "val": 3, "test": 4}

    for r in records:
        row_idx = r["row"] - 1
        col_idx = r["col"] - 1
        grid_matrix[row_idx, col_idx] = split_map.get(r["split"], 0)
        if r["label"] == 1:
            loss_matrix[row_idx, col_idx] = 1

    fig, ax = plt.subplots(figsize=(14, 8), dpi=150)
    cmap = plt.cm.colors.ListedColormap(["#2ca02c", "#7f7f7f", "#1f77b4", "#ff7f0e"])
    im = ax.imshow(grid_matrix, cmap=cmap, vmin=1, vmax=4)

    # Overlay stars on positive deforestation patches
    pos_rows, pos_cols = np.where(loss_matrix == 1)
    ax.scatter(pos_cols, pos_rows, marker="*", color="red", s=70, label="Deforestation >= 6.25 ha (Label=1)")

    # Formatting
    ax.set_xticks(range(grid.cols_patches))
    ax.set_xticklabels([f"C{c+1}" for c in range(grid.cols_patches)], fontsize=8, rotation=90)
    ax.set_yticks(range(grid.rows_patches))
    ax.set_yticklabels([f"R{r+1}" for r in range(grid.rows_patches)], fontsize=8)
    ax.set_title("Locked Spatial Partitioning & Deforestation Distribution (Tile 20LMR, 27x23 Patches)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Patch Column (2.56 km per column)")
    ax.set_ylabel("Patch Row (2.56 km per row)")

    # Colorbar legend
    cbar = fig.colorbar(im, ticks=[1.375, 2.125, 2.875, 3.625], ax=ax, fraction=0.03, pad=0.04)
    cbar.ax.set_yticklabels(["Train (Cols 1-15)", "Buffer 1 (Col 16)", "Val (Cols 17-21)", "Buffer 2 / Test (Cols 22-27)"])

    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()
    logger.info("Saved spatial split layout to %s", output_png)
    return output_png


def generate_patch_visual_qc(
    patch_id: str,
    patches_dir: Path,
    output_png: Path,
    title_suffix: str = "",
) -> Path:
    """Render T1 RGB, T2 RGB, PRODES mask, and Exclusion mask for a single patch."""
    output_png.parent.mkdir(parents=True, exist_ok=True)
    npz_path = patches_dir / f"{patch_id}.npz"
    if not npz_path.exists():
        logger.warning("Patch not found: %s", npz_path)
        return output_png

    data = np.load(npz_path)
    t1 = data["t1"]  # (256, 256, 4) -> B2, B3, B4, B8
    t2 = data["t2"]
    mask = data["mask"]
    exclude = data["exclude"]
    label = int(data["label"])

    # Make RGB representations (B4=red, B3=green, B2=blue)
    # Clip reflectance DN (0-3000 typical for land)
    def to_rgb(arr):
        rgb = np.stack([arr[:, :, 2], arr[:, :, 1], arr[:, :, 0]], axis=-1).astype(float)
        rgb = np.clip(rgb / 2500.0, 0.0, 1.0)
        return rgb

    rgb1 = to_rgb(t1)
    rgb2 = to_rgb(t2)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), dpi=150)

    axes[0].imshow(rgb1)
    axes[0].set_title(f"T1 RGB (2020-08-08)\n{patch_id}")
    axes[0].axis("off")

    axes[1].imshow(rgb2)
    axes[1].set_title(f"T2 RGB (2021-08-08)\nLabel: {label}")
    axes[1].axis("off")

    axes[2].imshow(mask, cmap="Reds", vmin=0, vmax=1)
    axes[2].set_title("PRODES 2021 Mask\n(1=Deforestation)")
    axes[2].axis("off")

    axes[3].imshow(exclude, cmap="Blues", vmin=0, vmax=1)
    axes[3].set_title("Unified Exclusion Mask\n(1=Excluded/Prior/Water)")
    axes[3].axis("off")

    plt.suptitle(f"Quality Control Diagnostic: {patch_id} {title_suffix}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()
    logger.info("Saved patch QC to %s", output_png)
    return output_png
