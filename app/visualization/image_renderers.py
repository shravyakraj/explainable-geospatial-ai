"""Image and figure rendering components for Streamlit explainability dashboard."""

from typing import Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import scipy.ndimage


def array_to_pil_image(arr: np.ndarray) -> Image.Image:
    """Convert float [0, 1] RGB array of shape (H, W, 3) to PIL Image."""
    arr_clipped = np.clip(arr, 0.0, 1.0)
    arr_uint8 = (arr_clipped * 255.0).astype(np.uint8)
    return Image.fromarray(arr_uint8)


def create_highlighted_change_image(
    rgb_image: np.ndarray,
    change_salience: np.ndarray,
    threshold: float = 0.35,
) -> Image.Image:
    """Create satellite image with a heat/emphasis map overlay and contour around strongest change evidence."""
    rgb = np.clip(rgb_image, 0.0, 1.0).copy()
    sal = np.clip(change_salience, 0.0, 1.0)

    mask = sal >= threshold
    if np.any(mask):
        # Normalized relative intensity above threshold
        weights = np.clip((sal - threshold) / (1.0 - threshold + 1e-6), 0.0, 1.0)

        # Heat emphasis transition:
        # Soft warm blush at boundary -> rich coral/crimson at peak attribution
        c_low = np.array([0.98, 0.58, 0.56], dtype=np.float32)
        c_high = np.array([0.91, 0.16, 0.24], dtype=np.float32)
        tint_color = (1.0 - weights[..., None]) * c_low + weights[..., None] * c_high

        # Continuous alpha: smoothly transitions from 0.35 above threshold to 0.70 at peak
        alpha = (0.35 + 0.35 * (weights ** 0.65))[..., None]

        # Blend on top of original imagery preserving underlying satellite texture
        rgb = np.where(mask[..., None], (1.0 - alpha) * rgb + alpha * tint_color, rgb)

        # Soft contour outlining the cohesive peak evidence region
        contour_sal_thresh = max(0.50, threshold + 0.10) if np.any(sal >= 0.50) else threshold
        peak_mask = sal >= contour_sal_thresh
        if np.any(peak_mask):
            closed = scipy.ndimage.binary_closing(peak_mask, iterations=2)
            dilated = scipy.ndimage.binary_dilation(closed, iterations=2)
            contour = dilated & (~closed)
            rgb[contour] = np.array([0.88, 0.12, 0.22], dtype=np.float32)

    rgb_clipped = np.clip(rgb, 0.0, 1.0)
    return Image.fromarray((rgb_clipped * 255.0).astype(np.uint8))


def create_comparison_grid_figure(
    rgb_t1: np.ndarray,
    rgb_t2: np.ndarray,
    cam_t1: np.ndarray,
    cam_t2: np.ndarray,
    change_exp: np.ndarray,
    delta_ndvi: np.ndarray,
    overlay_t1: np.ndarray,
    overlay_t2: np.ndarray,
    overlay_change: np.ndarray,
    title: str = "Bi-Temporal Explainability Grid",
) -> plt.Figure:
    """Create a structured 3x3 scientific comparison figure."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.96)

    # Row 1: True Color & Spectral Change
    axes[0, 0].imshow(rgb_t1)
    axes[0, 0].set_title("T1 True Color (2020-08-08)\nRGB: B04, B03, B02", fontsize=11, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(rgb_t2)
    axes[0, 1].set_title("T2 True Color (2021-08-08)\nRGB: B04, B03, B02", fontsize=11, fontweight="bold")
    axes[0, 1].axis("off")

    im_dndvi = axes[0, 2].imshow(delta_ndvi, cmap="RdYlGn", vmin=-0.6, vmax=0.2)
    axes[0, 2].set_title("Direct Spectral ΔNDVI\n(NDVI_T2 - NDVI_T1)", fontsize=11, fontweight="bold")
    axes[0, 2].axis("off")
    fig.colorbar(im_dndvi, ax=axes[0, 2], fraction=0.046, pad=0.04, label="ΔNDVI")

    # Row 2: Raw Grad-CAM Salience Maps
    im_c1 = axes[1, 0].imshow(cam_t1, cmap="magma", vmin=0.0, vmax=1.0)
    axes[1, 0].set_title("T1 Grad-CAM Activation\n(Baseline Salience)", fontsize=11, fontweight="bold")
    axes[1, 0].axis("off")
    fig.colorbar(im_c1, ax=axes[1, 0], fraction=0.046, pad=0.04, label="Activation")

    im_c2 = axes[1, 1].imshow(cam_t2, cmap="magma", vmin=0.0, vmax=1.0)
    axes[1, 1].set_title("T2 Grad-CAM Activation\n(Monitoring Salience)", fontsize=11, fontweight="bold")
    axes[1, 1].axis("off")
    fig.colorbar(im_c2, ax=axes[1, 1], fraction=0.046, pad=0.04, label="Activation")

    im_ch = axes[1, 2].imshow(change_exp, cmap="plasma", vmin=0.0, vmax=1.0)
    axes[1, 2].set_title("Derived Relative Salience\nPre-norm: ReLU(C2 - C1)", fontsize=11, fontweight="bold")
    axes[1, 2].axis("off")
    fig.colorbar(im_ch, ax=axes[1, 2], fraction=0.046, pad=0.04, label="Relative Salience")

    # Row 3: Image Overlays
    axes[2, 0].imshow(overlay_t1)
    axes[2, 0].set_title("T1 Scene + Grad-CAM\nBaseline Salience", fontsize=11, fontweight="bold")
    axes[2, 0].axis("off")

    axes[2, 1].imshow(overlay_t2)
    axes[2, 1].set_title("T2 Scene + Grad-CAM\nRelative Salience Region", fontsize=11, fontweight="bold")
    axes[2, 1].axis("off")

    axes[2, 2].imshow(overlay_change)
    axes[2, 2].set_title("T2 Scene + Change Salience\nModel-Highlighted Region", fontsize=11, fontweight="bold")
    axes[2, 2].axis("off")

    plt.tight_layout()
    return fig


def create_spectral_profile_figure(
    t1_patch: np.ndarray,
    t2_patch: np.ndarray,
    ndvi_t1: np.ndarray,
    ndvi_t2: np.ndarray,
) -> plt.Figure:
    """Create spectral band reflectance and NDVI histogram comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 1. Mean Band Reflectance Comparison
    bands = ["B02 (Blue)", "B03 (Green)", "B04 (Red)", "B08 (NIR)"]
    mean_t1 = [np.mean(t1_patch[..., i]) for i in range(4)]
    mean_t2 = [np.mean(t2_patch[..., i]) for i in range(4)]

    x = np.arange(len(bands))
    width = 0.35

    axes[0].bar(x - width/2, mean_t1, width, label="T1 (2020-08-08)", color="#2ca02c", alpha=0.85)
    axes[0].bar(x + width/2, mean_t2, width, label="T2 (2021-08-08)", color="#d62728", alpha=0.85)
    axes[0].set_ylabel("Mean Surface Reflectance", fontweight="bold")
    axes[0].set_title("Mean Spectral Reflectance by Band", fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(bands)
    axes[0].grid(True, linestyle="--", alpha=0.3)
    axes[0].legend()

    # 2. NDVI Distribution Comparison
    axes[1].hist(ndvi_t1.ravel(), bins=40, range=(-0.2, 1.0), alpha=0.6, color="#2ca02c", label="T1 NDVI (Baseline)")
    axes[1].hist(ndvi_t2.ravel(), bins=40, range=(-0.2, 1.0), alpha=0.6, color="#d62728", label="T2 NDVI (Monitoring)")
    axes[1].axvline(x=0.6, color="grey", linestyle=":", label="Healthy Canopy (~0.6)")
    axes[1].set_xlabel("NDVI Value", fontweight="bold")
    axes[1].set_ylabel("Pixel Frequency", fontweight="bold")
    axes[1].set_title("Patch NDVI Pixel Distribution", fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    return fig
