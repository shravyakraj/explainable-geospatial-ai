"""Image and figure rendering components for Streamlit explainability dashboard."""

from typing import Any, Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import scipy.ndimage

# Dark theme aesthetic constants
DARK_FIG_BG = "#131920"
DARK_PANEL_BG = "#0D1117"
DARK_TEXT = "#E6EDF3"
DARK_MUTED = "#8B949E"
DARK_BORDER = "#30363D"
DARK_GRID = "#21262D"
EMERALD_ACCENT = "#10B981"


def _style_colorbar(cb, label: Optional[str] = None):
    """Apply consistent dark styling to a matplotlib colorbar."""
    if label:
        cb.set_label(label, color=DARK_TEXT, fontsize=9, fontweight="bold")
    cb.ax.yaxis.set_tick_params(color=DARK_BORDER, labelcolor=DARK_MUTED, labelsize=8)
    cb.outline.set_edgecolor(DARK_BORDER)


def array_to_pil_image(arr: np.ndarray) -> Image.Image:
    """Convert float [0, 1] RGB array of shape (H, W, 3) to PIL Image."""
    arr_clipped = np.clip(arr, 0.0, 1.0)
    arr_uint8 = (arr_clipped * 255.0).astype(np.uint8)
    return Image.fromarray(arr_uint8)


def create_display_rgb(
    bands: np.ndarray,
    w_mask: Optional[np.ndarray] = None,
    p_low: float = 1.0,
    p_high: float = 99.0,
    joint_reference: Optional[np.ndarray] = None,
    gamma: float = 1.25,
    preserve_color_balance: bool = True,
) -> np.ndarray:
    """Create a refined, natural true-color float32 [0, 1] RGB composite for display only.

    Sentinel-2 input contract:
    - Input shape: (H, W, 4) with bands [B02 (Blue), B03 (Green), B04 (Red), B08 (NIR)],
      or (H, W, 3) for standard RGB imagery.
    - Display channel mapping: Red = B04 (channel 2), Green = B03 (channel 1), Blue = B02 (channel 0).
    - Near-Infrared B08 (channel 3) is strictly excluded from visible RGB rendering.
    - Uses joint scalar percentile stretching (default 1.0% to 99.0%) across all visible channels
      when preserve_color_balance=True. This strictly preserves the natural physical reflectance
      ratios between bands, ensuring forest renders natural deep green and bare soil renders warm
      earth tones without artificial cyan or blue tinting.
    - Applies a subtle gamma correction (default gamma=1.25) to gently lift dense forest canopy
      shadows and midtones without clipping or blowing out highlights.
    - If joint_reference is provided (e.g. combined T1+T2), percentiles are computed jointly
      to guarantee identical radiometric scaling between observation dates.
    - Safe against NaNs, Infs, excluded pixels, and constant/flat bands.
    - Does NOT mutate model input arrays.
    """
    arr = np.nan_to_num(bands, nan=0.0, posinf=1.0, neginf=0.0)

    if arr.ndim == 3 and arr.shape[-1] >= 4:
        # Sentinel-2: Red = B04 (idx 2), Green = B03 (idx 1), Blue = B02 (idx 0)
        rgb = arr[..., [2, 1, 0]].astype(np.float32)
    elif arr.ndim == 3 and arr.shape[-1] == 3:
        # Standard RGB: Red = 0, Green = 1, Blue = 2
        rgb = arr[..., :3].astype(np.float32)
    else:
        raise ValueError(f"Expected array with 3 or 4 channels, got shape {arr.shape}")

    # Determine reference array for percentile estimation
    if joint_reference is not None:
        ref_arr = np.nan_to_num(joint_reference, nan=0.0, posinf=1.0, neginf=0.0)
        if ref_arr.ndim == 3 and ref_arr.shape[-1] >= 4:
            ref_rgb = ref_arr[..., [2, 1, 0]].astype(np.float32)
        elif ref_arr.ndim == 3 and ref_arr.shape[-1] == 3:
            ref_rgb = ref_arr[..., :3].astype(np.float32)
        else:
            ref_rgb = rgb
    else:
        ref_rgb = rgb

    # Determine valid pixel mask for statistical calculation
    if w_mask is not None:
        w_bool = (w_mask[..., 0] > 0.5) if w_mask.ndim == 3 else (w_mask > 0.5)
        # Handle case where joint reference has different height or insufficient valid pixels
        if w_bool.shape[0] != ref_rgb.shape[0] or np.sum(w_bool) < 50:
            w_bool = np.ones(ref_rgb.shape[:2], dtype=bool)
    else:
        w_bool = np.ones(ref_rgb.shape[:2], dtype=bool)

    if preserve_color_balance:
        # Scientific remote sensing practice: stretch all visible channels by shared scalar percentiles
        # to strictly preserve R:G:B spectral proportions
        ref_valid_vals = ref_rgb[w_bool]
        if ref_valid_vals.size == 0:
            p_lo_val, p_hi_val = 0.0, 1.0
        else:
            p_lo_val = float(np.percentile(ref_valid_vals, p_low))
            p_hi_val = float(np.percentile(ref_valid_vals, p_high))

        # Handle flat or zero dynamic range safely
        if (p_hi_val - p_lo_val) < 1e-4:
            p_lo_val = float(np.min(ref_valid_vals)) if ref_valid_vals.size > 0 else 0.0
            p_hi_val = float(np.max(ref_valid_vals)) if ref_valid_vals.size > 0 else 1.0
            if (p_hi_val - p_lo_val) < 1e-4:
                p_hi_val = p_lo_val + 1.0

        out = np.clip((rgb - p_lo_val) / (p_hi_val - p_lo_val), 0.0, 1.0)
    else:
        # Fallback per-channel stretch
        out = np.zeros_like(rgb, dtype=np.float32)
        for c in range(3):
            c_vals = ref_rgb[w_bool, c]
            if len(c_vals) == 0:
                p_lo_c, p_hi_c = 0.0, 1.0
            else:
                p_lo_c = float(np.percentile(c_vals, p_low))
                p_hi_c = float(np.percentile(c_vals, p_high))
            if (p_hi_c - p_lo_c) < 1e-4:
                p_lo_c = float(np.min(c_vals)) if len(c_vals) > 0 else 0.0
                p_hi_c = float(np.max(c_vals)) if len(c_vals) > 0 else 1.0
                if (p_hi_c - p_lo_c) < 1e-4:
                    p_hi_c = p_lo_c + 1.0
            out[..., c] = np.clip((rgb[..., c] - p_lo_c) / (p_hi_c - p_lo_c), 0.0, 1.0)

    # Subtle gamma adjustment to naturally illuminate canopy shadow detail without highlight blowout
    if gamma is not None and gamma > 0 and abs(gamma - 1.0) > 1e-3:
        out = np.power(out, 1.0 / float(gamma))

    return out.astype(np.float32)


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
        weights = np.clip((sal - threshold) / (1.0 - threshold + 1e-6), 0.0, 1.0)
        c_low = np.array([0.98, 0.58, 0.56], dtype=np.float32)
        c_high = np.array([0.91, 0.16, 0.24], dtype=np.float32)
        tint_color = (1.0 - weights[..., None]) * c_low + weights[..., None] * c_high
        alpha = (0.35 + 0.35 * (weights ** 0.65))[..., None]
        rgb = np.where(mask[..., None], (1.0 - alpha) * rgb + alpha * tint_color, rgb)

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
    title: Optional[str] = None,
) -> plt.Figure:
    """Create a 3x3 diagnostic comparison grid with dark geospatial styling."""
    fig, axes = plt.subplots(3, 3, figsize=(14, 14), dpi=150, facecolor=DARK_FIG_BG)
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995, color=DARK_TEXT)

    # Row 1: True Color & ΔNDVI
    axes[0, 0].imshow(rgb_t1)
    axes[0, 0].set_title("T1 Scene (08 Aug 2020)\nTrue Color Surface Reflectance", fontsize=11, fontweight="bold", color=DARK_TEXT)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(rgb_t2)
    axes[0, 1].set_title("T2 Scene (08 Aug 2021)\nTrue Color Surface Reflectance", fontsize=11, fontweight="bold", color=DARK_TEXT)
    axes[0, 1].axis("off")

    im_ndvi = axes[0, 2].imshow(delta_ndvi, cmap="RdYlGn", vmin=-0.8, vmax=0.8)
    axes[0, 2].set_title("Spectral Delta-NDVI (T2 - T1)\nVegetation Loss / Regrowth", fontsize=11, fontweight="bold", color=DARK_TEXT)
    axes[0, 2].axis("off")
    cb_ndvi = fig.colorbar(im_ndvi, ax=axes[0, 2], fraction=0.046, pad=0.04)
    _style_colorbar(cb_ndvi, label="ΔNDVI")

    # Row 2: Raw Heatmaps
    im_cam1 = axes[1, 0].imshow(cam_t1, cmap="magma", vmin=0, vmax=1)
    axes[1, 0].set_title("T1 Grad-CAM Activation\nPre-Disturbance Baseline", fontsize=11, fontweight="bold", color=DARK_TEXT)
    axes[1, 0].axis("off")
    cb_cam1 = fig.colorbar(im_cam1, ax=axes[1, 0], fraction=0.046, pad=0.04)
    _style_colorbar(cb_cam1, label="Class Activation")

    im_cam2 = axes[1, 1].imshow(cam_t2, cmap="magma", vmin=0, vmax=1)
    axes[1, 1].set_title("T2 Grad-CAM Activation\nDisturbance Anomaly Salience", fontsize=11, fontweight="bold", color=DARK_TEXT)
    axes[1, 1].axis("off")
    cb_cam2 = fig.colorbar(im_cam2, ax=axes[1, 1], fraction=0.046, pad=0.04)
    _style_colorbar(cb_cam2, label="Class Activation")

    im_ch = axes[1, 2].imshow(change_exp, cmap="magma", vmin=0, vmax=1)
    axes[1, 2].set_title("Change Salience Map\nIntegrated Temporal Contrast", fontsize=11, fontweight="bold", color=DARK_TEXT)
    axes[1, 2].axis("off")
    cb_ch = fig.colorbar(im_ch, ax=axes[1, 2], fraction=0.046, pad=0.04)
    _style_colorbar(cb_ch, label="Relative Salience")

    # Row 3: Image Overlays
    axes[2, 0].imshow(overlay_t1)
    axes[2, 0].set_title("T1 Scene + Grad-CAM\nBaseline Salience", fontsize=11, fontweight="bold", color=DARK_TEXT)
    axes[2, 0].axis("off")

    axes[2, 1].imshow(overlay_t2)
    axes[2, 1].set_title("T2 Scene + Grad-CAM\nRelative Salience Region", fontsize=11, fontweight="bold", color=DARK_TEXT)
    axes[2, 1].axis("off")

    axes[2, 2].imshow(overlay_change)
    axes[2, 2].set_title("T2 Scene + Change Salience\nModel-Highlighted Region", fontsize=11, fontweight="bold", color=DARK_TEXT)
    axes[2, 2].axis("off")

    plt.tight_layout()
    return fig


def create_spectral_profile_figure(
    t1_patch: np.ndarray,
    t2_patch: np.ndarray,
    ndvi_t1: np.ndarray,
    ndvi_t2: np.ndarray,
) -> plt.Figure:
    """Create spectral band reflectance and NDVI histogram comparison in dark styling."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=150, facecolor=DARK_FIG_BG)
    for ax in axes:
        ax.set_facecolor(DARK_PANEL_BG)
        ax.tick_params(colors=DARK_MUTED)
        for spine in ax.spines.values():
            spine.set_color(DARK_BORDER)

    bands = ["B02 (Blue)", "B03 (Green)", "B04 (Red)", "B08 (NIR)"]
    mean_t1 = [np.mean(t1_patch[..., i]) for i in range(4)]
    mean_t2 = [np.mean(t2_patch[..., i]) for i in range(4)]

    x = np.arange(len(bands))
    width = 0.35

    axes[0].bar(x - width/2, mean_t1, width, label="T1 (2020-08-08)", color="#10B981", alpha=0.9)
    axes[0].bar(x + width/2, mean_t2, width, label="T2 (2021-08-08)", color="#EF4444", alpha=0.9)
    axes[0].set_ylabel("Mean Surface Reflectance", fontweight="bold", color=DARK_TEXT)
    axes[0].set_title("Mean Spectral Reflectance by Band", fontweight="bold", color=DARK_TEXT)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(bands, color=DARK_TEXT)
    axes[0].grid(True, linestyle="--", alpha=0.3, color=DARK_GRID)
    axes[0].legend(facecolor=DARK_FIG_BG, edgecolor=DARK_BORDER, labelcolor=DARK_TEXT)

    axes[1].hist(ndvi_t1.ravel(), bins=40, range=(-0.2, 1.0), alpha=0.6, color="#10B981", label="T1 NDVI (Baseline)")
    axes[1].hist(ndvi_t2.ravel(), bins=40, range=(-0.2, 1.0), alpha=0.6, color="#EF4444", label="T2 NDVI (Monitoring)")
    axes[1].axvline(x=0.6, color=DARK_MUTED, linestyle=":", label="Healthy Canopy (~0.6)")
    axes[1].set_xlabel("NDVI Value", fontweight="bold", color=DARK_TEXT)
    axes[1].set_ylabel("Pixel Frequency", fontweight="bold", color=DARK_TEXT)
    axes[1].set_title("Patch NDVI Pixel Distribution", fontweight="bold", color=DARK_TEXT)
    axes[1].grid(True, linestyle="--", alpha=0.3, color=DARK_GRID)
    axes[1].legend(facecolor=DARK_FIG_BG, edgecolor=DARK_BORDER, labelcolor=DARK_TEXT)

    plt.tight_layout()
    return fig


def create_segmentation_figure(
    rgb_t2: np.ndarray,
    prob_map: np.ndarray,
    binary_pred: np.ndarray,
    w_mask: Optional[np.ndarray] = None,
    threshold: float = 0.500,
) -> plt.Figure:
    """Create 3-panel figure showing T2 RGB, U-Net Probability Map, and Masked Prediction."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150, facecolor=DARK_FIG_BG)

    # 1. T2 True-Color RGB
    axes[0].imshow(np.clip(rgb_t2, 0.0, 1.0))
    axes[0].set_title("Monitoring Scene (08 Aug 2021)\nRefined True Color", fontsize=11, fontweight="bold", color=DARK_TEXT)
    axes[0].axis("off")

    # 2. Probability Heatmap with threshold contour
    im_p = axes[1].imshow(prob_map, cmap="magma", vmin=0.0, vmax=1.0)
    if np.any(prob_map >= threshold):
        axes[1].contour(prob_map >= threshold, levels=[0.5], colors="#00F0FF", linewidths=1.2)
    axes[1].set_title(f"Early-Fusion U-Net Probability Map\n[Cyan: τ* ≥ {threshold:.2f}]", fontsize=11, fontweight="bold", color=DARK_TEXT)
    axes[1].axis("off")
    cb_p = fig.colorbar(im_p, ax=axes[1], fraction=0.046, pad=0.04)
    _style_colorbar(cb_p, label="Deforestation Probability")

    # 3. Dense Masked Overlay
    overlay = np.clip(rgb_t2, 0.0, 1.0).copy()
    pred_pos = binary_pred > 0.5
    if w_mask is not None:
        pred_pos = pred_pos & (w_mask[..., 0] > 0.5)
        excl = w_mask[..., 0] <= 0.5
        overlay[excl] = overlay[excl] * 0.4 + np.array([0.25, 0.28, 0.32]) * 0.6

    overlay[pred_pos] = overlay[pred_pos] * 0.4 + np.array([0.95, 0.22, 0.22]) * 0.6
    axes[2].imshow(overlay)
    axes[2].set_title(f"Predicted Deforestation Mask\n[Red: Clearing ({np.sum(pred_pos)*0.01:.1f} ha)]", fontsize=11, fontweight="bold", color=DARK_TEXT)
    axes[2].axis("off")

    plt.tight_layout()
    return fig


def create_confusion_map_figure(
    rgb_t2: np.ndarray,
    y_ref: np.ndarray,
    y_pred: np.ndarray,
    w_mask: np.ndarray,
) -> Tuple[plt.Figure, Dict[str, Any]]:
    """Create pixel confusion map comparing model predictions to reference cartography in dark styling."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150, facecolor=DARK_FIG_BG)

    w_2d = w_mask[..., 0] > 0.5
    ref_pos = (y_ref[..., 0] > 0.5) & w_2d
    pred_pos = (y_pred[..., 0] > 0.5 if y_pred.ndim == 3 else y_pred > 0.5) & w_2d

    tp = ref_pos & pred_pos
    fp = (~ref_pos) & pred_pos
    fn = ref_pos & (~pred_pos)
    tn = (~ref_pos) & (~pred_pos) & w_2d
    excl = ~w_2d

    tp_count = int(np.sum(tp))
    fp_count = int(np.sum(fp))
    fn_count = int(np.sum(fn))
    tn_count = int(np.sum(tn))
    valid_count = tp_count + fp_count + fn_count + tn_count

    iou = tp_count / max(1, tp_count + fp_count + fn_count)
    dice = 2 * tp_count / max(1, 2 * tp_count + fp_count + fn_count)
    precision = tp_count / max(1, tp_count + fp_count)
    recall = tp_count / max(1, tp_count + fn_count)

    # Render True Color Scene
    axes[0].imshow(np.clip(rgb_t2, 0.0, 1.0))
    axes[0].set_title("Monitoring (2021) Sentinel-2 Scene", fontsize=11, fontweight="bold", color=DARK_TEXT)
    axes[0].axis("off")

    # Render Discrete Confusion Color Map
    conf_img = np.zeros((256, 256, 3), dtype=np.float32)
    conf_img[tn] = [0.08, 0.24, 0.12]  # Intact Forest (deep forest emerald)
    conf_img[tp] = [0.10, 0.85, 0.35]  # True Positive (vivid emerald green)
    conf_img[fp] = [0.95, 0.20, 0.20]  # False Positive (crimson red)
    conf_img[fn] = [0.95, 0.75, 0.15]  # False Negative (amber gold)
    conf_img[excl] = [0.20, 0.23, 0.28]  # Excluded Clouds / Prior Loss (neutral dark slate)

    axes[1].imshow(conf_img)
    axes[1].set_title(
        f"Pixel Confusion vs. PRODES Reference\n"
        f"IoU: {iou:.3f} | Dice: {dice:.3f} | Prec: {precision:.3f} | Rec: {recall:.3f}",
        fontsize=11,
        fontweight="bold",
        color=DARK_TEXT,
    )
    axes[1].axis("off")

    plt.tight_layout()

    stats = {
        "tp": tp_count,
        "fp": fp_count,
        "fn": fn_count,
        "tn": tn_count,
        "valid_pixels": valid_count,
        "tp_ha": round(tp_count * 0.01, 2),
        "fp_ha": round(fp_count * 0.01, 2),
        "fn_ha": round(fn_count * 0.01, 2),
        "ref_loss_ha": round((tp_count + fn_count) * 0.01, 2),
        "pred_loss_ha": round((tp_count + fp_count) * 0.01, 2),
        "iou": round(iou, 4),
        "dice": round(dice, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }

    return fig, stats


def create_integrated_gradients_figure(
    spatial_attr: np.ndarray,
    t1_attr: np.ndarray,
    t2_attr: np.ndarray,
    w_mask: np.ndarray,
    per_band_attributions: Dict[str, Any],
    completeness_error: float = 0.0,
) -> plt.Figure:
    """Create 4-panel figure showing Net Spatial Attribution, T1 Attr, T2 Attr, and Spectral Bar Chart."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), dpi=150, facecolor=DARK_FIG_BG)

    valid_attr = spatial_attr[w_mask[..., 0] > 0.5]
    v_abs = np.percentile(np.abs(valid_attr), 99) if len(valid_attr) > 0 else 1.0
    v_abs = max(1e-4, float(v_abs))

    masked_net = np.ma.masked_where(w_mask[..., 0] <= 0.5, spatial_attr)
    masked_t1 = np.ma.masked_where(w_mask[..., 0] <= 0.5, t1_attr)
    masked_t2 = np.ma.masked_where(w_mask[..., 0] <= 0.5, t2_attr)

    cmap = "coolwarm"

    # 1. Net Spatial
    im1 = axes[0].imshow(masked_net, cmap=cmap, vmin=-v_abs, vmax=v_abs)
    axes[0].set_title("Net Attribution (T1+T2)\n[Red: Pushes Loss, Blue: Suppresses]", fontsize=10, fontweight="bold", color=DARK_TEXT)
    axes[0].axis("off")
    cb1 = fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    _style_colorbar(cb1)

    # 2. T1 Spatial
    im2 = axes[1].imshow(masked_t1, cmap=cmap, vmin=-v_abs, vmax=v_abs)
    axes[1].set_title("T1 Pre-Clearing Sensitivity\n[Initial Forest Canopy]", fontsize=10, fontweight="bold", color=DARK_TEXT)
    axes[1].axis("off")
    cb2 = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    _style_colorbar(cb2)

    # 3. T2 Spatial
    im3 = axes[2].imshow(masked_t2, cmap=cmap, vmin=-v_abs, vmax=v_abs)
    axes[2].set_title("T2 Post-Disturbance Sensitivity\n[Bare Soil vs Vegetation]", fontsize=10, fontweight="bold", color=DARK_TEXT)
    axes[2].axis("off")
    cb3 = fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    _style_colorbar(cb3)

    # 4. Spectral Breakdown Bar Chart
    axes[3].set_facecolor(DARK_PANEL_BG)
    axes[3].tick_params(colors=DARK_MUTED)
    for spine in axes[3].spines.values():
        spine.set_color(DARK_BORDER)

    band_keys = ["B02", "B03", "B04", "B08"]
    labels = ["B02 (Blue)", "B03 (Grn)", "B04 (Red)", "B08 (NIR)"]
    t1_vals = [per_band_attributions[k]["t1"] for k in band_keys]
    t2_vals = [per_band_attributions[k]["t2"] for k in band_keys]

    x_idx = np.arange(len(labels))
    bar_width = 0.35

    axes[3].bar(x_idx - bar_width/2, t1_vals, bar_width, label="T1 (2020)", color="#38BDF8")
    axes[3].bar(x_idx + bar_width/2, t2_vals, bar_width, label="T2 (2021)", color="#F43F5E")
    axes[3].axhline(0, color=DARK_BORDER, linewidth=0.8, linestyle="--")
    axes[3].set_xticks(x_idx)
    axes[3].set_xticklabels(labels, fontsize=8, color=DARK_TEXT)
    axes[3].set_title(f"Spectral Breakdown by Date\nCompleteness Err: {completeness_error:.2%}", fontsize=10, fontweight="bold", color=DARK_TEXT)
    axes[3].grid(True, linestyle=":", alpha=0.3, color=DARK_GRID)
    axes[3].legend(fontsize=8, facecolor=DARK_FIG_BG, edgecolor=DARK_BORDER, labelcolor=DARK_TEXT)

    plt.tight_layout()
    return fig


def create_lime_figure(
    rgb_composite: np.ndarray,
    segments: np.ndarray,
    supporting_mask: np.ndarray,
    opposing_mask: np.ndarray,
    top_features: List[Tuple[int, float]],
    target_date: str = "T2",
) -> plt.Figure:
    """Create figure displaying LIME superpixels, supporting vs opposing masks, and ranked weights in dark styling."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=150, facecolor=DARK_FIG_BG)

    # 1. Superpixel Segmentation Boundaries
    boundaries = scipy.ndimage.generic_gradient_magnitude(segments.astype(np.float32), scipy.ndimage.sobel) > 0
    seg_img = np.clip(rgb_composite, 0.0, 1.0).copy()
    seg_img[boundaries] = [0.95, 0.85, 0.2]  # Yellow superpixel boundaries
    axes[0].imshow(seg_img)
    axes[0].set_title(f"{target_date} CIR Superpixels (SLIC)\n[Yellow: 35 Segments]", fontsize=10, fontweight="bold", color=DARK_TEXT, pad=8)
    axes[0].axis("off")

    # 2. Supporting vs Opposing Masks
    mask_img = np.clip(rgb_composite, 0.0, 1.0).copy()
    mask_img[supporting_mask] = mask_img[supporting_mask] * 0.4 + np.array([0.86, 0.15, 0.15]) * 0.6
    mask_img[opposing_mask] = mask_img[opposing_mask] * 0.4 + np.array([0.15, 0.40, 0.85]) * 0.6
    axes[1].imshow(mask_img)
    axes[1].set_title(f"{target_date} Saliency Direction\n[Red: Supports Loss, Blue: Opposes]", fontsize=10, fontweight="bold", color=DARK_TEXT, pad=8)
    axes[1].axis("off")

    # 3. Top Feature Weights
    axes[2].set_facecolor(DARK_PANEL_BG)
    axes[2].tick_params(colors=DARK_MUTED)
    for spine in axes[2].spines.values():
        spine.set_color(DARK_BORDER)

    top_k = min(8, len(top_features))
    features = top_features[:top_k]
    seg_labels = [f"Seg {sid}" for sid, _ in features]
    weights = [w for _, w in features]
    colors = ["#EF4444" if w > 0 else "#3B82F6" for w in weights]

    y_pos = np.arange(top_k)
    axes[2].barh(y_pos, weights, color=colors, height=0.6)
    axes[2].set_yticks(y_pos)
    axes[2].set_yticklabels(seg_labels, fontsize=9, color=DARK_TEXT)
    axes[2].invert_yaxis()
    axes[2].axvline(0, color=DARK_BORDER, linestyle="--", linewidth=0.8)
    axes[2].set_title("Ranked Superpixel Weights\n(Surrogate Ridge Coefficients)", fontsize=10, fontweight="bold", color=DARK_TEXT, pad=8)
    axes[2].set_xlabel("Local Attribution Weight", fontsize=9, color=DARK_TEXT)
    axes[2].grid(True, linestyle=":", alpha=0.3, color=DARK_GRID)

    plt.tight_layout()
    return fig


def create_bitemporal_comparison_figure(
    rgb_t1: np.ndarray,
    rgb_t2: np.ndarray,
    date1_label: str = "Date 1 (08 Aug 2020) True Color",
    date2_label: str = "Date 2 (08 Aug 2021) True Color",
    title: Optional[str] = None,
) -> plt.Figure:
    """Create side-by-side comparison of pre- and post-disturbance true color satellite imagery in dark styling."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), dpi=150, facecolor=DARK_FIG_BG)
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", color=DARK_TEXT, y=0.98)

    axes[0].imshow(np.clip(rgb_t1, 0.0, 1.0))
    axes[0].set_title(date1_label, fontsize=11, fontweight="bold", color=DARK_TEXT, pad=8)
    axes[0].axis("off")

    axes[1].imshow(np.clip(rgb_t2, 0.0, 1.0))
    axes[1].set_title(date2_label, fontsize=11, fontweight="bold", color=DARK_TEXT, pad=8)
    axes[1].axis("off")

    plt.tight_layout()
    return fig


def create_prediction_mask_figure(
    prob_map: np.ndarray,
    pred_mask: np.ndarray,
    w_mask: Optional[np.ndarray] = None,
    threshold: float = 0.500,
    title: Optional[str] = None,
) -> plt.Figure:
    """Create 2-panel figure showing continuous U-Net probability map and discrete binary predicted mask in dark styling."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), dpi=150, facecolor=DARK_FIG_BG)
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", color=DARK_TEXT, y=0.98)

    # 1. Probability Map
    im_p = axes[0].imshow(prob_map, cmap="magma", vmin=0.0, vmax=1.0)
    if np.any(prob_map >= threshold):
        axes[0].contour(prob_map >= threshold, levels=[0.5], colors="#00F0FF", linewidths=1.2)
    axes[0].set_title(f"Continuous Probability Map\n[Cyan contour: tau* >= {threshold:.2f}]", fontsize=11, fontweight="bold", color=DARK_TEXT, pad=8)
    axes[0].axis("off")
    cb_p = fig.colorbar(im_p, ax=axes[0], fraction=0.046, pad=0.04)
    _style_colorbar(cb_p, label="Deforestation Probability")

    # 2. Discrete Binary Prediction Mask
    mask_display = np.zeros((256, 256, 3), dtype=np.float32)
    # Intact / undisturbed forest (deep dark forest green background)
    mask_display[:] = [0.06, 0.16, 0.08]

    pred_pos = pred_mask > 0.5
    if w_mask is not None:
        pred_pos = pred_pos & (w_mask[..., 0] > 0.5)
        # Excluded pixels in neutral dark slate
        excl = w_mask[..., 0] <= 0.5
        mask_display[excl] = [0.20, 0.23, 0.28]

    # Predicted deforestation in bright coral-red
    mask_display[pred_pos] = [0.95, 0.25, 0.25]

    axes[1].imshow(mask_display)
    loss_ha = float(np.sum(pred_pos)) * 0.01
    axes[1].set_title(
        f"Discrete Segmentation Prediction (tau* = {threshold:.2f})\n"
        f"[Red: Predicted Clearing ({loss_ha:.2f} ha) | Green: Intact]",
        fontsize=11,
        fontweight="bold",
        color=DARK_TEXT,
    )
    axes[1].axis("off")

    plt.tight_layout()
    return fig


def create_overlay_figure(
    rgb_t2: np.ndarray,
    pred_mask: np.ndarray,
    w_mask: Optional[np.ndarray] = None,
    alpha: float = 0.50,
    title: Optional[str] = None,
) -> plt.Figure:
    """Create figure showing predicted deforestation mask overlaid on Date 2 satellite imagery with adjustable opacity in dark styling."""
    fig, ax = plt.subplots(figsize=(7, 7), dpi=150, facecolor=DARK_FIG_BG)
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", color=DARK_TEXT, y=0.98)

    base = np.clip(rgb_t2, 0.0, 1.0).copy()
    overlay = base.copy()

    pred_pos = pred_mask > 0.5
    if w_mask is not None:
        pred_pos = pred_pos & (w_mask[..., 0] > 0.5)
        excl = w_mask[..., 0] <= 0.5
        # Tint excluded regions neutral dark slate
        overlay[excl] = overlay[excl] * 0.5 + np.array([0.25, 0.28, 0.32], dtype=np.float32) * 0.5

    # Tint predicted clearing in vivid, clear red (#DC2626) with configurable alpha
    c_red = np.array([0.86, 0.15, 0.15], dtype=np.float32)
    overlay[pred_pos] = (1.0 - alpha) * overlay[pred_pos] + alpha * c_red

    ax.imshow(overlay)
    loss_ha = float(np.sum(pred_pos)) * 0.01
    ax.set_title(
        f"Deforestation Mask Overlay on Monitoring Scene\n"
        f"[Red tint: {loss_ha:.2f} ha predicted clearing (opacity={alpha:.2f})]",
        fontsize=11,
        fontweight="bold",
        color=DARK_TEXT,
        pad=10,
    )
    ax.axis("off")

    plt.tight_layout()
    return fig


def create_annotated_candidate_regions_figure(
    rgb_disp_t2: np.ndarray,
    regions: List[Any],
    binary_mask: np.ndarray,
    w_mask: Optional[np.ndarray] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """Create satellite overview map with outlines and numbered callouts for each suspected deforestation region in dark styling."""
    fig, ax = plt.subplots(figsize=(8.5, 8.5), dpi=150, facecolor=DARK_FIG_BG)
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", color=DARK_TEXT, y=0.98)

    base = np.clip(rgb_disp_t2, 0.0, 1.0).copy()
    ax.imshow(base)

    # Plot contours for candidate areas
    b_2d = (binary_mask[..., 0] > 0.5) if binary_mask.ndim == 3 else (binary_mask > 0.5)
    if w_mask is not None:
        w_2d = (w_mask[..., 0] > 0.5) if w_mask.ndim == 3 else (w_mask > 0.5)
        b_2d = b_2d & w_2d

    if np.any(b_2d):
        ax.contour(b_2d, levels=[0.5], colors="#EF4444", linewidths=1.8)

    # Add numbered callout boxes for candidate regions
    for reg in regions[:12]:  # Annotate prominent candidate regions to maintain readability
        # Centroid in pixel coordinates (r_c, c_c)
        r_c, c_c = reg.centroid_pixel
        label_text = f"{reg.region_id}\n{reg.area_ha:.1f} ha"

        # Offset text so it doesn't obscure the center of the clearing
        offset_x = 14 if c_c < 180 else -38
        offset_y = -14 if r_c > 60 else 22

        ax.annotate(
            label_text,
            xy=(c_c, r_c),
            xytext=(c_c + offset_x, r_c + offset_y),
            fontsize=8,
            fontweight="bold",
            color="#FFFFFF",
            bbox=dict(boxstyle="round,pad=0.30", fc="#064E3B", ec="#EF4444", lw=1.4, alpha=0.95),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1", color="#EF4444", lw=1.3),
        )

    tot_ha = sum(r.area_ha for r in regions)
    ax.set_title(
        f"Candidate Deforestation Event Inventory (8-Connected Segmentation)\n"
        f"[{len(regions)} Identified Regions | {tot_ha:.2f} ha Total Predicted Clearing | Red: Boundary]",
        fontsize=10,
        fontweight="bold",
        color=DARK_TEXT,
        pad=10,
    )
    ax.axis("off")
    plt.tight_layout()
    return fig
