"""Grad-CAM explainability and bi-temporal visualization for deforestation detection."""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


def compute_gradcam(
    model: keras.Model,
    t1_patch: np.ndarray,
    t2_patch: np.ndarray,
    target_layer_name: str = "target_relu",
    return_change: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray, float], Tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    """Compute bi-temporal Grad-CAM heatmaps for T1 and T2 inputs across all Siamese variants.

    Args:
        model: Trained Siamese deforestation model.
        t1_patch: Normalized T1 array of shape (256, 256, 4).
        t2_patch: Normalized T2 array of shape (256, 256, 4).
        target_layer_name: Name of convolutional activation layer in shared backbone.

    Returns:
        cam_t1: Grad-CAM heatmap for T1, resized to (256, 256), in [0, 1].
        cam_t2: Grad-CAM heatmap for T2, resized to (256, 256), in [0, 1].
        pred_prob: Deforestation probability scalar output.
    """
    t1_tensor = tf.convert_to_tensor(t1_patch[np.newaxis, ...], dtype=tf.float32)
    t2_tensor = tf.convert_to_tensor(t2_patch[np.newaxis, ...], dtype=tf.float32)

    backbone = model.get_layer("shared_backbone")
    target_layer = backbone.get_layer(target_layer_name)

    feature_submodel = keras.Model(
        inputs=backbone.input,
        outputs=target_layer.output,
        name="cam_feature_submodel",
    )

    has_gmp = "gmp" in [l.name for l in backbone.layers]
    has_signed = "signed_feature_difference" in [l.name for l in model.layers]
    has_concat = "feature_fusion" in [l.name for l in model.layers]

    with tf.GradientTape(persistent=True) as tape:
        act_t1 = feature_submodel(t1_tensor)
        act_t2 = feature_submodel(t2_tensor)
        tape.watch(act_t1)
        tape.watch(act_t2)

        if has_gmp:
            gap1 = backbone.get_layer("gap")(act_t1)
            gmp1 = backbone.get_layer("gmp")(act_t1)
            gap2 = backbone.get_layer("gap")(act_t2)
            gmp2 = backbone.get_layer("gmp")(act_t2)
            feat_t1 = backbone.get_layer("pool_concat")([gap1, gmp1])
            feat_t2 = backbone.get_layer("pool_concat")([gap2, gmp2])
        else:
            feat_t1 = backbone.get_layer("gap")(act_t1)
            feat_t2 = backbone.get_layer("gap")(act_t2)

        diff_abs = model.get_layer("abs_feature_difference")([feat_t1, feat_t2])

        if has_signed:
            diff_signed = model.get_layer("signed_feature_difference")([feat_t1, feat_t2])
            fused = model.get_layer("feature_fusion")([diff_abs, diff_signed])
        elif has_concat:
            fused = model.get_layer("feature_fusion")([diff_abs, feat_t1, feat_t2])
        else:
            fused = diff_abs

        if "fusion_bn" in [l.name for l in model.layers]:
            fused = model.get_layer("fusion_bn")(fused)

        dense_out = model.get_layer("head_dense")(fused)
        drop_out = model.get_layer("head_dropout")(dense_out, training=False)
        output_prob = model.get_layer("deforestation_prob")(drop_out)

    pred_prob = float(output_prob.numpy()[0, 0])

    grad_t1 = tape.gradient(output_prob, act_t1)
    grad_t2 = tape.gradient(output_prob, act_t2)
    del tape

    def _compute_raw_cam(act: tf.Tensor, grad: tf.Tensor) -> np.ndarray:
        weights = tf.reduce_mean(grad, axis=(1, 2), keepdims=True)
        cam = tf.reduce_sum(weights * act, axis=-1)
        cam = tf.nn.relu(cam)
        return cam.numpy()[0]

    raw_t1 = _compute_raw_cam(act_t1, grad_t1)
    raw_t2 = _compute_raw_cam(act_t2, grad_t2)

    # Option B: Compute temporal comparison from pre-normalization CAM activations,
    # then normalize only the final relative salience difference map.
    raw_change = np.maximum(raw_t2 - raw_t1, 0.0)

    def _normalize_and_resize(arr: np.ndarray) -> np.ndarray:
        max_val = np.max(arr)
        if max_val > 1e-8:
            norm_arr = arr / max_val
        else:
            norm_arr = np.zeros_like(arr)
        resized = tf.image.resize(
            norm_arr[..., np.newaxis], (256, 256), method="bilinear"
        ).numpy()[:, :, 0]
        r_max = np.max(resized)
        if r_max > 1e-8:
            resized = resized / r_max
        return np.clip(resized, 0.0, 1.0)

    cam_t1 = _normalize_and_resize(raw_t1)
    cam_t2 = _normalize_and_resize(raw_t2)
    cam_change = _normalize_and_resize(raw_change)

    if return_change:
        return cam_t1, cam_t2, cam_change, pred_prob
    return cam_t1, cam_t2, pred_prob


def create_rgb_composite(patch: np.ndarray, stretch: float = 3.0) -> np.ndarray:
    """Create RGB true-color image from Sentinel-2 4-band patch (B02, B03, B04, B08)."""
    rgb = np.stack([patch[..., 2], patch[..., 1], patch[..., 0]], axis=-1)
    return np.clip(rgb * stretch, 0.0, 1.0)


def create_false_color_composite(patch: np.ndarray, stretch: float = 2.5) -> np.ndarray:
    """Create standard false-color NIR-R-G composite (B08, B04, B03)."""
    cir = np.stack([patch[..., 3], patch[..., 2], patch[..., 1]], axis=-1)
    return np.clip(cir * stretch, 0.0, 1.0)


def compute_ndvi(patch: np.ndarray) -> np.ndarray:
    """Compute Normalized Difference Vegetation Index (B08 - B04) / (B08 + B04)."""
    nir = patch[..., 3]
    red = patch[..., 2]
    denom = nir + red + 1e-6
    return (nir - red) / denom


def plot_explainability_case(
    t1_patch: np.ndarray,
    t2_patch: np.ndarray,
    label: int,
    pred_prob: float,
    cam_t1: np.ndarray,
    cam_t2: np.ndarray,
    patch_id: str,
    output_path: Path,
    title_prefix: str = "Evaluation Case",
) -> None:
    """Create comprehensive multi-panel explainability figure."""
    rgb_t1 = create_rgb_composite(t1_patch)
    rgb_t2 = create_rgb_composite(t2_patch)
    cir_t1 = create_false_color_composite(t1_patch)
    cir_t2 = create_false_color_composite(t2_patch)

    ndvi_t1 = compute_ndvi(t1_patch)
    ndvi_t2 = compute_ndvi(t2_patch)
    delta_ndvi = ndvi_t2 - ndvi_t1

    pred_label = 1 if pred_prob >= 0.5 else 0
    status = "Correct" if pred_label == label else "Incorrect"
    lbl_str = "Deforestation (1)" if label == 1 else "Intact Forest (0)"
    pred_str = f"Deforestation ({pred_prob:.3f})" if pred_label == 1 else f"Intact Forest ({pred_prob:.3f})"

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    plt.suptitle(
        f"{title_prefix}: {patch_id}\n"
        f"Ground Truth: {lbl_str} | Prediction: {pred_str} [{status}]",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # Row 1: Imagery & Spectral Change
    axes[0, 0].imshow(rgb_t1)
    axes[0, 0].set_title("T1 True Color (2020-08-08)\nRGB: B04, B03, B02", fontsize=11)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(rgb_t2)
    axes[0, 1].set_title("T2 True Color (2021-08-08)\nRGB: B04, B03, B02", fontsize=11)
    axes[0, 1].axis("off")

    im_dndvi = axes[0, 2].imshow(delta_ndvi, cmap="RdYlGn", vmin=-0.6, vmax=0.2)
    axes[0, 2].set_title("Spectral Delta-NDVI\n(NDVI_T2 - NDVI_T1)", fontsize=11)
    axes[0, 2].axis("off")
    fig.colorbar(im_dndvi, ax=axes[0, 2], fraction=0.046, pad=0.04, label="ΔNDVI")

    rgb_diff = np.abs(rgb_t2 - rgb_t1)
    axes[0, 3].imshow(np.clip(rgb_diff * 1.5, 0.0, 1.0))
    axes[0, 3].set_title("RGB Absolute Difference\n|T2 - T1|", fontsize=11)
    axes[0, 3].axis("off")

    # Row 2: False Color and Grad-CAM Overlays
    axes[1, 0].imshow(cir_t1)
    axes[1, 0].set_title("T1 False Color NIR\nCIR: B08, B04, B03", fontsize=11)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(cir_t2)
    axes[1, 1].set_title("T2 False Color NIR\nCIR: B08, B04, B03", fontsize=11)
    axes[1, 1].axis("off")

    axes[1, 2].imshow(rgb_t1)
    im_cam1 = axes[1, 2].imshow(cam_t1, cmap="magma", alpha=0.55, vmin=0.0, vmax=1.0)
    axes[1, 2].set_title("T1 Grad-CAM Overlay\n(Pre-clearing Context)", fontsize=11)
    axes[1, 2].axis("off")
    fig.colorbar(im_cam1, ax=axes[1, 2], fraction=0.046, pad=0.04, label="Grad-CAM")

    axes[1, 3].imshow(rgb_t2)
    im_cam2 = axes[1, 3].imshow(cam_t2, cmap="magma", alpha=0.55, vmin=0.0, vmax=1.0)
    axes[1, 3].set_title("T2 Grad-CAM Overlay\n(Model-Highlighted Region)", fontsize=11)
    axes[1, 3].axis("off")
    fig.colorbar(im_cam2, ax=axes[1, 3], fraction=0.046, pad=0.04, label="Grad-CAM")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved explainability plot: %s", output_path)


def overlay_heatmap_on_image(
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "magma",
) -> np.ndarray:
    """Blend a 2D heatmap in [0, 1] over a base RGB image in [0, 1].

    Args:
        image_rgb: Base RGB image array of shape (H, W, 3) with values in [0, 1].
        heatmap: 2D salience heatmap of shape (H, W) with values in [0, 1].
        alpha: Blending weight for the heatmap overlay (0.0 = base only, 1.0 = heatmap only).
        colormap: Name of matplotlib colormap to apply (e.g. 'magma', 'plasma', 'viridis').

    Returns:
        Blended RGB array of shape (H, W, 3) with values in [0, 1].
    """
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(colormap)
    heatmap_colored = cmap(np.clip(heatmap, 0.0, 1.0))[..., :3]
    blended = (1.0 - alpha) * image_rgb + alpha * heatmap_colored
    return np.clip(blended, 0.0, 1.0).astype(np.float32)
