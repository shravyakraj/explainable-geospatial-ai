"""Production Inference and Explainability Contract for Bi-Temporal Deforestation Detection."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import keras
import numpy as np
import tensorflow as tf
import yaml

from geospatial_ai.explainability import (
    compute_gradcam,
    compute_ndvi,
    create_false_color_composite,
    create_rgb_composite,
    overlay_heatmap_on_image,
)
from geospatial_ai.model import AbsoluteDifference, SignedDifference

logger = logging.getLogger(__name__)

CUSTOM_OBJECTS = {
    "AbsoluteDifference": AbsoluteDifference,
    "SignedDifference": SignedDifference,
}


def compute_derived_change_explanation(
    cam_t1: np.ndarray,
    cam_t2: np.ndarray,
) -> np.ndarray:
    """Compute derived relative salience map highlighting spatial regions where positive model activation increased.

    Scientific distinction:
    - cam_t1: model activation on baseline scene (2020-08-08)
    - cam_t2: model activation on monitoring scene (2021-08-08)
    - derived_change: relative directional salience difference ReLU(cam_t2 - cam_t1).
      This indicates qualitative spatial attribution where model salience shifted, not an absolute
      decision weight, calibrated probability, or physical segmentation boundary.
    """
    diff = cam_t2 - cam_t1
    pos_diff = np.maximum(diff, 0.0)
    max_val = np.max(pos_diff)
    if max_val > 1e-8:
        derived = pos_diff / max_val
    else:
        derived = np.zeros_like(pos_diff)
    return np.clip(derived, 0.0, 1.0)


def generate_scientific_explanation_text(
    probability: float,
    threshold: float,
    delta_ndvi: np.ndarray,
) -> str:
    """Generate plain-English evidence narrative based on satellite observations."""
    is_deforest = probability >= threshold
    pct_loss_area = float(np.mean(delta_ndvi < -0.15) * 100.0)

    if is_deforest:
        explanation = (
            f"The two satellite images show a noticeable change in this area between 2020 and 2021. "
            f"The highlighted part of the image is where the strongest signs of forest-cover change appear. "
            f"In the later image, about {pct_loss_area:.1f}% of the patch shows a noticeable drop in the vegetation signal."
        )
    else:
        explanation = (
            "The two satellite images do not show strong enough signs of forest clearing to classify this area as a deforestation anomaly. "
            "The areas that mattered most for the prediction look broadly similar between the two dates."
        )

    return explanation


class ProductionDeforestationClassifier:
    """Production wrapper for locked Phase 2D deforestation detection model."""

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        model_path: Optional[Union[str, Path]] = None,
        threshold: Optional[float] = None,
    ):
        if config_path is None:
            config_path = Path(__file__).resolve().parent.parent.parent / "configs" / "production_model.yaml"
        self.config_path = Path(config_path)

        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}

        if model_path is None:
            configured_ckpt = self.config.get("model_metadata", {}).get("checkpoint_path")
            if configured_ckpt:
                model_path = Path(__file__).resolve().parent.parent.parent / configured_ckpt
            else:
                model_path = Path(__file__).resolve().parent.parent.parent / "models" / "champion_phase2d_mobilenetv3.keras"

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            exp_ckpt = Path(__file__).resolve().parent.parent.parent / "experiments" / "phase2d" / "checkpoint_fam-a2.keras"
            if exp_ckpt.exists():
                self.model_path = exp_ckpt

        if threshold is not None:
            self.threshold = float(threshold)
        else:
            self.threshold = float(self.config.get("decision_contract", {}).get("decision_threshold", 0.2700))

        self.model_version = self.config.get("model_metadata", {}).get("model_version", "2.4.0-phase2d")
        self.model_name = self.config.get("model_metadata", {}).get("model_name", "Pretrained_MobileNetV3_Frozen")

        self._model: Optional[keras.Model] = None

    def get_model(self) -> keras.Model:
        """Lazy load and cache production model."""
        if self._model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Production model checkpoint not found at: {self.model_path}")
            logger.info("Loading production model from %s", self.model_path)
            self._model = keras.models.load_model(
                str(self.model_path),
                custom_objects=CUSTOM_OBJECTS,
                safe_mode=False,
            )
        return self._model

    def predict(
        self,
        t1_patch: np.ndarray,
        t2_patch: np.ndarray,
    ) -> Dict[str, Any]:
        """Perform bi-temporal classification on a single patch pair.

        Args:
            t1_patch: Earlier Sentinel-2 patch of shape (256, 256, 4), values in [0, 1].
            t2_patch: Later Sentinel-2 patch of shape (256, 256, 4), values in [0, 1].

        Returns:
            Structured prediction dictionary matching the inference contract.
        """
        if t1_patch.shape != (256, 256, 4) or t2_patch.shape != (256, 256, 4):
            raise ValueError(
                f"Expected patch dimensions (256, 256, 4), received T1: {t1_patch.shape}, T2: {t2_patch.shape}"
            )

        model = self.get_model()
        t1_batch = t1_patch[np.newaxis, ...]
        t2_batch = t2_patch[np.newaxis, ...]

        prob_tensor = model.predict({"input_t1": t1_batch, "input_t2": t2_batch}, batch_size=1, verbose=0)
        probability = float(prob_tensor[0, 0])

        is_deforest = bool(probability >= self.threshold)
        pred_label = "Deforestation" if is_deforest else "Intact Forest"
        pred_class = 1 if is_deforest else 0

        # Decision margin: normalized distance from decision threshold
        # Explicit definition: Decision margin is a derived distance from the classification threshold
        # and is not a calibrated probability of prediction correctness.
        if probability >= self.threshold:
            decision_margin = (probability - self.threshold) / (1.0 - self.threshold)
        else:
            decision_margin = (self.threshold - probability) / self.threshold
        decision_margin = float(np.clip(decision_margin, 0.0, 1.0))

        return {
            "prediction": pred_label,
            "predicted_class": pred_class,
            "probability": round(probability, 4),
            "decision_margin": round(decision_margin, 4),
            "threshold": round(self.threshold, 4),
            "model_name": self.model_name,
            "model_version": self.model_version,
            "input_shape": [256, 256, 4],
            "bands": ["B02", "B03", "B04", "B08"],
        }

    def explain(
        self,
        t1_patch: np.ndarray,
        t2_patch: np.ndarray,
    ) -> Dict[str, Any]:
        """Compute full bi-temporal inference, Grad-CAM heatmaps, spectral indices, and overlays.

        Args:
            t1_patch: Earlier Sentinel-2 patch of shape (256, 256, 4).
            t2_patch: Later Sentinel-2 patch of shape (256, 256, 4).

        Returns:
            Dictionary containing prediction, Grad-CAM heatmaps, spectral products, and overlays.
        """
        pred_res = self.predict(t1_patch, t2_patch)
        model = self.get_model()

        # Compute Grad-CAM heatmaps (Option B: pre-normalization temporal change map)
        cam_t1, cam_t2, change_exp, _ = compute_gradcam(model, t1_patch, t2_patch, return_change=True)

        # Spectral products
        rgb_t1 = create_rgb_composite(t1_patch)
        rgb_t2 = create_rgb_composite(t2_patch)
        cir_t1 = create_false_color_composite(t1_patch)
        cir_t2 = create_false_color_composite(t2_patch)

        ndvi_t1 = compute_ndvi(t1_patch)
        ndvi_t2 = compute_ndvi(t2_patch)
        delta_ndvi = ndvi_t2 - ndvi_t1
        rgb_diff = np.clip(np.abs(rgb_t2 - rgb_t1) * 1.5, 0.0, 1.0)

        # Heatmap overlays
        overlay_t1 = overlay_heatmap_on_image(rgb_t1, cam_t1, alpha=0.55, colormap="magma")
        overlay_t2 = overlay_heatmap_on_image(rgb_t2, cam_t2, alpha=0.55, colormap="magma")
        overlay_change = overlay_heatmap_on_image(rgb_t2, change_exp, alpha=0.60, colormap="plasma")

        # Scientific explanation text
        explanation_text = generate_scientific_explanation_text(
            probability=pred_res["probability"],
            threshold=self.threshold,
            delta_ndvi=delta_ndvi,
        )

        return {
            "prediction": pred_res,
            "cam_t1": cam_t1,
            "cam_t2": cam_t2,
            "change_explanation": change_exp,
            "rgb_t1": rgb_t1,
            "rgb_t2": rgb_t2,
            "cir_t1": cir_t1,
            "cir_t2": cir_t2,
            "ndvi_t1": ndvi_t1,
            "ndvi_t2": ndvi_t2,
            "delta_ndvi": delta_ndvi,
            "rgb_diff": rgb_diff,
            "overlay_t1": overlay_t1,
            "overlay_t2": overlay_t2,
            "overlay_change": overlay_change,
            "explanation_text": explanation_text,
        }
