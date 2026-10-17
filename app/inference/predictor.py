"""Streamlit cached predictor interface for Production Deforestation Classifier and Segmentation Model."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import streamlit as st
import tensorflow as tf

from geospatial_ai.inference import ProductionDeforestationClassifier
from geospatial_ai.lime_explainability import BiTemporalLimeExplainer, LimeExplanationResult
from geospatial_ai.segmentation_explainability import (
    IntegratedGradientsConfig,
    SegmentationExplanationResult,
    compute_integrated_gradients,
)

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

CHAMPION_CLASSIFIER_PATH = PROJECT_ROOT / "models" / "champion_phase2d_mobilenetv3.keras"
CHAMPION_SEGMENTATION_PATH = PROJECT_ROOT / "models" / "phase5_unet_champion.keras"
PRODUCTION_CONFIG_PATH = PROJECT_ROOT / "configs" / "production_model.yaml"


@st.cache_resource(show_spinner="Initializing Production MobileNetV3 Siamese Model...")
def get_classifier() -> ProductionDeforestationClassifier:
    """Load and cache singleton instance of production deforestation classifier."""
    model_path = CHAMPION_CLASSIFIER_PATH
    if not model_path.exists():
        model_path = PROJECT_ROOT / "experiments" / "phase2d" / "checkpoint_fam-a2.keras"

    logger.info("Initializing cached classifier from %s", model_path)
    classifier = ProductionDeforestationClassifier(
        config_path=PRODUCTION_CONFIG_PATH,
        model_path=model_path,
    )
    # Warmup model graph
    _ = classifier.get_model()
    return classifier


@st.cache_resource(show_spinner="Initializing Phase 5B Early-Fusion U-Net Champion...")
def get_segmentation_model() -> tf.keras.Model:
    """Load and cache singleton instance of Phase 5B Early-Fusion U-Net."""
    if not CHAMPION_SEGMENTATION_PATH.exists():
        raise FileNotFoundError(f"Segmentation champion not found at: {CHAMPION_SEGMENTATION_PATH}")
    logger.info("Loading cached segmentation model from %s", CHAMPION_SEGMENTATION_PATH)
    model = tf.keras.models.load_model(CHAMPION_SEGMENTATION_PATH, compile=False)
    # Warmup forward pass
    dummy = tf.zeros((1, 256, 256, 8), dtype=tf.float32)
    _ = model(dummy, training=False)
    return model


@st.cache_resource(show_spinner="Initializing Bi-Temporal LIME Explainer...")
def get_lime_explainer() -> BiTemporalLimeExplainer:
    """Load and cache singleton instance of bi-temporal LIME explainer."""
    classifier = get_classifier()
    return BiTemporalLimeExplainer(
        classifier=classifier,
        random_state=42,
    )


def run_inference_and_explain(
    t1_patch: np.ndarray,
    t2_patch: np.ndarray,
) -> Dict[str, Any]:
    """Execute classification prediction and Grad-CAM explainability."""
    classifier = get_classifier()
    return classifier.explain(t1_patch, t2_patch)


def run_lime_explanation(
    t1_patch: np.ndarray,
    t2_patch: np.ndarray,
    target_date: str = "t2",
    patch_id: Optional[str] = None,
    num_samples: int = 100,
) -> LimeExplanationResult:
    """Execute bi-temporal counterfactual LIME explanation with error handling."""
    explainer = get_lime_explainer()
    logger.info("Running LIME explanation for target_date=%s on patch=%s", target_date, patch_id)
    return explainer.explain_patch(
        t1_patch=t1_patch,
        t2_patch=t2_patch,
        target_date=target_date,
        patch_id=patch_id,
        num_samples=num_samples,
    )


def run_segmentation_inference(
    x_8channel: np.ndarray,
    threshold: float = 0.500,
    w_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Execute dense U-Net segmentation forward pass and area aggregation."""
    model = get_segmentation_model()
    if x_8channel.ndim == 3:
        x_input = x_8channel[np.newaxis, ...]
    else:
        x_input = x_8channel

    # Inference forward pass
    preds = model(x_input, training=False).numpy()[0]  # Shape: (256, 256, 1)

    prob_map = preds[..., 0]
    binary_pred = (prob_map >= threshold).astype(np.uint8)

    if w_mask is not None:
        w_2d = w_mask[..., 0] > 0.5
        binary_pred_valid = binary_pred & w_2d
        total_valid_pixels = int(np.sum(w_2d))
        predicted_loss_pixels = int(np.sum(binary_pred_valid))
    else:
        binary_pred_valid = binary_pred
        total_valid_pixels = int(256 * 256)
        predicted_loss_pixels = int(np.sum(binary_pred))

    # Area conversions: 1 pixel at 10m = 100 m^2 = 0.01 ha
    predicted_loss_ha = round(predicted_loss_pixels * 0.01, 2)
    valid_area_ha = round(total_valid_pixels * 0.01, 2)

    return {
        "probability_map": prob_map,
        "binary_mask": binary_pred,
        "binary_mask_valid": binary_pred_valid,
        "decision_threshold": threshold,
        "total_valid_pixels": total_valid_pixels,
        "predicted_loss_pixels": predicted_loss_pixels,
        "predicted_loss_ha": predicted_loss_ha,
        "valid_area_ha": valid_area_ha,
        "loss_fraction_of_valid": round((predicted_loss_pixels / max(1, total_valid_pixels)), 4),
    }


def run_integrated_gradients_explanation(
    x_8channel: np.ndarray,
    w_mask: np.ndarray,
    patch_id: str = "patch_sample",
    y_ref: Optional[np.ndarray] = None,
    num_steps: int = 50,
) -> SegmentationExplanationResult:
    """Execute Integrated Gradients dense attribution on the segmentation champion."""
    model = get_segmentation_model()
    config = IntegratedGradientsConfig(
        num_steps=num_steps,
        batch_step=25,
        baseline_type="zero",
        target_type="valid_sum",
        decision_threshold=0.500,
        quadrature="midpoint",
    )
    return compute_integrated_gradients(
        model=model,
        x=x_8channel,
        w=w_mask,
        config=config,
        y_ref=y_ref,
        patch_id=patch_id,
    )
