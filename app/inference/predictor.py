"""Streamlit cached predictor interface for Production Deforestation Classifier."""

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import streamlit as st

from geospatial_ai.inference import ProductionDeforestationClassifier

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@st.cache_resource(show_spinner="Initializing Production MobileNetV3 Siamese Model...")
def get_classifier() -> ProductionDeforestationClassifier:
    """Load and cache singleton instance of production deforestation classifier."""
    config_path = PROJECT_ROOT / "configs" / "production_model.yaml"
    model_path = PROJECT_ROOT / "models" / "champion_phase2d_mobilenetv3.keras"
    if not model_path.exists():
        model_path = PROJECT_ROOT / "experiments" / "phase2d" / "checkpoint_fam-a2.keras"

    logger.info("Initializing cached classifier from %s", model_path)
    classifier = ProductionDeforestationClassifier(
        config_path=config_path,
        model_path=model_path,
    )
    # Warmup model graph
    _ = classifier.get_model()
    return classifier


def run_inference_and_explain(
    t1_patch: np.ndarray,
    t2_patch: np.ndarray,
) -> Dict[str, Any]:
    """Execute prediction and explainability through cached production classifier."""
    classifier = get_classifier()
    return classifier.explain(t1_patch, t2_patch)
