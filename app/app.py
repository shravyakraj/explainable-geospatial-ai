"""Main Streamlit Application: Explainable AI Based Deforestation Anomaly Detection."""

import logging
from pathlib import Path
import sys

# Resolve project paths and configure import search paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = str(PROJECT_ROOT / "app")
SRC_DIR = str(PROJECT_ROOT / "src")

# 1. Ensure project root and src/ are on sys.path so 'app' and 'geospatial_ai' are importable
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# 2. When Streamlit executes 'streamlit run app/app.py', it inserts the script's
# parent directory ('.../app') into sys.path[0]. Because 'app/app.py' resides in
# that directory, Python resolves 'import app' to the single-file module 'app.py'
# rather than the package 'app' (the directory containing __init__.py).
# We purge APP_DIR from sys.path so 'app' always resolves to the root package.
while APP_DIR in sys.path:
    sys.path.remove(APP_DIR)

# 3. If 'app' was already erroneously cached as a non-package module in sys.modules,
# evict it so Python imports the proper 'app' package.
if "app" in sys.modules and not hasattr(sys.modules["app"], "__path__"):
    del sys.modules["app"]

import streamlit as st

from app.components.explainability_view import render_explainability_view
from app.components.header import render_header
from app.components.patch_selector import render_patch_selector
from app.components.prediction_card import render_prediction_card
from app.inference.predictor import run_inference_and_explain
from app.utils.data_loader import load_patch_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("streamlit_app")


def main():
    st.set_page_config(
        page_title="Explainable Geospatial AI | Deforestation Detection",
        page_icon="🌲",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Clean academic styling
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1350px;
        }
        h1, h2, h3, h4 {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            color: #1e293b;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 12px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 42px;
            font-size: 0.95rem;
            font-weight: 600;
            padding: 0 16px;
            border-radius: 6px;
        }
        code {
            color: #0f172a;
            background-color: #f1f5f9;
            padding: 0.15rem 0.35rem;
            border-radius: 4px;
            font-size: 0.88em;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # 1. Header
    render_header()

    # 2. Sidebar Patch Selector
    try:
        selected_record, reveal_gt = render_patch_selector()
    except Exception as e:
        st.error(f"Error loading patch manifest: {e}")
        logger.exception("Error in render_patch_selector")
        return

    # 3. Load Selected Patch Data
    try:
        t1, t2, ground_truth_label = load_patch_data(selected_record["absolute_path"])
    except Exception as e:
        st.error(f"Error loading patch file from `{selected_record.get('patch_file')}`: {e}")
        logger.exception("Error loading patch data")
        return

    # 4. Execute Production Inference & Explainability
    try:
        with st.spinner("Executing model inference & computing bi-temporal Grad-CAM..."):
            explanation_data = run_inference_and_explain(t1, t2)
    except Exception as e:
        st.error(f"Inference error on patch `{selected_record.get('patch_id')}`: {e}")
        logger.exception("Error during run_inference_and_explain")
        return

    # 5. Render Prediction Decision Card
    render_prediction_card(
        prediction_result=explanation_data["prediction"],
        ground_truth_label=ground_truth_label,
        reveal_ground_truth=reveal_gt,
    )

    # 6. Render Explainability & Attribution Panels
    render_explainability_view(
        explanation_data=explanation_data,
        t1_patch=t1,
        t2_patch=t2,
    )

    # 7. Footer & System Provenance
    st.markdown("---")
    st.markdown(
        """
        <div style="font-size: 0.82rem; color: #64748b; text-align: center; margin-top: 1rem;">
            <strong>Explainable AI Based Deforestation Anomaly Detection System</strong> | 
            Production Model: <code>MobileNetV3-Small Siamese</code> (93.33% Test Acc, 100% Recall) | 
            Framework: <code>TensorFlow 2.16 + Keras 3.15</code> | 
            Apple Silicon M3 Hardware Acceleration Enabled
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
