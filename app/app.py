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

# 2. Purge APP_DIR from sys.path so 'app' resolves to the package directory
while APP_DIR in sys.path:
    sys.path.remove(APP_DIR)

# 3. If 'app' was already cached as a non-package module in sys.modules, evict it
if "app" in sys.modules and not hasattr(sys.modules["app"], "__path__"):
    del sys.modules["app"]

import streamlit as st

from app.components.explainability_view import render_explainability_view
from app.components.header import render_header
from app.components.patch_selector import render_patch_selector
from app.components.prediction_card import render_prediction_card
from app.inference.predictor import run_inference_and_explain
from app.utils.data_loader import load_patch_data
from app.visualization.image_renderers import array_to_pil_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("streamlit_app")


def main():
    st.set_page_config(
        page_title="Explainable AI Based Deforestation Anomaly Detection Using Satellite Images",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Soft salmon-pink × lilac atmospheric editorial stylesheet
    st.markdown(
        """
        <style>
        /* Overall Page & Background: subtle salmon-to-lilac atmospheric gradient */
        .stApp {
            background: linear-gradient(155deg, #FCE9E7 0%, #FAF1F3 45%, #EEE8F8 100%);
            background-attachment: fixed;
            color: #2B2D31;
        }
        .main .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2.5rem;
            padding-left: 2rem;
            padding-right: 2rem;
            max-width: 1240px;
        }
        
        /* Typography: Clean, editorial, highly legible */
        h1, h2, h3, h4, p, span, label, div {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            color: #2B2D31;
        }
        h1 {
            font-size: 2.0rem;
            font-weight: 800;
            color: #2B2D31;
            letter-spacing: -0.02em;
            margin-bottom: 0.2rem;
        }
        h2 {
            font-size: 1.45rem;
            font-weight: 800;
            color: #2B2D31;
            letter-spacing: -0.01em;
            margin-top: 0;
            margin-bottom: 0.35rem;
        }
        h3 {
            font-size: 1.15rem;
            font-weight: 800;
            color: #2B2D31;
            margin-top: 0.4rem;
            margin-bottom: 0.25rem;
        }
        
        /* Pale Dusty-Lilac / Blush Editorial Sidebar (No Green) */
        section[data-testid="stSidebar"] {
            background-color: #F3ECF5;
            border-right: 1.5px solid #E4D9EB;
        }
        section[data-testid="stSidebar"] .block-container {
            padding-top: 1.4rem;
            padding-left: 1.2rem;
            padding-right: 1.2rem;
        }
        
        /* Monospace / Code pills */
        code {
            color: #2B2D31;
            background-color: #EBE3F9;
            padding: 0.15rem 0.4rem;
            border-radius: 3px;
            font-size: 0.85em;
            font-family: SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        }

        /* Large, prominent satellite images with subtle editorial frames */
        div[data-testid="stImage"] img {
            border: 1.5px solid #E5DBF5;
            border-radius: 4px;
            box-shadow: none;
            display: block;
            width: 100%;
        }

        /* Expander Styling */
        div[data-testid="stExpander"] {
            border: 1.5px solid #E5DBF5;
            border-radius: 4px;
            background-color: #FAF8F5;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # 1. Project Title (Clean, nothing underneath)
    render_header()

    # 2. Section: Assessment of forest change over time (Breathing on atmospheric background)
    st.markdown(
        """
        <div style="margin-top: 0.4rem; margin-bottom: 1.8rem;">
            <h2 style="font-size: 1.45rem; font-weight: 800; color: #2B2D31; margin-top: 0; margin-bottom: 0.35rem; letter-spacing: -0.01em;">
                Assessment of forest change over time
            </h2>
            <p style="font-size: 0.96rem; color: #4A4D53; line-height: 1.6; margin: 0; max-width: 920px;">
                Compare satellite images from two different observation dates to see whether a deforestation anomaly was detected and explore the visual evidence behind the result.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3. Sidebar Patch Selector
    try:
        selected_record, reveal_gt = render_patch_selector()
    except Exception as e:
        st.error(f"Error loading patch manifest: {e}")
        logger.exception("Error in render_patch_selector")
        return

    # 4. Load Selected Patch Data
    try:
        t1, t2, ground_truth_label = load_patch_data(selected_record["absolute_path"])
    except Exception as e:
        st.error(f"Error loading patch data: {e}")
        logger.exception("Error loading patch data")
        return

    # 5. Execute Production Inference & Explainability
    try:
        with st.spinner("Analyzing satellite images..."):
            explanation_data = run_inference_and_explain(t1, t2)
    except Exception as e:
        st.error(f"Inference error on selected patch: {e}")
        logger.exception("Error during run_inference_and_explain")
        return

    # 6. Section: Satellite comparison
    st.markdown(
        """
        <div style="margin-top: 1.6rem; margin-bottom: 1.0rem;">
            <h2 style="font-size: 1.45rem; font-weight: 800; color: #2B2D31; margin-top: 0; margin-bottom: 0.35rem; letter-spacing: -0.01em;">
                Satellite comparison
            </h2>
            <p style="font-size: 0.96rem; color: #4A4D53; line-height: 1.5; margin: 0; max-width: 920px;">
                High-resolution Sentinel-2 true-color surface reflectance (10m per pixel) over a 2.56 × 2.56 km Amazon study area.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.markdown(
            "<div style='font-size: 0.92rem; font-weight: 800; color: #2B2D31; margin-bottom: 0.45rem; letter-spacing: 0.03em;'>BEFORE · 08 AUG 2020</div>",
            unsafe_allow_html=True,
        )
        st.image(
            array_to_pil_image(explanation_data["rgb_t1"]),
            caption="Sentinel-2 True Color (Red, Green, Blue · 10m resolution)",
            use_column_width=True,
        )

    with col_img2:
        st.markdown(
            "<div style='font-size: 0.92rem; font-weight: 800; color: #2B2D31; margin-bottom: 0.45rem; letter-spacing: 0.03em;'>AFTER · 08 AUG 2021</div>",
            unsafe_allow_html=True,
        )
        st.image(
            array_to_pil_image(explanation_data["rgb_t2"]),
            caption="Sentinel-2 True Color (Red, Green, Blue · 10m resolution)",
            use_column_width=True,
        )

    # 7. Dramatic Result banner in Salmon/Lilac family + PRODES check
    render_prediction_card(
        prediction_result=explanation_data["prediction"],
        ground_truth_label=ground_truth_label,
        reveal_ground_truth=reveal_gt,
    )

    # 8. Where the change stands out + Change analysis + Why this prediction was made + Automated narrative
    render_explainability_view(
        explanation_data=explanation_data,
        t1_patch=t1,
        t2_patch=t2,
    )

    # 9. Centered Playful Footer (ONLY THIS, NOTHING ELSE)
    st.markdown(
        """
        <div style="text-align: center; margin-top: 3.5rem; margin-bottom: 2rem; padding-top: 1.5rem; border-top: 1px solid #EBE3F9; font-size: 0.8rem; color: #7B818C;">
            did this instead of sleeping, lol
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
