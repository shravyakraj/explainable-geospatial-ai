"""Streamlit view component for Local Interpretable Model-agnostic Explanations (LIME)."""

from typing import Any, Dict, Optional
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from app.inference.predictor import run_lime_explanation
from app.visualization.image_renderers import array_to_pil_image, create_lime_figure


def render_lime_view(
    t1_patch: np.ndarray,
    t2_patch: np.ndarray,
    rgb_t1: np.ndarray,
    rgb_t2: np.ndarray,
    patch_id: Optional[str] = None,
):
    """Render bi-temporal counterfactual LIME local saliency view."""
    st.markdown(
        """
        <div style="margin-top: 1.2rem; margin-bottom: 1.0rem;">
            <h2 style="font-size: 1.35rem; font-weight: 700; color: #F0F6FC; margin-top: 0; margin-bottom: 0.35rem; display: flex; align-items: center; gap: 0.5rem;">
                <span style="color: #10B981;">•</span> Local Interpretable Model-agnostic Explanations (LIME)
            </h2>
            <p style="font-size: 0.90rem; color: #94A3B8; line-height: 1.6; margin: 0; max-width: 920px;">
                LIME fits a local linear surrogate model (Ridge regression) over stochastic superpixel perturbations to explain the
                production Siamese MobileNetV3 classifier's decision.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Explanation direction selector
    col_ctrl1, col_ctrl2 = st.columns([2, 3])
    with col_ctrl1:
        date_choice = st.radio(
            "Select Counterfactual Explanation Date:",
            ["T2 (Monitoring Scene · 08 Aug 2021)", "T1 (Baseline Scene · 08 Aug 2020)"],
            index=0,
            help="Choose which temporal observation to perturb while holding the counterpart scene constant.",
        )
        target_date = "t2" if "T2" in date_choice else "t1"

    with col_ctrl2:
        st.markdown(
            f"""
            <div style="background-color: #111720; border: 1px solid #1F2937; border-radius: 6px; padding: 0.85rem 1.1rem; font-size: 0.85rem; color: #94A3B8; line-height: 1.6;">
                <strong style="color: #10B981;">Temporal Counterfactual Logic:</strong><br>
                {"When explaining T2, deactivated superpixels are substituted with their corresponding 2020 baseline state. This models: <em>'What would the classifier predict if this region had remained undisturbed?'</em>"
                 if target_date == 't2' else
                 "When explaining T1, deactivated superpixels are substituted with their 2021 condition. This models: <em>'What would the classifier predict if this region had already exhibited its 2021 condition in 2020?'</em>"}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Cache key for session state
    lime_key = f"lime_result_{patch_id}_{target_date}"

    if st.button("Generate LIME Local Explanation", key=f"btn_{lime_key}"):
        with st.spinner(f"Computing 100 superpixel perturbation evaluations for {target_date.upper()}..."):
            try:
                lime_res = run_lime_explanation(
                    t1_patch=t1_patch,
                    t2_patch=t2_patch,
                    target_date=target_date,
                    patch_id=patch_id,
                    num_samples=100,
                )
                st.session_state[lime_key] = lime_res
            except Exception as e:
                st.error(f"LIME computation encountered an error: {e}")
                st.session_state[lime_key] = None

    lime_result = st.session_state.get(lime_key, None)

    if lime_result is not None:
        # Display surrogate quality metrics
        st.markdown(
            f"""
            <div style="display: flex; gap: 1.0rem; flex-wrap: wrap; margin-top: 1.0rem; margin-bottom: 1.2rem;">
                <div style="background-color: #111720; border: 1px solid #1F2937; border-radius: 6px; padding: 0.7rem 0.9rem; flex: 1; min-width: 170px;">
                    <div style="font-size: 0.72rem; text-transform: uppercase; color: #10B981; font-weight: 700; letter-spacing: 0.05em;">Target Observation</div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #F0F6FC; margin-top: 0.2rem;">{target_date.upper()} ({'2021-08-08' if target_date=='t2' else '2020-08-08'})</div>
                </div>
                <div style="background-color: #111720; border: 1px solid #1F2937; border-radius: 6px; padding: 0.7rem 0.9rem; flex: 1; min-width: 170px;">
                    <div style="font-size: 0.72rem; text-transform: uppercase; color: #34D399; font-weight: 700; letter-spacing: 0.05em;">Surrogate Local R² Score</div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #6EE7B7; margin-top: 0.2rem;">{lime_result.local_r2_score:.4f}</div>
                </div>
                <div style="background-color: #111720; border: 1px solid #1F2937; border-radius: 6px; padding: 0.7rem 0.9rem; flex: 1; min-width: 170px;">
                    <div style="font-size: 0.72rem; text-transform: uppercase; color: #38BDF8; font-weight: 700; letter-spacing: 0.05em;">Surrogate Intercept</div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #F0F6FC; margin-top: 0.2rem;">{lime_result.intercept:.4f}</div>
                </div>
                <div style="background-color: #111720; border: 1px solid #1F2937; border-radius: 6px; padding: 0.7rem 0.9rem; flex: 1; min-width: 170px;">
                    <div style="font-size: 0.72rem; text-transform: uppercase; color: #10B981; font-weight: 700; letter-spacing: 0.05em;">Superpixel Count</div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #F0F6FC; margin-top: 0.2rem;">{lime_result.num_segments} Segments</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Render 3-panel figure
        composite = rgb_t2 if target_date == "t2" else rgb_t1
        lime_fig = create_lime_figure(
            rgb_composite=composite,
            segments=lime_result.segments,
            supporting_mask=lime_result.supporting_mask,
            opposing_mask=lime_result.opposing_mask,
            top_features=lime_result.top_features,
            target_date=target_date.upper(),
        )
        st.pyplot(lime_fig)
        plt.close(lime_fig)

        # Automated narrative
        st.markdown(
            f"""
            <div style="background-color: #111720; border: 1px solid #1F2937; border-left: 4px solid #10B981; border-radius: 6px; padding: 0.9rem 1.2rem; margin-top: 1.0rem; margin-bottom: 1.2rem; font-size: 0.88rem; color: #94A3B8; line-height: 1.65;">
                <div style="color: #6EE7B7; font-weight: 700; font-size: 0.84rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.4rem;">Automated LIME Diagnostic Narrative</div>
                {lime_result.explanation_narrative}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Scientific limitations alert
    st.info(
        "**Methodological Caveat:** LIME feature attributions quantify local linear surrogate sensitivity under stochastic "
        "superpixel perturbations. They approximate a local temporal counterfactual and do NOT establish ecological causality, "
        "biological ground truth, or legally binding deforestation boundaries."
    )
