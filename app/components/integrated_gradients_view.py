"""Streamlit view component for Integrated Gradients dense spatial attribution."""

from typing import Any, Dict, Optional
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from app.inference.predictor import run_integrated_gradients_explanation
from app.visualization.image_renderers import create_integrated_gradients_figure


def render_integrated_gradients_view(
    x_8channel: np.ndarray,
    w_mask: np.ndarray,
    y_ref: Optional[np.ndarray] = None,
    patch_id: Optional[str] = None,
):
    """Render dense Integrated Gradients feature attributions for segmentation."""
    st.markdown(
        """
        <div style="margin-top: 1.2rem; margin-bottom: 1.0rem;">
            <h2 style="font-size: 1.35rem; font-weight: 700; color: #F0F6FC; margin-top: 0; margin-bottom: 0.35rem; display: flex; align-items: center; gap: 0.5rem;">
                <span style="color: #10B981;">•</span> Integrated Gradients Dense Spatial Attribution
            </h2>
            <p style="font-size: 0.90rem; color: #94A3B8; line-height: 1.6; margin: 0; max-width: 920px;">
                Axiomatic path-integrated attribution (Sundararajan et al., 2017) attributes the U-Net's dense predictions back
                to each spectral band across both observation dates.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    ig_key = f"ig_result_{patch_id}"

    # Controls
    col_c1, col_c2 = st.columns([1, 2])
    with col_c1:
        num_steps = st.selectbox(
            "Integration Step Count (m):",
            [50, 25, 20],
            index=0,
            help="Higher steps improve axiomatic convergence completeness at minimal extra computation.",
        )
    with col_c2:
        st.markdown(
            """
            <div style="background-color: #111720; border: 1px solid #1F2937; border-radius: 6px; padding: 0.75rem 1.0rem; font-size: 0.85rem; color: #94A3B8; line-height: 1.55;">
                <strong style="color: #10B981;">Baseline:</strong> Zero normalized reflectance (<em>x₀ = 0</em>, absence of optical signal).<br>
                <strong style="color: #10B981;">Target Objective S(X):</strong> Total expected deforestation area over valid observation pixels (<em>W = 1</em>).
            </div>
            """,
            unsafe_allow_html=True,
        )

    if st.button("Compute Integrated Gradients Attribution", key=f"btn_{ig_key}"):
        with st.spinner(f"Computing {num_steps} path-integrated backpropagation steps on Apple Silicon M3..."):
            try:
                ig_res = run_integrated_gradients_explanation(
                    x_8channel=x_8channel,
                    w_mask=w_mask,
                    patch_id=patch_id or "patch_sample",
                    y_ref=y_ref,
                    num_steps=num_steps,
                )
                st.session_state[ig_key] = ig_res
            except Exception as e:
                st.error(f"Integrated Gradients computation failed: {e}")
                st.session_state[ig_key] = None

    ig_result = st.session_state.get(ig_key, None)

    if ig_result is not None:
        # Completeness & Convergence Verification Cards
        st.markdown(
            f"""
            <div style="display: flex; gap: 1.0rem; flex-wrap: wrap; margin-top: 1.0rem; margin-bottom: 1.2rem;">
                <div style="background-color: #111720; border: 1px solid #1F2937; border-radius: 6px; padding: 0.7rem 0.9rem; flex: 1; min-width: 170px;">
                    <div style="font-size: 0.72rem; text-transform: uppercase; color: #10B981; font-weight: 700; letter-spacing: 0.05em;">Target Output S(X)</div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #F0F6FC; margin-top: 0.2rem;">{ig_result.target_scalar_value:.1f} px <span style="font-size: 0.75rem; color: #64748B;">({ig_result.target_scalar_value*0.01:.1f} ha)</span></div>
                </div>
                <div style="background-color: #111720; border: 1px solid #1F2937; border-radius: 6px; padding: 0.7rem 0.9rem; flex: 1; min-width: 170px;">
                    <div style="font-size: 0.72rem; text-transform: uppercase; color: #10B981; font-weight: 700; letter-spacing: 0.05em;">Baseline Output S(x₀)</div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #F0F6FC; margin-top: 0.2rem;">{ig_result.baseline_scalar_value:.1f} px</div>
                </div>
                <div style="background-color: #111720; border: 1px solid #1F2937; border-radius: 6px; padding: 0.7rem 0.9rem; flex: 1; min-width: 170px;">
                    <div style="font-size: 0.72rem; text-transform: uppercase; color: #38BDF8; font-weight: 700; letter-spacing: 0.05em;">Net Attribution (Σ IG)</div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: {'#FCA5A5' if ig_result.sum_attributions > 0 else '#60A5FA'}; margin-top: 0.2rem;">{ig_result.sum_attributions:+.1f}</div>
                </div>
                <div style="background-color: #111720; border: 1px solid #1F2937; border-radius: 6px; padding: 0.7rem 0.9rem; flex: 1; min-width: 170px;">
                    <div style="font-size: 0.72rem; text-transform: uppercase; color: #34D399; font-weight: 700; letter-spacing: 0.05em;">Completeness Error</div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #6EE7B7; margin-top: 0.2rem;">{ig_result.completeness_relative_error:.3%} <span style="font-size: 0.75rem; color: #64748B;">(Axiom verified)</span></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Render 4-panel figure
        fig_ig = create_integrated_gradients_figure(
            spatial_attr=ig_result.spatial_attribution,
            t1_attr=ig_result.t1_spatial_attribution,
            t2_attr=ig_result.t2_spatial_attribution,
            w_mask=w_mask,
            per_band_attributions=ig_result.per_band_attributions,
            completeness_error=ig_result.completeness_relative_error,
        )
        st.pyplot(fig_ig)
        plt.close(fig_ig)

        # Biophysical Remote Sensing Narrative
        t1_b08 = ig_result.per_band_attributions.get("B08", {}).get("t1", 0.0)
        t2_b04 = ig_result.per_band_attributions.get("B04", {}).get("t2", 0.0)
        t2_b08 = ig_result.per_band_attributions.get("B08", {}).get("t2", 0.0)

        st.markdown(
            f"""
            <div style="background-color: #111720; border: 1px solid #1F2937; border-left: 4px solid #10B981; border-radius: 6px; padding: 0.9rem 1.2rem; margin-top: 1.0rem; margin-bottom: 1.2rem; font-size: 0.88rem; color: #94A3B8; line-height: 1.65;">
                <div style="color: #6EE7B7; font-weight: 700; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">Biophysical Remote Sensing Interpretation</div>
                • <strong style="color: #F0F6FC;">Initial Canopy Sensitivity (T1 B08 NIR: {t1_b08:+.1f}):</strong> {'Strong positive attribution confirms high pre-existing canopy is necessary to trigger a deforestation prediction.' if t1_b08 > 0 else 'Negative or low attribution reflects stable non-loss terrain.'}<br>
                • <strong style="color: #F0F6FC;">Post-Clearing Soil Exposure (T2 B04 Red: {t2_b04:+.1f}):</strong> {'High positive attribution indicates bare soil exposure post-clearing strongly elevated model confidence.' if t2_b04 > 100 else 'Low red sensitivity indicates absence of bare ground disturbance.'}<br>
                • <strong style="color: #F0F6FC;">Vegetation Retention Suppression (T2 B08 NIR: {t2_b08:+.1f}):</strong> {'Negative attribution confirms healthy green canopy at T2 actively suppresses the deforestation prediction.' if t2_b08 < 0 else 'Positive NIR contribution reflects subtle canopy stress.'}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Scientific limitations note
    st.info(
        "**Methodological Caveat:** Integrated Gradients quantifies directional model sensitivity along a linear interpolation "
        "trajectory from zero reflectance. It identifies which spectral inputs drove the neural network's mathematical output, "
        "but does NOT establish ecological causality, biological ground truth, or legal responsibility."
    )
