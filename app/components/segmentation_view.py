"""Streamlit view component for Early-Fusion U-Net deforestation segmentation."""

from typing import Any, Dict, Optional
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from app.inference.predictor import run_segmentation_inference
from app.visualization.image_renderers import create_segmentation_figure


def render_segmentation_view(
    x_8channel: np.ndarray,
    rgb_t2: np.ndarray,
    w_mask: Optional[np.ndarray] = None,
    patch_id: Optional[str] = None,
):
    """Render dense Early-Fusion U-Net pixel-level deforestation segmentation."""
    st.markdown(
        """
        <div style="margin-top: 1.2rem; margin-bottom: 1.0rem;">
            <h2 style="font-size: 1.35rem; font-weight: 700; color: #F0F6FC; margin-top: 0; margin-bottom: 0.35rem; display: flex; align-items: center; gap: 0.5rem;">
                <span style="color: #10B981;">•</span> Early-Fusion U-Net Pixel-Level Deforestation Segmentation
            </h2>
            <p style="font-size: 0.90rem; color: #94A3B8; line-height: 1.6; margin: 0; max-width: 920px;">
                The Phase 5B Early-Fusion U-Net processes all 8 bi-temporal Sentinel-2 spectral channels simultaneously to generate a
                dense probability map at native 10m pixel resolution (256 × 256 pixels = 655.36 ha).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Execute segmentation forward pass
    seg_result = run_segmentation_inference(
        x_8channel=x_8channel,
        threshold=0.500,  # Locked validation champion threshold
        w_mask=w_mask,
    )

    # Metrics summary cards
    pred_ha = seg_result["predicted_loss_ha"]
    valid_ha = seg_result["valid_area_ha"]
    total_patch_ha = 655.36
    excl_ha = round(total_patch_ha - valid_ha, 2)
    loss_pct = seg_result["loss_fraction_of_valid"] * 100.0

    st.markdown(
        f"""
        <div style="display: flex; gap: 1.0rem; flex-wrap: wrap; margin-top: 0.8rem; margin-bottom: 1.2rem;">
            <div style="background-color: #111720; border: 1px solid #1F2937; border-radius: 6px; padding: 0.8rem 1.0rem; flex: 1; min-width: 170px;">
                <div style="font-size: 0.72rem; text-transform: uppercase; color: #10B981; font-weight: 700; letter-spacing: 0.05em;">Locked Threshold (τ*)</div>
                <div style="font-size: 1.25rem; font-weight: 700; color: #F0F6FC; margin-top: 0.2rem;">0.500 <span style="font-size: 0.75rem; color: #64748B; font-weight: 500;">(Validation-locked)</span></div>
            </div>
            <div style="background-color: #111720; border: 1px solid #1F2937; border-radius: 6px; padding: 0.8rem 1.0rem; flex: 1; min-width: 170px;">
                <div style="font-size: 0.72rem; text-transform: uppercase; color: #F87171; font-weight: 700; letter-spacing: 0.05em;">Predicted Loss Area</div>
                <div style="font-size: 1.25rem; font-weight: 700; color: #FCA5A5; margin-top: 0.2rem;">{pred_ha:.2f} ha <span style="font-size: 0.75rem; color: #64748B; font-weight: 500;">({seg_result['predicted_loss_pixels']:,} px)</span></div>
            </div>
            <div style="background-color: #111720; border: 1px solid #1F2937; border-radius: 6px; padding: 0.8rem 1.0rem; flex: 1; min-width: 170px;">
                <div style="font-size: 0.72rem; text-transform: uppercase; color: #34D399; font-weight: 700; letter-spacing: 0.05em;">Valid Observation Area</div>
                <div style="font-size: 1.25rem; font-weight: 700; color: #6EE7B7; margin-top: 0.2rem;">{valid_ha:.2f} ha <span style="font-size: 0.75rem; color: #64748B; font-weight: 500;">({(valid_ha/total_patch_ha)*100:.1f}%)</span></div>
            </div>
            <div style="background-color: #111720; border: 1px solid #1F2937; border-radius: 6px; padding: 0.8rem 1.0rem; flex: 1; min-width: 170px;">
                <div style="font-size: 0.72rem; text-transform: uppercase; color: #38BDF8; font-weight: 700; letter-spacing: 0.05em;">Loss Fraction of Valid</div>
                <div style="font-size: 1.25rem; font-weight: 700; color: #F0F6FC; margin-top: 0.2rem;">{loss_pct:.2f}%</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Render 3-panel figure
    fig_seg = create_segmentation_figure(
        rgb_t2=rgb_t2,
        prob_map=seg_result["probability_map"],
        binary_pred=seg_result["binary_mask_valid"],
        w_mask=w_mask,
        threshold=0.500,
    )
    st.pyplot(fig_seg)
    plt.close(fig_seg)

    # Scientific limitations note
    st.info(
        "**Regulatory Cartography Context:** Predicted areas reflect pixel-level model inferences at 10m spatial resolution "
        "(1 pixel = 0.01 ha). PRODES regulatory masks enforce a 6.25 ha Minimum Mapping Unit (MMU) (~625 contiguous pixels) "
        "and rasterization introduces 1–3 pixel (~10–30m) boundary uncertainties. Disparities on small clearings reflect mapping "
        "granularity rather than definitive classification failure."
    )
