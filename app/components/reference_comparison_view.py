"""Streamlit view component for reference comparison and official benchmark evaluation."""

from typing import Any, Dict, Optional
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from app.utils.data_loader import load_official_test_metrics
from app.visualization.image_renderers import create_confusion_map_figure


def render_reference_comparison_view(
    rgb_t2: np.ndarray,
    y_pred: np.ndarray,
    w_mask: np.ndarray,
    y_ref: Optional[np.ndarray] = None,
    patch_id: Optional[str] = None,
):
    """Render pixel confusion comparison against PRODES reference labels and benchmark report card."""
    st.markdown(
        """
        <div style="margin-top: 1.2rem; margin-bottom: 1.0rem;">
            <h2 style="font-size: 1.35rem; font-weight: 700; color: #F0F6FC; margin-top: 0; margin-bottom: 0.35rem; display: flex; align-items: center; gap: 0.5rem;">
                <span style="color: #10B981;">•</span> Reference Cartography Comparison & Scientific Evaluation
            </h2>
            <p style="font-size: 0.90rem; color: #94A3B8; line-height: 1.6; margin: 0; max-width: 920px;">
                Compare pixel predictions directly against official PRODES 2021 regulatory cartography and review the
                reproducible benchmark metrics established on the geographically held-out test partitions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Patch-level comparison if reference mask is available
    if y_ref is not None and np.sum(w_mask) > 0:
        st.markdown("### A. Pixel Confusion vs. PRODES Reference (Selected Patch)")

        fig_conf, stats = create_confusion_map_figure(
            rgb_t2=rgb_t2,
            y_ref=y_ref,
            y_pred=y_pred,
            w_mask=w_mask,
        )
        st.pyplot(fig_conf)
        plt.close(fig_conf)

        # Pixel & Area breakdown cards
        st.markdown(
            f"""
            <div style="display: flex; gap: 1.0rem; flex-wrap: wrap; margin-top: 0.6rem; margin-bottom: 1.2rem;">
                <div style="background-color: #111720; border: 1px solid #1F2937; border-radius: 6px; padding: 0.7rem 0.9rem; flex: 1; min-width: 140px;">
                    <div style="font-size: 0.72rem; text-transform: uppercase; color: #34D399; font-weight: 700; letter-spacing: 0.05em;">True Positives (TP)</div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #6EE7B7; margin-top: 0.2rem;">{stats['tp_ha']:.2f} ha <span style="font-size: 0.75rem; color: #64748B;">({stats['tp']:,} px)</span></div>
                </div>
                <div style="background-color: #111720; border: 1px solid #1F2937; border-radius: 6px; padding: 0.7rem 0.9rem; flex: 1; min-width: 140px;">
                    <div style="font-size: 0.72rem; text-transform: uppercase; color: #F87171; font-weight: 700; letter-spacing: 0.05em;">False Positives (FP)</div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #FCA5A5; margin-top: 0.2rem;">{stats['fp_ha']:.2f} ha <span style="font-size: 0.75rem; color: #64748B;">({stats['fp']:,} px)</span></div>
                </div>
                <div style="background-color: #111720; border: 1px solid #1F2937; border-radius: 6px; padding: 0.7rem 0.9rem; flex: 1; min-width: 140px;">
                    <div style="font-size: 0.72rem; text-transform: uppercase; color: #FBBF24; font-weight: 700; letter-spacing: 0.05em;">False Negatives (FN)</div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #FDE68A; margin-top: 0.2rem;">{stats['fn_ha']:.2f} ha <span style="font-size: 0.75rem; color: #64748B;">({stats['fn']:,} px)</span></div>
                </div>
                <div style="background-color: #111720; border: 1px solid #1F2937; border-radius: 6px; padding: 0.7rem 0.9rem; flex: 1; min-width: 140px;">
                    <div style="font-size: 0.72rem; text-transform: uppercase; color: #10B981; font-weight: 700; letter-spacing: 0.05em;">Patch Exact IoU</div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #F0F6FC; margin-top: 0.2rem;">{stats['iou']:.4f}</div>
                </div>
                <div style="background-color: #111720; border: 1px solid #1F2937; border-radius: 6px; padding: 0.7rem 0.9rem; flex: 1; min-width: 140px;">
                    <div style="font-size: 0.72rem; text-transform: uppercase; color: #10B981; font-weight: 700; letter-spacing: 0.05em;">Patch Dice Score</div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #F0F6FC; margin-top: 0.2rem;">{stats['dice']:.4f}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("PRODES reference mask is not available for this patch (e.g. custom user upload). Displaying benchmark metrics below.")

    st.markdown("<hr style='border: none; border-top: 1px solid #1F2937; margin: 1.8rem 0 1.2rem 0;'>", unsafe_allow_html=True)

    # 2. Official Evaluation Benchmark Metrics Card
    st.markdown("### B. Verified Benchmark Results (Held-Out Test Partitions)")

    metrics_data = load_official_test_metrics()
    if metrics_data is not None:
        tab_prim, tab_sec = st.tabs([
            "Primary Test Partition (74 Eligible Patches, Columns 23–27)",
            "Secondary Clean Test Subset (60 Classification Clean Patches)",
        ])

        with tab_prim:
            st.markdown(
                """
                *Evaluated on **3,787,261 valid pixels** across all 74 eligible test patches (valid_fraction ≥ 0.50).*
                """
            )
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Exact IoU (Primary)", "0.6451", "95% CI: [0.5173, 0.7283]")
            col2.metric("Global F1 / Dice", "0.7843", "Prec: 0.7320 | Rec: 0.8445")
            col3.metric("Pos-Patch Macro-Dice", "0.7962", "Baseline ΔNDVI: 0.5724")
            col4.metric("Boundary-Tolerant Dice", "0.8072", "1-pixel (10m) dilation")

            st.markdown(
                """
                | Method | Exact IoU | Global Dice | Precision | Recall | False-Positive Area | Detected Ref Area | PRODES Ref Area |
                | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
                | **Champion U-Net (Masked BCE, τ*=0.500)** | **0.6451** | **0.7843** | **0.7320** | **0.8445** | **663.15 ha** | **1,811.64 ha** | **2,145.21 ha** |
                | Baseline $\\Delta\\text{NDVI}$ (τ*=0.175) | 0.4057 | 0.5773 | 0.6040 | 0.5528 | 777.46 ha | 1,185.82 ha | 2,145.21 ha |
                | Baseline Spectral CVA (τ*=0.075) | 0.2193 | 0.3597 | 0.3027 | 0.4430 | 2,188.92 ha | 950.31 ha | 2,145.21 ha |
                """
            )

        with tab_sec:
            st.markdown(
                """
                *Evaluated on **3,225,081 valid pixels** across the 60 clean patches evaluated by MobileNetV3 in Phase 2D.*
                """
            )
            scol1, scol2, scol3, scol4 = st.columns(4)
            scol1.metric("Exact IoU", "0.6725", "95% CI: [0.5587, 0.7542]")
            scol2.metric("Global F1 / Dice", "0.8042", "Prec: 0.7677 | Rec: 0.8444")
            scol3.metric("Pos-Patch Macro-Dice", "0.7908", "Baseline ΔNDVI: 0.5812")
            scol4.metric("Clean Negative Acc", "58.82%", "20 / 34 clean negatives")

            st.markdown(
                """
                | Method | Exact IoU | Global Dice | Precision | Recall | False-Positive Area | Detected Ref Area | PRODES Ref Area |
                | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
                | **Champion U-Net (Masked BCE, τ*=0.500)** | **0.6725** | **0.8042** | **0.7677** | **0.8444** | **547.35 ha** | **1,808.78 ha** | **2,142.21 ha** |
                | Baseline $\\Delta\\text{NDVI}$ (τ*=0.175) | 0.4489 | 0.6196 | 0.7059 | 0.5522 | 492.81 ha | 1,182.86 ha | 2,142.21 ha |
                | Baseline Spectral CVA (τ*=0.075) | 0.2532 | 0.4041 | 0.3720 | 0.4422 | 1,599.38 ha | 947.38 ha | 2,142.21 ha |
                """
            )

    # Scientific generalization warning
    st.warning(
        "**Geographic Scope & Verification Note:** These benchmark metrics reflect evaluation strictly within the geographically "
        "held-out test partition in Rondônia, Brazil (Tile 20LMR, Orbit 53). They do NOT represent universal performance guarantees "
        "for arbitrary user uploads or external satellite scenes across other biomes without localized calibration."
    )
