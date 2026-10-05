"""Header component for Streamlit application."""

import streamlit as st


def render_header():
    """Render application header, titles, and model provenance badges."""
    st.markdown(
        """
        <div style="padding-bottom: 1.2rem; border-bottom: 1px solid #e0e0e0; margin-bottom: 1.5rem;">
            <h1 style="font-size: 2.1rem; font-weight: 700; color: #1e293b; margin-bottom: 0.3rem;">
                Explainable AI Based Deforestation Anomaly Detection
            </h1>
            <p style="font-size: 1.05rem; color: #475569; margin-bottom: 0.8rem;">
                Bi-Temporal Sentinel-2 MSI Multi-Spectral Patch Classification & Spatial Grad-CAM Attribution
            </p>
            <div style="display: flex; gap: 0.6rem; flex-wrap: wrap;">
                <span style="background-color: #f1f5f9; color: #334155; padding: 0.25rem 0.65rem; border-radius: 4px; font-size: 0.82rem; font-weight: 600; border: 1px solid #cbd5e1;">
                    Model: MobileNetV3-Small Siamese (Phase 2D Champion)
                </span>
                <span style="background-color: #ecfdf5; color: #065f46; padding: 0.25rem 0.65rem; border-radius: 4px; font-size: 0.82rem; font-weight: 600; border: 1px solid #a7f3d0;">
                    Held-Out Test Accuracy: 93.33% (56/60)
                </span>
                <span style="background-color: #eff6ff; color: #1e40af; padding: 0.25rem 0.65rem; border-radius: 4px; font-size: 0.82rem; font-weight: 600; border: 1px solid #bfdbfe;">
                    Deforestation Recall: 100.0% (Zero Missed Events)
                </span>
                <span style="background-color: #f8fafc; color: #475569; padding: 0.25rem 0.65rem; border-radius: 4px; font-size: 0.82rem; font-weight: 600; border: 1px solid #e2e8f0;">
                    Study Area: Amazon MGRS 20LMR
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
