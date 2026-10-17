"""Header component for Streamlit application."""

import streamlit as st


def render_header():
    """Render a restrained, professional header with project identity and analysis context."""
    st.markdown(
        """
        <div style="padding-bottom: 1.0rem; margin-bottom: 1.5rem; border-bottom: 1px solid #1F2937;">
            <div style="display: flex; align-items: flex-start; justify-content: space-between; flex-wrap: wrap; gap: 1.0rem;">
                <div>
                    <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.35rem;">
                        <span style="font-size: 0.68rem; font-weight: 700; letter-spacing: 0.08em; color: #10B981; text-transform: uppercase; background-color: rgba(16, 185, 129, 0.12); padding: 0.15rem 0.5rem; border-radius: 4px; border: 1px solid rgba(16, 185, 129, 0.25);">
                            GEOSPATIAL AI · SATELLITE RADAR & OPTICAL
                        </span>
                        <span style="font-size: 0.72rem; color: #64748B;">•</span>
                        <span style="font-size: 0.72rem; color: #94A3B8; font-weight: 500;">SIAMESE CLASSIFIER + EARLY-FUSION U-NET</span>
                    </div>
                    <h1 style="font-size: 1.75rem; font-weight: 700; color: #F0F6FC; line-height: 1.2; margin: 0; letter-spacing: -0.02em;">
                        Explainable AI Based Deforestation Anomaly Detection
                    </h1>
                    <div style="font-size: 0.85rem; color: #94A3B8; margin-top: 0.35rem; line-height: 1.5;">
                        Bi-Temporal Multispectral Anomaly Detection & Biophysical Saliency Attribution (Sentinel-2 MSI Level-2A)
                    </div>
                </div>
                <div style="display: flex; align-items: center; gap: 0.6rem; flex-wrap: wrap; margin-top: 0.4rem;">
                    <span style="font-size: 0.76rem; font-weight: 600; color: #34D399; background-color: rgba(16, 185, 129, 0.10); padding: 0.3rem 0.7rem; border-radius: 5px; border: 1px solid rgba(16, 185, 129, 0.25); display: inline-flex; align-items: center; gap: 0.45rem;">
                        <span style="display: inline-block; width: 6px; height: 6px; border-radius: 50%; background-color: #10B981;"></span>
                        Amazon Biome · MGRS 20LMR
                    </span>
                    <span style="font-size: 0.76rem; font-weight: 600; color: #38BDF8; background-color: rgba(56, 189, 248, 0.10); padding: 0.3rem 0.7rem; border-radius: 5px; border: 1px solid rgba(56, 189, 248, 0.25); display: inline-flex; align-items: center; gap: 0.45rem;">
                        <span style="display: inline-block; width: 6px; height: 6px; border-radius: 50%; background-color: #38BDF8;"></span>
                        EPSG:32720 · 10m Ground Sample
                    </span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
