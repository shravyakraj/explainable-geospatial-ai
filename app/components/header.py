"""Header component for Streamlit application."""

import streamlit as st


def render_header():
    """Render the project title with sensible line wrapping and absolutely nothing underneath."""
    st.markdown(
        """
        <div style="padding-bottom: 0.8rem; margin-bottom: 1.1rem; border-bottom: 2px solid #DCE4FF;">
            <h1 style="font-size: 1.95rem; font-weight: 800; color: #2B2D31; line-height: 1.25; margin: 0; letter-spacing: -0.02em;">
                Explainable AI Based Deforestation Anomaly Detection Using Satellite Images
            </h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
