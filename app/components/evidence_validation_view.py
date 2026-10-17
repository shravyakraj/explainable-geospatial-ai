"""Evidence validation report component for unsupported or insufficient satellite inputs."""

from typing import Any, Dict
import streamlit as st
from PIL import Image

from app.utils.report_exporter import generate_evidence_validation_html_report
from app.visualization.image_renderers import array_to_pil_image


def render_evidence_validation_view(validation_result: Dict[str, Any]):
    """Render structured evidence validation diagnostic report when input criteria are not satisfied."""
    st.markdown(
        """
        <div style="background-color: #1A1408; border-left: 5px solid #F59E0B; border: 1px solid #38270C; border-left-width: 5px; border-radius: 6px; padding: 1.2rem 1.6rem; margin-top: 1.0rem; margin-bottom: 1.4rem;">
            <h2 style="font-size: 1.4rem; font-weight: 700; color: #FBBF24; margin: 0; line-height: 1.2;">
                Evidence Validation Report: Input Insufficient for Deforestation Analysis
            </h2>
            <p style="font-size: 0.92rem; color: #FDE68A; line-height: 1.6; margin-top: 0.5rem; margin-bottom: 0;">
                The uploaded files were inspected against scientific remote sensing requirements.
                Because essential evidence is missing, the application will not produce a false or ungrounded prediction.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Display preview of submitted imagery
    images = validation_result.get("images", [])
    meta_list = validation_result.get("image_metadata", [])

    if images:
        st.markdown("### Submitted Image Evidence")
        cols = st.columns(len(images))
        for idx, (col, img_arr, m) in enumerate(zip(cols, images, meta_list)):
            with col:
                st.markdown(f"**Image {idx+1}: {m.get('filename')}**")
                st.image(array_to_pil_image(img_arr), use_column_width=True)
                st.caption(f"{m.get('format')} · {m.get('width')} × {m.get('height')} px · 3 channels (RGB)")

    # Section: Identified Deficiencies
    st.markdown("### Scientific Evaluation of Submitted Evidence")
    deficiencies = validation_result.get("deficiencies", [])
    for d in deficiencies:
        st.markdown(
            f"""
            <div style="background-color: #111720; border: 1px solid #1F2937; border-radius: 6px; padding: 0.85rem 1.2rem; margin-bottom: 0.6rem; font-size: 0.90rem; color: #F0F6FC; line-height: 1.55;">
                • {d}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Section: Exact Requirements
    st.markdown("### Input Specification for Verified Analysis")
    st.markdown(
        """
        <div style="background-color: #111720; border: 1px solid #1F2937; border-radius: 6px; padding: 1.1rem 1.4rem; margin-bottom: 1.4rem; font-size: 0.88rem; color: #E2E8F0; line-height: 1.6;">
            <strong style="color: #10B981;">To generate a scientifically valid deforestation analysis report, provide:</strong><br>
            1. <strong>Bi-Temporal Observations:</strong> Registered imagery from two distinct dates across an observation interval (e.g. 1 year).<br>
            2. <strong>Near-Infrared Spectral Band (B08):</strong> Surface reflectance in NIR is required to separate healthy vegetation from clearing.<br>
            3. <strong>Compatible Formats:</strong> Either a <code>.npz</code> research archive containing 4 bands per date, or paired multi-band GeoTIFFs containing Sentinel-2 bands <code>[B02 (Blue), B03 (Green), B04 (Red), B08 (NIR)]</code>.<br>
            4. <strong>Spatial Resolution:</strong> Calibrated 10 meters per pixel for hectare conversion.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Report Download Button
    html_report = generate_evidence_validation_html_report(validation_result)
    st.download_button(
        label="Download Evidence Validation Report (HTML)",
        data=html_report,
        file_name="evidence_validation_report.html",
        mime="text/html",
        help="Download self-contained HTML report documenting the evidence validation findings.",
    )
