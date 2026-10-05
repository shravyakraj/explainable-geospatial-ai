"""Explainability, spatial change detection, and evidence narrative components for Streamlit application."""

from typing import Any, Dict
import matplotlib.pyplot as plt
import streamlit as st

from app.visualization.image_renderers import (
    array_to_pil_image,
    create_comparison_grid_figure,
    create_highlighted_change_image,
    create_spectral_profile_figure,
)


def render_explainability_view(
    explanation_data: Dict[str, Any],
    t1_patch: Any,
    t2_patch: Any,
):
    """Render Where the change stands out, Change analysis, Why prediction was made, and Automated evidence narrative."""
    is_deforest = explanation_data["prediction"]["prediction"] == "Deforestation"

    # =========================================================================
    # Section: Where the change stands out
    # =========================================================================
    if is_deforest:
        st.markdown(
            """
            <div style="margin-top: 2.0rem; margin-bottom: 1.0rem; border-top: 1.5px solid #EBE3F9; padding-top: 1.6rem;">
                <h2 style="font-size: 1.45rem; font-weight: 800; color: #2B2D31; margin-top: 0; margin-bottom: 0.35rem; letter-spacing: -0.01em;">
                    Where the change stands out
                </h2>
                <p style="font-size: 0.96rem; color: #4A4D53; line-height: 1.6; margin: 0; max-width: 900px;">
                    The highlighted area shows where the strongest change-related visual evidence appears within this satellite patch.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Translucent coral/blush heat emphasis overlay with soft contour around strongest evidence
        highlighted_img = create_highlighted_change_image(
            rgb_image=explanation_data["rgb_t2"],
            change_salience=explanation_data["change_explanation"],
            threshold=0.35,
        )

        col_h1, col_h2, col_h3 = st.columns([1, 8, 1])
        with col_h2:
            st.image(
                highlighted_img,
                caption="After (08 Aug 2021) · Satellite view with strongest change-related evidence highlighted",
                use_column_width=True,
            )
            st.markdown(
                """
                <div style="background-color: #FAF4F8; border: 1px solid #EBE3F9; border-radius: 4px; padding: 0.65rem 1.2rem; margin-top: 0.5rem; margin-bottom: 1.4rem; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 0.6rem; font-size: 0.86rem; color: #2B2D31;">
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <span style="display: inline-block; width: 14px; height: 14px; border-radius: 3px; background-color: #E85C62;"></span>
                        <span><strong>Stronger change-related evidence</strong> → deeper highlight</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <span style="display: inline-block; width: 14px; height: 14px; border-radius: 3px; background-color: #4A6741;"></span>
                        <span><strong>Less change-related evidence</strong> → original imagery</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            """
            <div style="background-color: #F4EEF9; border-radius: 4px; padding: 1.4rem 1.8rem; margin-top: 1.8rem; margin-bottom: 1.2rem;">
                <h2 style="font-size: 1.4rem; font-weight: 800; color: #2B2D31; margin-top: 0; margin-bottom: 0.35rem; letter-spacing: -0.01em;">
                    Where the change stands out
                </h2>
                <div style="font-size: 1.05rem; font-weight: 700; color: #2B2D31; margin-top: 0.2rem; margin-bottom: 0.25rem;">
                    No deforestation anomaly was detected in this selected area.
                </div>
                <p style="font-size: 0.94rem; color: #52555C; line-height: 1.55; margin: 0; max-width: 900px;">
                    The comparison did not identify evidence strong enough to classify this area as a deforestation anomaly.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # =========================================================================
    # Section: Change analysis
    # =========================================================================
    st.markdown(
        """
        <div style="margin-top: 2.0rem; margin-bottom: 1.0rem; border-top: 1.5px solid #EBE3F9; padding-top: 1.6rem;">
            <h2 style="font-size: 1.45rem; font-weight: 800; color: #2B2D31; margin-top: 0; margin-bottom: 0.35rem; letter-spacing: -0.01em;">
                Change analysis
            </h2>
            <p style="font-size: 0.96rem; color: #4A4D53; line-height: 1.6; margin: 0; max-width: 900px;">
                Before and after comparison of vegetation vigor using color-infrared satellite imagery (Near-Infrared, Red, Green). Dense forest canopy reflects strongly in near-infrared.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_ch1, col_ch2 = st.columns(2)
    with col_ch1:
        st.markdown(
            "<div style='font-size: 0.92rem; font-weight: 800; color: #2B2D31; margin-bottom: 0.45rem; letter-spacing: 0.03em;'>BEFORE · 08 AUG 2020</div>",
            unsafe_allow_html=True,
        )
        st.image(
            array_to_pil_image(explanation_data["cir_t1"]),
            caption="Vegetation view (08 Aug 2020) · Color-infrared false color",
            use_column_width=True,
        )

    with col_ch2:
        st.markdown(
            "<div style='font-size: 0.92rem; font-weight: 800; color: #2B2D31; margin-bottom: 0.45rem; letter-spacing: 0.03em;'>AFTER · 08 AUG 2021</div>",
            unsafe_allow_html=True,
        )
        st.image(
            array_to_pil_image(explanation_data["cir_t2"]),
            caption="Vegetation view (08 Aug 2021) · Color-infrared false color",
            use_column_width=True,
        )

    # =========================================================================
    # Section: Why this prediction was made
    # =========================================================================
    st.markdown(
        """
        <div style="margin-top: 2.0rem; margin-bottom: 1.0rem; border-top: 1.5px solid #EBE3F9; padding-top: 1.6rem;">
            <h2 style="font-size: 1.45rem; font-weight: 800; color: #2B2D31; margin-top: 0; margin-bottom: 0.35rem; letter-spacing: -0.01em;">
                Why this prediction was made
            </h2>
            <p style="font-size: 0.96rem; color: #4A4D53; line-height: 1.6; margin: 0; max-width: 900px;">
                The highlighted areas show which parts of the images contributed most strongly to the prediction.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_att1, col_att2 = st.columns(2)
    with col_att1:
        st.markdown(
            "<div style='font-size: 0.92rem; font-weight: 800; color: #2B2D31; margin-bottom: 0.45rem;'>What stands out in the 2020 image</div>",
            unsafe_allow_html=True,
        )
        st.image(
            array_to_pil_image(explanation_data["overlay_t1"]),
            caption="Visual highlights for the earlier image (08 Aug 2020)",
            use_column_width=True,
        )

    with col_att2:
        st.markdown(
            "<div style='font-size: 0.92rem; font-weight: 800; color: #2B2D31; margin-bottom: 0.45rem;'>What stands out in the 2021 image</div>",
            unsafe_allow_html=True,
        )
        st.image(
            array_to_pil_image(explanation_data["overlay_t2"]),
            caption="Visual highlights for the later image (08 Aug 2021)",
            use_column_width=True,
        )

    # =========================================================================
    # Section: Automated evidence narrative
    # =========================================================================
    st.markdown(
        f"""
        <div style="background-color: #FFF5CB; border-radius: 4px; padding: 1.5rem 1.8rem; margin-top: 2.2rem; margin-bottom: 1.6rem;">
            <h2 style="font-size: 1.45rem; font-weight: 800; color: #2B2D31; margin-top: 0; margin-bottom: 0.25rem; letter-spacing: -0.01em;">
                Automated evidence narrative
            </h2>
            <div style="font-size: 0.88rem; color: #625732; margin-bottom: 0.85rem;">
                Here is what the satellite comparison found, in plain language.
            </div>
            <div style="background-color: #FAF8F5; border-radius: 4px; padding: 1.2rem 1.4rem; font-size: 0.96rem; color: #2B2D31; line-height: 1.65;">
                {explanation_data['explanation_text']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # =========================================================================
    # Technical Details & Spectral Charts (Collapsible)
    # =========================================================================
    with st.expander("Technical details & spectral charts", expanded=False):
        st.markdown(
            "<div style='font-size: 0.88rem; font-weight: 700; color: #2B2D31; margin-bottom: 0.3rem;'>Quantitative Spectral Reflectance & NDVI Profiles</div>",
            unsafe_allow_html=True,
        )
        spec_fig = create_spectral_profile_figure(
            t1_patch=t1_patch,
            t2_patch=t2_patch,
            ndvi_t1=explanation_data["ndvi_t1"],
            ndvi_t2=explanation_data["ndvi_t2"],
        )
        st.pyplot(spec_fig)
        plt.close(spec_fig)

        st.markdown(
            "<div style='font-size: 0.88rem; font-weight: 700; color: #2B2D31; margin-top: 0.8rem; margin-bottom: 0.3rem;'>Complete 3×3 Multi-Spectral & Attribution Matrix</div>",
            unsafe_allow_html=True,
        )
        grid_fig = create_comparison_grid_figure(
            rgb_t1=explanation_data["rgb_t1"],
            rgb_t2=explanation_data["rgb_t2"],
            cam_t1=explanation_data["cam_t1"],
            cam_t2=explanation_data["cam_t2"],
            change_exp=explanation_data["change_explanation"],
            delta_ndvi=explanation_data["delta_ndvi"],
            overlay_t1=explanation_data["overlay_t1"],
            overlay_t2=explanation_data["overlay_t2"],
            overlay_change=explanation_data["overlay_change"],
            title="Bi-Temporal Multi-Spectral & Attribution Matrix",
        )
        st.pyplot(grid_fig)
        plt.close(grid_fig)
