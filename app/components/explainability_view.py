"""Explainability and visualization component for Streamlit application."""

from typing import Any, Dict
import matplotlib.pyplot as plt
import streamlit as st

from app.visualization.image_renderers import (
    array_to_pil_image,
    create_comparison_grid_figure,
    create_spectral_profile_figure,
)


def render_explainability_view(
    explanation_data: Dict[str, Any],
    t1_patch: Any,
    t2_patch: Any,
):
    """Render interactive explainability visual panels and automated scientific text."""
    st.markdown("### 🔍 Model Explainability & Spatial Attribution")

    tabs = st.tabs([
        "🔥 Bi-Temporal Grad-CAM",
        "🌿 Multi-Spectral & NDVI Analysis",
        "📊 Complete 3×3 Scientific Grid",
        "📝 Automated Scientific Summary",
    ])

    with tabs[0]:
        st.markdown(
            """
            **Spatial Feature Attribution Across Observation Dates**  
            - **T1 Grad-CAM**: Baseline pre-event model activation (2020-08-08)  
            - **T2 Grad-CAM**: Monitoring post-event model activation (2021-08-08)  
            - **Derived Relative Salience**: Pre-normalization activation difference ($\text{ReLU}(\text{CAM}_{T2}^{\text{raw}} - \text{CAM}_{T1}^{\text{raw}})$), normalized to $[0, 1]$. Highlights spatial regions where positive model activation increased.
            """
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(
                array_to_pil_image(explanation_data["overlay_t1"]),
                caption="T1 Baseline Scene + Grad-CAM (2020-08-08)",
                use_column_width=True,
            )
            st.caption("Relative spatial salience in baseline scene influencing latent representation.")

        with col2:
            st.image(
                array_to_pil_image(explanation_data["overlay_t2"]),
                caption="T2 Monitoring Scene + Grad-CAM (2021-08-08)",
                use_column_width=True,
            )
            st.caption("Relative spatial salience across features in the monitoring scene.")

        with col3:
            st.image(
                array_to_pil_image(explanation_data["overlay_change"]),
                caption="Derived Relative Salience Map",
                use_column_width=True,
            )
            st.caption("Pre-normalization difference highlighting regions of increased positive model activation.")

    with tabs[1]:
        st.markdown(
            """
            **Direct Biophysical & Spectral Comparison**  
            Sentinel-2 multi-spectral bands reveal vegetation vitality:
            - **CIR (Color Infrared)**: $B08 \to \text{Red}, B04 \to \text{Green}, B03 \to \text{Blue}$. Healthy dense canopy appears intense red. Clearings appear cyan/grey.
            - **Spectral ΔNDVI**: Direct physical change in vegetation index ($\text{NDVI}_{T2} - \text{NDVI}_{T1}$).
            """
        )

        col_cir1, col_cir2 = st.columns(2)
        with col_cir1:
            st.image(
                array_to_pil_image(explanation_data["cir_t1"]),
                caption="T1 False Color NIR (CIR: B08, B04, B03, 2020-08-08)",
                use_column_width=True,
            )
        with col_cir2:
            st.image(
                array_to_pil_image(explanation_data["cir_t2"]),
                caption="T2 False Color NIR (CIR: B08, B04, B03, 2021-08-08)",
                use_column_width=True,
            )

        st.markdown("#### Quantitative Spectral Reflectance & NDVI Profiles")
        spec_fig = create_spectral_profile_figure(
            t1_patch=t1_patch,
            t2_patch=t2_patch,
            ndvi_t1=explanation_data["ndvi_t1"],
            ndvi_t2=explanation_data["ndvi_t2"],
        )
        st.pyplot(spec_fig)
        plt.close(spec_fig)

    with tabs[2]:
        st.markdown("#### High-Resolution 3×3 Scientific Comparison Grid")
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
            title=f"Bi-Temporal Explainability Summary",
        )
        st.pyplot(grid_fig)
        plt.close(grid_fig)

    with tabs[3]:
        st.markdown("#### Evidence-Based Scientific Explanation")
        st.info(explanation_data["explanation_text"])

        st.markdown(
            r"""
            > **Scientific Integrity & Limitations Note**:  
            > - The explanation above is strictly synthesized from measured spectral data (Sentinel-2 BOA surface reflectance and calculated $\Delta\text{NDVI}$) and empirical model outputs.  
            > - **No speculative causal mechanisms** (e.g. logging, road construction, agriculture, fire) are claimed.  
            > - **Grad-CAM is a spatial attribution technique**, indicating which feature regions contributed most strongly to the classification decision; it is **not** a pixel-perfect segmentation mask.
            """
        )
