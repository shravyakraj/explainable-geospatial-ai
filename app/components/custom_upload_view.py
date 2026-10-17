"""Streamlit view component for user-uploaded custom .npz satellite patch analysis."""

from typing import Any, Dict
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from app.inference.predictor import (
    run_inference_and_explain,
    run_integrated_gradients_explanation,
    run_segmentation_inference,
)
from app.utils.data_loader import validate_uploaded_npz
from app.visualization.image_renderers import (
    array_to_pil_image,
    create_confusion_map_figure,
    create_integrated_gradients_figure,
    create_segmentation_figure,
)


def render_custom_upload_view():
    """Render custom patch upload, validation, and isolated multi-model inference."""
    st.markdown(
        """
        <div style="margin-top: 1.2rem; margin-bottom: 1.0rem;">
            <h2 style="font-size: 1.35rem; font-weight: 700; color: #F0F6FC; margin-top: 0; margin-bottom: 0.35rem; display: flex; align-items: center; gap: 0.5rem;">
                <span style="color: #10B981;">•</span> Custom Patch Upload & Inference Validation
            </h2>
            <p style="font-size: 0.90rem; color: #94A3B8; line-height: 1.6; margin: 0; max-width: 920px;">
                Upload a custom Sentinel-2 <code>.npz</code> archive containing aligned bi-temporal imagery to evaluate both
                the MobileNetV3 Siamese Classifier and the Early-Fusion U-Net.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="background-color: #111720; border: 1px solid #1F2937; border-radius: 6px; padding: 0.9rem 1.2rem; margin-bottom: 1.2rem; font-size: 0.85rem; color: #94A3B8; line-height: 1.65;">
            <div style="color: #10B981; font-weight: 700; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.4rem;">Expected File Contract:</div>
            • <strong style="color: #F0F6FC;">Archive format:</strong> <code>.npz</code> file.<br>
            • <strong style="color: #F0F6FC;">Required arrays:</strong> Either <code>t1</code> and <code>t2</code> (each shape <code>(256, 256, 4)</code>) or <code>x</code> (shape <code>(256, 256, 8)</code>).<br>
            • <strong style="color: #F0F6FC;">Band ordering:</strong> Sentinel-2 <code>[B02 (Blue), B03 (Green), B04 (Red), B08 (NIR)]</code>.<br>
            • <strong style="color: #F0F6FC;">Optional arrays:</strong> <code>mask</code> (shape <code>(256, 256)</code>) and <code>exclude</code> (shape <code>(256, 256)</code>) for reference comparison.<br>
            • <em>Isolation Guarantee: Uploaded files are processed in-memory and never added to training or benchmark evaluation splits.</em>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Choose a .npz Sentinel-2 patch archive:",
        type=["npz"],
        help="Upload a 256x256 bi-temporal Sentinel-2 patch.",
    )

    if uploaded_file is None:
        st.info("Awaiting file upload. Select a .npz file above to initiate automated validation and analysis.")
        return

    # Validate uploaded file
    with st.spinner("Validating input array dimensions, data types, and temporal channels..."):
        val_result = validate_uploaded_npz(uploaded_file)

    if not val_result["valid"]:
        st.error(f"Validation Failure: {val_result['error']}")
        return

    st.success(f"Validation Passed! Detected arrays: {val_result['keys_found']}. Reference mask present: {val_result['has_reference']}.")

    t1 = val_result["t1"]
    t2 = val_result["t2"]
    x_8ch = val_result["x"]
    y_ref = val_result["y"]
    w_mask = val_result["w"]

    # RGB preview
    rgb_t1 = t1[..., [2, 1, 0]]
    rgb_t2 = t2[..., [2, 1, 0]]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Uploaded T1 (Before) True Color**")
        st.image(array_to_pil_image(rgb_t1), use_column_width=True)
    with col2:
        st.markdown("**Uploaded T2 (After) True Color**")
        st.image(array_to_pil_image(rgb_t2), use_column_width=True)

    # Run inference tabs for uploaded patch
    up_tab_clf, up_tab_seg, up_tab_ig = st.tabs([
        "1. Classification Inference",
        "2. Segmentation Inference",
        "3. Integrated Gradients",
    ])

    with up_tab_clf:
        with st.spinner("Running MobileNetV3 Siamese Classification..."):
            clf_res = run_inference_and_explain(t1, t2)
            pred_meta = clf_res["prediction"]

        is_deforest = pred_meta["prediction"] == "Deforestation"
        banner_bg = "#1C1215" if is_deforest else "#0D1E16"
        banner_border = "#EF4444" if is_deforest else "#10B981"
        banner_text = "#FCA5A5" if is_deforest else "#6EE7B7"
        banner_outline = "#3E1B21" if is_deforest else "#133827"

        st.markdown(
            f"""
            <div style="background-color: {banner_bg}; border: 1px solid {banner_outline}; border-left: 5px solid {banner_border}; border-radius: 6px; padding: 1.0rem 1.4rem; margin-top: 0.8rem; margin-bottom: 1.0rem; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);">
                <h3 style="margin: 0; font-size: 1.25rem; font-weight: 700; color: {banner_text};">
                    Prediction: {pred_meta['prediction']}
                </h3>
                <p style="margin: 0.4rem 0 0 0; font-size: 0.90rem; color: #CBD5E1; line-height: 1.55;">
                    <strong style="color: #F0F6FC;">Detection Probability:</strong> {pred_meta['probability'] * 100:.1f}% (Decision threshold: {pred_meta.get('threshold', 0.27) * 100:.0f}%)<br>
                    <strong style="color: #F0F6FC;">Status:</strong> {'Deforestation anomaly detected across observation interval.' if is_deforest else 'Canopy state classified as intact / undisturbed forest.'}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with up_tab_seg:
        with st.spinner("Running Early-Fusion U-Net Pixel Segmentation..."):
            seg_res = run_segmentation_inference(x_8channel=x_8ch, threshold=0.500, w_mask=w_mask)

        fig_seg = create_segmentation_figure(
            rgb_t2=rgb_t2,
            prob_map=seg_res["probability_map"],
            binary_pred=seg_res["binary_mask_valid"],
            w_mask=w_mask,
            threshold=0.500,
        )
        st.pyplot(fig_seg)
        plt.close(fig_seg)

        st.write(
            f"**Predicted Clearing Area:** {seg_res['predicted_loss_ha']:.2f} ha "
            f"({seg_res['predicted_loss_pixels']:,} pixels out of {seg_res['total_valid_pixels']:,} valid pixels)."
        )

        if val_result["has_reference"] and y_ref is not None:
            st.markdown("#### Reference Comparison (Available in Uploaded File)")
            fig_conf, conf_stats = create_confusion_map_figure(rgb_t2, y_ref, seg_res["binary_mask_valid"], w_mask)
            st.pyplot(fig_conf)
            plt.close(fig_conf)
            st.write(
                f"**Exact IoU:** {conf_stats['iou']:.4f} | **Dice:** {conf_stats['dice']:.4f} | "
                f"**TP:** {conf_stats['tp_ha']:.2f} ha | **FP:** {conf_stats['fp_ha']:.2f} ha | **FN:** {conf_stats['fn_ha']:.2f} ha"
            )
        else:
            st.info("Reference mask not included in uploaded file. Showing raw model predictions without ground-truth comparison.")

    with up_tab_ig:
        if st.button("Generate Integrated Gradients for Uploaded Patch"):
            with st.spinner("Computing Integrated Gradients on uploaded patch..."):
                ig_res = run_integrated_gradients_explanation(
                    x_8channel=x_8ch,
                    w_mask=w_mask,
                    patch_id="uploaded_patch",
                    y_ref=y_ref,
                    num_steps=25,
                )
                fig_ig = create_integrated_gradients_figure(
                    spatial_attr=ig_res.spatial_attribution,
                    t1_attr=ig_res.t1_spatial_attribution,
                    t2_attr=ig_res.t2_spatial_attribution,
                    w_mask=w_mask,
                    per_band_attributions=ig_res.per_band_attributions,
                    completeness_error=ig_res.completeness_relative_error,
                )
                st.pyplot(fig_ig)
                plt.close(fig_ig)
                st.write(f"**Axiomatic Completeness Error:** {ig_res.completeness_relative_error:.3%}")
