"""Prediction and decision metrics component for Streamlit application."""

from typing import Any, Dict
import streamlit as st


def render_prediction_card(
    prediction_result: Dict[str, Any],
    ground_truth_label: int,
    reveal_ground_truth: bool = False,
):
    """Render dramatic, high-impact prediction result banner across a soft salmon/lilac visual field."""
    pred_label = prediction_result["prediction"]
    probability = float(prediction_result["probability"])
    threshold = float(prediction_result["threshold"])
    is_deforest = pred_label == "Deforestation"

    if is_deforest:
        banner_title = "DEFORESTATION DETECTED"
        bg_color = "#FDE8E8"       # Soft blush
        accent_color = "#E85C62"   # Muted salmon/coral
    else:
        banner_title = "NO DEFORESTATION DETECTED"
        bg_color = "#F4EEF9"       # Pale lilac/neutral
        accent_color = "#BCA8E8"   # Muted lilac

    st.markdown(
        f"""
        <div style="background-color: {bg_color}; border-left: 6px solid {accent_color}; border-radius: 4px; padding: 1.4rem 2rem; margin-top: 1.4rem; margin-bottom: 1.6rem;">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 1.2rem;">
                <div>
                    <h2 style="font-size: 1.65rem; font-weight: 800; color: #2B2D31; letter-spacing: -0.01em; margin: 0; line-height: 1.2;">
                        {banner_title}
                    </h2>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 2.8rem; font-weight: 900; line-height: 1; color: #2B2D31; letter-spacing: -0.03em;">
                        {probability * 100:.1f}%
                    </div>
                    <div style="font-size: 0.88rem; font-weight: 700; color: #4A4D53; margin-top: 0.25rem;">
                        Detection probability <span style="font-weight: 400; font-size: 0.8rem; color: #7B818C;">(threshold: {threshold * 100:.0f}%)</span>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if reveal_ground_truth:
        gt_str = "Deforestation (1)" if ground_truth_label == 1 else "Intact Forest (0)"
        correct = (1 if is_deforest else 0) == ground_truth_label
        gt_status = "Matches the PRODES reference" if correct else "Does not match the PRODES reference"

        st.markdown(
            f"""
            <div style="background-color: #FFF5CB; border-radius: 4px; padding: 0.8rem 1.4rem; margin-top: -0.9rem; margin-bottom: 1.6rem; font-size: 0.9rem; color: #2B2D31;">
                <span style="font-weight: 700;">PRODES reference:</span> <code>{gt_str}</code>
                <span style="margin-left: 0.6rem; color: #2B2D31; font-weight: 600;">·&nbsp; {gt_status}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
