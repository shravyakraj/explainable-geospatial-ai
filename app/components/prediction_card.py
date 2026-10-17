"""Prediction and decision metrics component for Streamlit application."""

from typing import Any, Dict
import streamlit as st


def render_prediction_card(
    prediction_result: Dict[str, Any],
    ground_truth_label: int,
    reveal_ground_truth: bool = False,
):
    """Render high-impact prediction result banner in the dark geospatial theme."""
    pred_label = prediction_result["prediction"]
    probability = float(prediction_result["probability"])
    threshold = float(prediction_result["threshold"])
    is_deforest = pred_label == "Deforestation"

    if is_deforest:
        banner_title = "DEFORESTATION DETECTED"
        bg_color = "#1C1215"
        border_outline = "#3E1B21"
        accent_color = "#EF4444"
        title_color = "#FCA5A5"
    else:
        banner_title = "NO DEFORESTATION DETECTED"
        bg_color = "#0D1E16"
        border_outline = "#133827"
        accent_color = "#10B981"
        title_color = "#6EE7B7"

    st.markdown(
        f"""
        <div style="background-color: {bg_color}; border: 1px solid {border_outline}; border-left: 5px solid {accent_color}; border-radius: 6px; padding: 1.4rem 2rem; margin-top: 1.2rem; margin-bottom: 1.4rem; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 1.2rem;">
                <div>
                    <h2 style="font-size: 1.45rem; font-weight: 800; color: {title_color}; letter-spacing: 0.01em; margin: 0; line-height: 1.2;">
                        {banner_title}
                    </h2>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 2.6rem; font-weight: 800; line-height: 1; color: #F0F6FC; font-variant-numeric: tabular-nums;">
                        {probability * 100:.1f}%
                    </div>
                    <div style="font-size: 0.82rem; font-weight: 600; color: #94A3B8; margin-top: 0.25rem;">
                        Detection probability <span style="font-weight: 400; color: #64748B;">(threshold: {threshold * 100:.0f}%)</span>
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
        badge_border = "#10B981" if correct else "#F59E0B"
        status_color = "#34D399" if correct else "#FBBF24"

        st.markdown(
            f"""
            <div style="background-color: #111720; border: 1px solid #1F2937; border-left: 4px solid {badge_border}; border-radius: 6px; padding: 0.8rem 1.4rem; margin-top: -0.7rem; margin-bottom: 1.6rem; font-size: 0.88rem; color: #94A3B8;">
                <span style="font-weight: 700; color: #F0F6FC;">PRODES reference:</span> <code style="background-color: #161F2E; color: #38BDF8; padding: 2px 6px; border-radius: 4px;">{gt_str}</code>
                <span style="margin-left: 0.6rem; color: {status_color}; font-weight: 600;">·&nbsp; {gt_status}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
