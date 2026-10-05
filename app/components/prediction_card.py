"""Prediction and decision metrics card component for Streamlit application."""

from typing import Any, Dict
import streamlit as st


def render_prediction_card(
    prediction_result: Dict[str, Any],
    ground_truth_label: int,
    reveal_ground_truth: bool = False,
):
    """Render prediction decision banner, probability score, and decision metrics."""
    pred_label = prediction_result["prediction"]
    probability = float(prediction_result["probability"])
    threshold = float(prediction_result["threshold"])
    decision_margin = float(prediction_result.get("decision_margin", 0.0))
    is_deforest = pred_label == "Deforestation"

    # Color scheme
    bg_color = "#fef2f2" if is_deforest else "#f0fdf4"
    border_color = "#f87171" if is_deforest else "#4ade80"
    text_color = "#991b1b" if is_deforest else "#166534"
    badge_bg = "#dc2626" if is_deforest else "#16a34a"

    st.markdown(
        f"""
        <div style="background-color: {bg_color}; border: 1.5px solid {border_color}; border-radius: 8px; padding: 1.2rem; margin-bottom: 1.2rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div>
                    <span style="background-color: {badge_bg}; color: white; padding: 0.3rem 0.8rem; border-radius: 4px; font-size: 0.95rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;">
                        {pred_label}
                    </span>
                    <span style="font-size: 1.25rem; font-weight: 700; color: {text_color}; margin-left: 0.8rem;">
                        Output Probability: {probability * 100:.1f}%
                    </span>
                </div>
                <div style="font-size: 0.9rem; color: #475569; font-weight: 500;">
                    Decision Threshold: <code>τ* = {threshold:.2f}</code> | Decision Margin: <code>{decision_margin * 100:.1f}%</code>
                </div>
            </div>
            <div style="margin-top: 0.8rem;">
                <div style="background-color: #e2e8f0; border-radius: 4px; height: 10px; width: 100%; overflow: hidden; position: relative;">
                    <div style="background-color: {badge_bg}; height: 100%; width: {probability * 100:.1f}%;"></div>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #64748b; margin-top: 0.2rem;">
                    <span>0.0 (Intact Forest)</span>
                    <span style="font-weight: 700; color: #334155;">▲ Decision Threshold (0.27)</span>
                    <span>1.0 (Deforestation)</span>
                </div>
            </div>
            <div style="margin-top: 0.5rem; font-size: 0.76rem; color: #64748b;">
                <em>Note: Decision margin is a derived distance from the classification threshold and is not a calibrated probability of prediction correctness. The output probability is a neural network decision score, not a frequentist certainty metric.</em>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if reveal_ground_truth:
        gt_str = "Deforestation (1)" if ground_truth_label == 1 else "Intact Forest (0)"
        correct = (1 if is_deforest else 0) == ground_truth_label
        gt_bg = "#ecfdf5" if correct else "#fef2f2"
        gt_border = "#6ee7b7" if correct else "#fca5a5"
        gt_status = "✅ PREDICTION MATCHES REFERENCE GROUND TRUTH" if correct else "⚠️ MISMATCH / FALSE ALARM"

        st.markdown(
            f"""
            <div style="background-color: {gt_bg}; border: 1px solid {gt_border}; border-radius: 6px; padding: 0.7rem 1rem; margin-bottom: 1.2rem; font-size: 0.9rem;">
                <strong>PRODES Reference Ground Truth:</strong> <code>{gt_str}</code> — <strong>{gt_status}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
