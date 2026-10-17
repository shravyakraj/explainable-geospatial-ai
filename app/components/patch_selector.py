"""Patch selector sidebar component for Streamlit application."""

from typing import Any, Dict, List, Optional, Tuple
import streamlit as st

from app.utils.data_loader import (
    compute_utm_center_coordinates,
    get_curated_demonstration_patches,
    get_usable_patches,
)


def render_patch_selector() -> Tuple[Dict[str, Any], bool]:
    """Render welcoming patch selection controls and return selected record and reveal_ground_truth flag."""
    st.sidebar.markdown(
        """
        <div style="padding-bottom: 0.4rem; margin-bottom: 0.6rem; border-bottom: 1px solid #1F2937;">
            <div style="font-size: 0.72rem; font-weight: 700; color: #10B981; text-transform: uppercase; letter-spacing: 0.08em;">
                Target Area Selection
            </div>
            <div style="font-size: 0.80rem; color: #94A3B8; margin-top: 0.15rem;">
                Select observation patch for analysis
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selection_mode = st.sidebar.radio(
        "Mode:",
        ["Curated Examples", "Browse Dataset Splits"],
        index=0,
        label_visibility="collapsed",
    )

    usable_patches = get_usable_patches()
    selected_record = None

    if selection_mode == "Curated Examples":
        curated = get_curated_demonstration_patches()
        options = {}
        for p in curated:
            lbl_type = "Deforestation" if int(p["label"]) == 1 else "Intact Forest"
            loss_ha = p.get("prodes_positive_ha", 0.0)
            split_name = p.get("split", "").capitalize()
            label_text = f"[{split_name}] {p['patch_id']} — {lbl_type} ({loss_ha:.1f} ha)" if loss_ha > 0 else f"[{split_name}] {p['patch_id']} — {lbl_type}"
            options[label_text] = p

        choice = st.sidebar.selectbox("Select Area:", list(options.keys()))
        selected_record = options[choice]

    else:
        split = st.sidebar.selectbox("Dataset Split:", ["test", "val", "train"], index=0)
        split_patches = [p for p in usable_patches if p.get("split") == split]

        patch_ids = [p["patch_id"] for p in split_patches]
        selected_id = st.sidebar.selectbox("Patch ID:", patch_ids)
        selected_record = next(p for p in split_patches if p["patch_id"] == selected_id)

    st.sidebar.markdown("<hr style='border: none; border-top: 1px solid #1F2937; margin: 1.0rem 0 0.8rem 0;'>", unsafe_allow_html=True)
    st.sidebar.markdown(
        """
        <div style="font-size: 0.72rem; font-weight: 700; color: #10B981; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.35rem;">
            Cartographic Verification
        </div>
        """,
        unsafe_allow_html=True,
    )
    reveal_gt = st.sidebar.checkbox("Show PRODES Official Record", value=False)

    # Geospatial Metadata Telemetry Card
    st.sidebar.markdown("<hr style='border: none; border-top: 1px solid #1F2937; margin: 1.0rem 0 0.8rem 0;'>", unsafe_allow_html=True)
    st.sidebar.markdown(
        """
        <div style="font-size: 0.72rem; font-weight: 700; color: #10B981; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.45rem;">
            Scene Telemetry & Coordinates
        </div>
        """,
        unsafe_allow_html=True,
    )
    bounds = selected_record.get("patch_bounds_epsg32720", [0, 0, 0, 0])
    geo_info = compute_utm_center_coordinates(bounds)

    st.sidebar.markdown(
        f"""
        <div style="background-color: #111720; border: 1px solid #1F2937; border-radius: 6px; padding: 0.85rem 1.0rem; font-size: 0.80rem; line-height: 1.7; box-shadow: 0 2px 6px rgba(0,0,0,0.3);">
            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #1A2433; padding-bottom: 0.35rem; margin-bottom: 0.4rem;">
                <span style="color: #94A3B8; font-weight: 600;">Patch ID</span>
                <code style="color: #38BDF8; background: #16202A; border: 1px solid #1F2E3D; padding: 0.1rem 0.4rem; border-radius: 4px; font-size: 0.76rem;">{selected_record['patch_id']}</code>
            </div>
            <div style="display: flex; justify-content: space-between;"><span style="color: #94A3B8;">Region</span><span style="color: #F0F6FC; font-weight: 500;">Amazon (20LMR)</span></div>
            <div style="display: flex; justify-content: space-between;"><span style="color: #94A3B8;">Grid Position</span><span style="color: #F0F6FC; font-variant-numeric: tabular-nums;">R{selected_record.get('row')}, C{selected_record.get('col')}</span></div>
            <div style="display: flex; justify-content: space-between;"><span style="color: #94A3B8;">Coverage</span><span style="color: #34D399; font-weight: 600;">{geo_info['area_ha']} ha</span></div>
            <div style="display: flex; justify-content: space-between;"><span style="color: #94A3B8;">Spatial Res</span><span style="color: #F0F6FC;">10m / pixel</span></div>
            <div style="display: flex; justify-content: space-between;"><span style="color: #94A3B8;">T1 Baseline</span><span style="color: #94A3B8; font-variant-numeric: tabular-nums;">08 Aug 2020</span></div>
            <div style="display: flex; justify-content: space-between;"><span style="color: #94A3B8;">T2 Monitor</span><span style="color: #94A3B8; font-variant-numeric: tabular-nums;">08 Aug 2021</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    return selected_record, reveal_gt
