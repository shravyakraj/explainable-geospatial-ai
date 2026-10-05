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
        <div style="padding-bottom: 0.4rem; margin-bottom: 0.6rem;">
            <div style="font-size: 0.82rem; font-weight: 800; color: #2B2D31; text-transform: uppercase; letter-spacing: 0.05em;">
                Choose an area
            </div>
            <div style="font-size: 0.82rem; color: #555960; margin-top: 0.15rem;">
                Pick a satellite patch to analyze
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

    st.sidebar.markdown("<hr style='border: none; border-top: 1px solid #EBE3F9; margin: 1.1rem 0 0.8rem 0;'>", unsafe_allow_html=True)
    st.sidebar.markdown(
        """
        <div style="font-size: 0.78rem; font-weight: 800; color: #2B2D31; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.3rem;">
            Ground Truth Check
        </div>
        """,
        unsafe_allow_html=True,
    )
    reveal_gt = st.sidebar.checkbox("Show PRODES Official Record", value=False)

    # Friendly Geospatial Metadata
    st.sidebar.markdown("<hr style='border: none; border-top: 1px solid #EBE3F9; margin: 1.1rem 0 0.8rem 0;'>", unsafe_allow_html=True)
    st.sidebar.markdown(
        """
        <div style="font-size: 0.78rem; font-weight: 800; color: #2B2D31; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.4rem;">
            Area Details
        </div>
        """,
        unsafe_allow_html=True,
    )
    bounds = selected_record.get("patch_bounds_epsg32720", [0, 0, 0, 0])
    geo_info = compute_utm_center_coordinates(bounds)

    st.sidebar.markdown(
        f"""
        <div style="background-color: #FAF2F7; border: 1.5px solid #E4D9EB; border-radius: 4px; padding: 0.8rem 0.95rem; font-size: 0.82rem; color: #2B2D31; line-height: 1.65;">
            <div><strong>Patch ID:</strong> <code>{selected_record['patch_id']}</code></div>
            <div><strong>Region:</strong> Amazon (MGRS 20LMR)</div>
            <div><strong>Grid:</strong> Row {selected_record.get('row')}, Col {selected_record.get('col')}</div>
            <div><strong>Area Size:</strong> {geo_info['area_ha']} hectares (2.56 × 2.56 km)</div>
            <div><strong>Resolution:</strong> 10 meters per pixel</div>
            <div><strong>Acquisition 1:</strong> 08 Aug 2020</div>
            <div><strong>Acquisition 2:</strong> 08 Aug 2021</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    return selected_record, reveal_gt
