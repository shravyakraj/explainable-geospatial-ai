"""Patch selector sidebar component for Streamlit application."""

from typing import Any, Dict, List, Optional, Tuple
import streamlit as st

from app.utils.data_loader import (
    compute_utm_center_coordinates,
    get_curated_demonstration_patches,
    get_usable_patches,
)


def render_patch_selector() -> Tuple[Dict[str, Any], bool]:
    """Render patch selection controls and return selected patch record and reveal_ground_truth flag."""
    st.sidebar.markdown("### 🛰️ Sentinel-2 Patch Selection")

    selection_mode = st.sidebar.radio(
        "Selection Mode:",
        ["Curated Demonstrations", "Browse by Dataset Split"],
        index=0,
    )

    usable_patches = get_usable_patches()
    selected_record = None

    if selection_mode == "Curated Demonstrations":
        curated = get_curated_demonstration_patches()
        options = {}
        for p in curated:
            lbl_type = "Deforestation" if int(p["label"]) == 1 else "Intact Forest"
            loss_ha = p.get("prodes_positive_ha", 0.0)
            split_name = p.get("split", "").capitalize()
            label_text = f"[{split_name}] {p['patch_id']} — {lbl_type} ({loss_ha:.1f} ha clearing)" if loss_ha > 0 else f"[{split_name}] {p['patch_id']} — {lbl_type}"
            options[label_text] = p

        choice = st.sidebar.selectbox("Select Demonstration Scenario:", list(options.keys()))
        selected_record = options[choice]

    else:
        split = st.sidebar.selectbox("Filter by Split:", ["test", "val", "train"], index=0)
        split_patches = [p for p in usable_patches if p.get("split") == split]

        patch_ids = [p["patch_id"] for p in split_patches]
        selected_id = st.sidebar.selectbox("Select Patch ID:", patch_ids)
        selected_record = next(p for p in split_patches if p["patch_id"] == selected_id)

    # Ground truth inspection toggle (blind by default for realistic testing)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔬 Evaluation Mode")
    reveal_gt = st.sidebar.checkbox("Reveal PRODES Reference Ground Truth", value=False)

    # Display Geographic Metadata Card
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📍 Patch Geospatial Metadata")
    bounds = selected_record.get("patch_bounds_epsg32720", [0, 0, 0, 0])
    geo_info = compute_utm_center_coordinates(bounds)

    st.sidebar.markdown(
        f"""
        - **Patch ID**: `{selected_record['patch_id']}`
        - **Split**: `{selected_record.get('split', 'N/A').upper()}`
        - **Grid Location**: Row {selected_record.get('row')}, Col {selected_record.get('col')}
        - **Spatial Extent**: 256 × 256 pixels (10m / px)
        - **Patch Surface Area**: {geo_info['area_ha']} hectares (2.56 km × 2.56 km)
        - **PRODES MMU**: 6.25 hectares (reference mapping threshold)
        - **UTM Center**: {geo_info['utm_easting']} E, {geo_info['utm_northing']} N
        - **CRS**: `{geo_info['crs']}`
        - **T1 Acquisition**: `2020-08-08`
        - **T2 Acquisition**: `2021-08-08`
        """
    )

    return selected_record, reveal_gt
