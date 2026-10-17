"""Streamlit main entry point for Explainable Deforestation Anomaly Detection & Segmentation Dashboard.

Implements a professional, report-first scientific workflow:
Upload / Select imagery -> Validate evidence -> Run supported analysis -> Generate coherent report.
"""

import logging
from pathlib import Path
import sys

# Resolve project paths and configure import search paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = str(PROJECT_ROOT / "app")
SRC_DIR = str(PROJECT_ROOT / "src")

# 1. Ensure project root and src/ are on sys.path so 'app' and 'geospatial_ai' are importable
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# 2. Purge APP_DIR from sys.path so 'app' resolves to the package directory
while APP_DIR in sys.path:
    sys.path.remove(APP_DIR)

# 3. If 'app' was already cached as a non-package module in sys.modules, evict it
if "app" in sys.modules and not hasattr(sys.modules["app"], "__path__"):
    del sys.modules["app"]

import numpy as np
import streamlit as st

from app.components.evidence_validation_view import render_evidence_validation_view
from app.components.header import render_header
from app.components.patch_selector import render_patch_selector
from app.components.unified_report_view import render_unified_analysis_report
from app.utils.data_loader import (
    compute_utm_center_coordinates,
    load_patch_data,
    load_segmentation_patch_data,
)
from app.utils.input_validator import (
    validate_bitemporal_multispectral_files,
    validate_npz_archive,
    validate_rgb_imagery,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("streamlit_app")


def main():
    st.set_page_config(
        page_title="Explainable AI Based Deforestation Anomaly Detection",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Professional Dark Geospatial Analytics Design System
    st.markdown(
        """
        <style>
        :root {
            --bg-page: #0B0F14;
            --bg-surface: #111720;
            --bg-surface-elevated: #161F2B;
            --bg-surface-active: #1D2837;
            --primary-forest: #10B981;
            --secondary-teal: #059669;
            --text-primary: #F0F6FC;
            --text-secondary: #94A3B8;
            --text-muted: #64748B;
            --border-subtle: #1F2937;
            --border-strong: #2D3A4B;
            --accent-emerald: #10B981;
            --accent-emerald-subtle: rgba(16, 185, 129, 0.12);
            --accent-ruby: #EF4444;
            --accent-ruby-subtle: rgba(239, 68, 68, 0.12);
        }
        .stApp {
            background-color: #0B0F14;
            color: #F0F6FC;
        }
        .main .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2.5rem;
            padding-left: 2rem;
            padding-right: 2rem;
            max-width: 1260px;
        }
        h1, h2, h3, h4, p, span, label, div {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            color: #F0F6FC;
        }
        h1 {
            font-size: 1.85rem;
            font-weight: 700;
            color: #F0F6FC;
            letter-spacing: -0.02em;
            margin-bottom: 0.2rem;
        }
        h2 {
            font-size: 1.35rem;
            font-weight: 700;
            color: #F0F6FC;
            letter-spacing: -0.01em;
            margin-top: 0;
            margin-bottom: 0.35rem;
        }
        h3 {
            font-size: 1.05rem;
            font-weight: 600;
            color: #E2E8F0;
            margin-top: 0.4rem;
            margin-bottom: 0.25rem;
        }
        section[data-testid="stSidebar"] {
            background-color: #0E141C;
            border-right: 1px solid #1F2937;
        }
        section[data-testid="stSidebar"] .block-container {
            padding-top: 1.4rem;
            padding-left: 1.2rem;
            padding-right: 1.2rem;
        }
        code {
            color: #34D399;
            background-color: #16202A;
            border: 1px solid #1F2E3D;
            padding: 0.15rem 0.4rem;
            border-radius: 4px;
            font-size: 0.85em;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        }
        div[data-testid="stImage"] img {
            border: 1px solid #1F2937;
            border-radius: 6px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
            display: block;
            width: 100%;
        }
        div[data-testid="stExpander"] {
            border: 1px solid #1F2937;
            border-radius: 6px;
            background-color: #111720;
        }
        div[data-testid="stMetric"] {
            background-color: #111720;
            border: 1px solid #1F2937;
            border-radius: 6px;
            padding: 0.85rem 1.1rem;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.25);
            transition: border-color 0.15s ease;
        }
        div[data-testid="stMetric"]:hover {
            border-color: #2D3A4B;
        }
        div[data-testid="stMetricValue"] {
            color: #F0F6FC !important;
            font-weight: 700 !important;
            font-size: 1.55rem !important;
            font-variant-numeric: tabular-nums;
        }
        div[data-testid="stMetricLabel"] {
            color: #10B981 !important;
            font-weight: 700 !important;
            font-size: 0.72rem !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
        }
        .stButton>button {
            background-color: #059669;
            color: #FFFFFF;
            border: 1px solid #10B981;
            border-radius: 5px;
            font-weight: 600;
            padding: 0.45rem 1.0rem;
            transition: all 0.15s ease;
        }
        .stButton>button:hover {
            background-color: #10B981;
            border-color: #34D399;
            color: #FFFFFF;
            box-shadow: 0 0 10px rgba(16, 185, 129, 0.25);
        }
        .stDownloadButton>button {
            background-color: #111720;
            color: #F0F6FC;
            border: 1px solid #2D3A4B;
            border-radius: 5px;
            font-weight: 600;
            padding: 0.45rem 1.0rem;
            transition: all 0.15s ease;
        }
        .stDownloadButton>button:hover {
            background-color: #161F2B;
            border-color: #10B981;
            color: #34D399;
        }
        div[data-baseweb="select"] > div {
            background-color: #161F2B;
            border-color: #2D3A4B;
            color: #F0F6FC;
            border-radius: 5px;
        }
        div[data-baseweb="select"] > div:hover {
            border-color: #10B981;
        }
        div[data-testid="stRadio"] label {
            color: #E2E8F0 !important;
            font-size: 0.88rem !important;
        }
        div[data-testid="stCheckbox"] label span {
            color: #CBD5E1 !important;
            font-size: 0.88rem !important;
        }
        div[data-testid="stSlider"] div[role="slider"] {
            background-color: #10B981 !important;
            border: 2px solid #064E3B !important;
        }
        div[data-testid="stSlider"] [data-baseweb="slider"] div {
            color: #94A3B8;
        }
        button[data-baseweb="tab"] {
            font-weight: 600;
            font-size: 0.86rem;
            color: #94A3B8 !important;
            border-bottom: 2px solid transparent !important;
            padding: 0.5rem 1.0rem !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #10B981 !important;
            border-bottom-color: #10B981 !important;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid #1F2937;
            border-radius: 6px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # 1. Project Header
    render_header()

    # 2. Input Source Selection
    st.sidebar.markdown(
        """
        <div style="padding-bottom: 0.4rem; margin-bottom: 0.6rem; border-bottom: 1px solid #1F2937;">
            <div style="font-size: 0.80rem; font-weight: 700; color: #10B981; text-transform: uppercase; letter-spacing: 0.05em;">
                Input Selection
            </div>
            <div style="font-size: 0.80rem; color: #94A3B8; margin-top: 0.15rem;">
                Select observation source for analysis
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    data_source_mode = st.sidebar.radio(
        "Source Mode:",
        ["Curated Benchmark Examples", "Upload Satellite Imagery"],
        index=0,
        label_visibility="collapsed",
    )

    # =========================================================================
    # Branch 1: Curated Benchmark Examples
    # =========================================================================
    if data_source_mode == "Curated Benchmark Examples":
        try:
            selected_record, reveal_gt = render_patch_selector()
        except Exception as e:
            st.error(f"Error loading patch manifest: {e}")
            logger.exception("Error in render_patch_selector")
            return

        patch_id = selected_record.get("patch_id", "patch_sample")
        patch_path = selected_record["absolute_path"]

        try:
            t1, t2, ground_truth_label = load_patch_data(patch_path)
            x_8ch, y_ref, w_mask = load_segmentation_patch_data(patch_path)
        except Exception as e:
            st.error(f"Error loading patch data: {e}")
            logger.exception("Error loading patch data")
            return

        bounds = selected_record.get("patch_bounds_epsg32720", [0, 0, 0, 0])
        geo_info = compute_utm_center_coordinates(bounds)

        metadata = {
            "patch_id": patch_id,
            "sensor": "Sentinel-2 MSI Level-2A",
            "spatial_resolution": "10 meters per pixel (0.01 ha/pixel)",
            "observation_dates": "08 Aug 2020 (T1) to 08 Aug 2021 (T2)",
            "dimensions": "256 × 256 pixels (2.56 × 2.56 km)",
            "area_ha": geo_info["area_ha"],
            "utm_coordinates": f"Easting: {geo_info['utm_easting']} m, Northing: {geo_info['utm_northing']} m",
            "projection": geo_info["crs"],
            "split": selected_record.get("split", "test").capitalize(),
            "patch_bounds_epsg32720": bounds,
        }

        # If user toggled off reference reveal, pass y_ref=None to evaluate autonomously
        effective_y_ref = y_ref if reveal_gt else None

        render_unified_analysis_report(
            patch_id=patch_id,
            t1=t1,
            t2=t2,
            x_8ch=x_8ch,
            metadata=metadata,
            y_ref=effective_y_ref,
            w_mask=w_mask,
            ground_truth_label=ground_truth_label,
            reveal_ground_truth=reveal_gt,
        )

    # =========================================================================
    # Branch 2: Upload Satellite Imagery
    # =========================================================================
    else:
        st.markdown(
            """
            <div style="background-color: #111720; border: 1px solid #1F2937; border-radius: 6px; padding: 1.1rem 1.4rem; margin-bottom: 1.4rem;">
                <h2 style="font-size: 1.35rem; font-weight: 700; color: #F0F6FC; margin-top: 0; margin-bottom: 0.35rem;">
                    Upload Satellite Imagery for Deforestation Analysis
                </h2>
                <p style="font-size: 0.90rem; color: #94A3B8; line-height: 1.55; margin: 0;">
                    Select an input format below to submit satellite imagery for automated evidence verification and analysis.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        upload_format = st.radio(
            "Select Upload Format:",
            [
                "Type A: Compatible Research Archive (.npz)",
                "Type B: Bi-Temporal Multispectral Files (Two Dates: TIFF / GeoTIFF / NPY)",
                "Type C: Standard RGB Satellite / Aerial Imagery (JPEG / PNG)",
            ],
            index=0,
        )

        # ---------------------------------------------------------------------
        # Upload Category A: .npz Archive
        # ---------------------------------------------------------------------
        if upload_format.startswith("Type A"):
            st.markdown(
                """
                <div style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 0.6rem;">
                    Upload a <code>.npz</code> archive containing either arrays <code>t1</code> and <code>t2</code> (shape <code>(256, 256, 4)</code>)
                    or <code>x</code> (shape <code>(256, 256, 8)</code>) with bands <code>[B02, B03, B04, B08]</code>.
                </div>
                """,
                unsafe_allow_html=True,
            )
            uploaded_file = st.file_uploader(
                "Choose a Sentinel-2 .npz archive:",
                type=["npz"],
                key="uploader_type_a",
            )

            if uploaded_file is not None:
                with st.spinner("Validating input array contracts and spectral channels..."):
                    res = validate_npz_archive(uploaded_file)

                if not res["valid"]:
                    st.error(f"Validation Failure: {res['error']}")
                else:
                    st.success("Evidence Verified: Compatible 8-channel bi-temporal multispectral research archive.")
                    render_unified_analysis_report(
                        patch_id=getattr(uploaded_file, "name", "uploaded_npz"),
                        t1=res["t1"],
                        t2=res["t2"],
                        x_8ch=res["x"],
                        metadata=res["metadata"],
                        y_ref=res["y"],
                        w_mask=res["w"],
                        ground_truth_label=res["label"],
                        reveal_ground_truth=res["has_reference"],
                    )
            else:
                st.info("Awaiting file upload. Select a .npz archive to begin analysis.")

        # ---------------------------------------------------------------------
        # Upload Category B: Paired Multispectral Files
        # ---------------------------------------------------------------------
        elif upload_format.startswith("Type B"):
            st.markdown(
                """
                <div style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 0.6rem;">
                    Upload two multi-band raster files representing Date 1 (Baseline) and Date 2 (Monitoring).
                    Each file must contain 4 spectral bands: Blue (B02), Green (B03), Red (B04), and Near-Infrared (B08).
                </div>
                """,
                unsafe_allow_html=True,
            )
            col_u1, col_u2 = st.columns(2)
            with col_u1:
                file_t1 = st.file_uploader("Date 1 (Before) Multispectral File:", type=["tif", "tiff", "npy", "npz"], key="file_t1")
            with col_u2:
                file_t2 = st.file_uploader("Date 2 (After) Multispectral File:", type=["tif", "tiff", "npy", "npz"], key="file_t2")

            if file_t1 is not None and file_t2 is not None:
                with st.spinner("Validating multispectral dimensions, band alignment, and normalization..."):
                    res = validate_bitemporal_multispectral_files(
                        file_t1=file_t1,
                        file_t2=file_t2,
                        name_t1=getattr(file_t1, "name", "t1_file"),
                        name_t2=getattr(file_t2, "name", "t2_file"),
                    )

                if not res["valid"]:
                    st.error(f"Validation Failure: {res['error']}")
                else:
                    st.success("Evidence Verified: Paired multispectral imagery with required B02, B03, B04, B08 channels.")
                    render_unified_analysis_report(
                        patch_id=f"{getattr(file_t1, 'name', 't1')}+{getattr(file_t2, 'name', 't2')}",
                        t1=res["t1"],
                        t2=res["t2"],
                        x_8ch=res["x"],
                        metadata=res["metadata"],
                        y_ref=None,
                        w_mask=res["w"],
                    )
            else:
                st.info("Upload both Date 1 and Date 2 multispectral files to initiate bi-temporal verification.")

        # ---------------------------------------------------------------------
        # Upload Category C: Standard RGB Imagery
        # ---------------------------------------------------------------------
        else:
            st.markdown(
                """
                <div style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 0.6rem;">
                    Upload standard 3-channel RGB imagery (JPEG, PNG). The system will inspect the submitted evidence
                    and generate an evidence validation diagnostic report.
                </div>
                """,
                unsafe_allow_html=True,
            )
            rgb_files = st.file_uploader(
                "Upload one or two RGB image files:",
                type=["png", "jpg", "jpeg", "tif"],
                accept_multiple_files=True,
                key="uploader_rgb",
            )

            if rgb_files:
                with st.spinner("Inspecting RGB image properties and spectral channels..."):
                    val_rgb = validate_rgb_imagery(
                        files=rgb_files,
                        filenames=[f.name for f in rgb_files],
                    )

                render_evidence_validation_view(val_rgb)
            else:
                st.info("Awaiting RGB image upload. Select standard image files to inspect evidence validity.")

    # Footer
    st.markdown(
        """
        <div style="text-align: center; margin-top: 3.5rem; margin-bottom: 2rem; padding-top: 1.5rem; border-top: 1px solid #1F2937; font-size: 0.8rem; color: #64748B;">
            Explainable AI Based Deforestation Anomaly Detection · Sentinel-2 Bi-Temporal Remote Sensing
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
