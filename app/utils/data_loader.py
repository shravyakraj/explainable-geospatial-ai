"""Data loading and metadata utilities for Streamlit dashboard."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import streamlit as st

from geospatial_ai.dataset import REFLECTANCE_SCALE, load_npz_patch
from geospatial_ai.segmentation_dataset import load_segmentation_npz

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "split_manifest.json"
SEGMENTATION_METRICS_PATH = PROJECT_ROOT / "reports" / "phase5" / "segmentation" / "phase5_segmentation_metrics.json"


@st.cache_data
def load_manifest() -> Dict[str, Any]:
    """Load and cache dataset split manifest."""
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Dataset manifest not found at: {MANIFEST_PATH}")
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def get_usable_patches() -> List[Dict[str, Any]]:
    """Return all usable patches (excluding label -1)."""
    manifest = load_manifest()
    usable = []
    for p in manifest.get("patches", []):
        if int(p.get("label", -1)) in (0, 1):
            p_copy = dict(p)
            p_copy["absolute_path"] = str(PROJECT_ROOT / p["patch_file"])
            usable.append(p_copy)
    return usable


@st.cache_data
def get_curated_demonstration_patches() -> List[Dict[str, Any]]:
    """Return a curated set of representative patches across splits and classes."""
    usable = get_usable_patches()

    # Prominent patches including Phase 5B/5C benchmark test patches
    curated_ids = [
        # Phase 5B/5C Diagnostic Held-Out Test Patches
        "patch_r13_c26",  # High-Confidence Clearing (TP, IoU=0.85, 159 ha)
        "patch_r01_c23",  # Disagreement / Clearing not in PRODES 2021 (FP)
        "patch_r02_c24",  # Clean Primary Forest (TN, 0 ha loss)
        "patch_r14_c27",  # Large deforestation polygon (TP, IoU=0.89, 219 ha)
        # Test split deforestation
        "patch_r18_c09",
        "patch_r19_c10",
        "patch_r19_c11",
        "patch_r20_c09",
        # Test split intact forest
        "patch_r18_c03",
        "patch_r18_c04",
        "patch_r20_c02",
        "patch_r21_c01",
        # Validation examples
        "patch_r14_c04",
        "patch_r14_c10",
        # Training examples
        "patch_r01_c01",
        "patch_r05_c12",
    ]

    patch_dict = {p["patch_id"]: p for p in usable}
    curated = [patch_dict[pid] for pid in curated_ids if pid in patch_dict]

    # If any not found, fill with first 12 usable
    if len(curated) < 6:
        curated = usable[:12]

    return curated


@st.cache_data
def load_patch_data(patch_file_path: str) -> Tuple[np.ndarray, np.ndarray, int]:
    """Load and cache 4-band normalized Sentinel-2 patch data for classification."""
    path = Path(patch_file_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Patch file not found: {path}")
    return load_npz_patch(str(path))


@st.cache_data
def load_segmentation_patch_data(
    patch_file_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and cache 8-channel input, reference mask, and validity mask for segmentation."""
    path = Path(patch_file_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Patch file not found: {path}")
    return load_segmentation_npz(path)


@st.cache_data
def load_official_test_metrics() -> Optional[Dict[str, Any]]:
    """Load official Phase 5B segmentation evaluation benchmark metrics."""
    if not SEGMENTATION_METRICS_PATH.exists():
        logger.warning("Segmentation metrics JSON not found at %s", SEGMENTATION_METRICS_PATH)
        return None
    with open(SEGMENTATION_METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_uploaded_npz(uploaded_file) -> Dict[str, Any]:
    """Validate user-uploaded .npz file strictly against Sentinel-2 input contracts.

    Returns:
        Dictionary with validation status, arrays, and error message if invalid.
    """
    try:
        # Load from buffer or temporary read
        npz_data = np.load(uploaded_file)
        keys = list(npz_data.keys())

        if "t1" in npz_data and "t2" in npz_data:
            t1_raw = npz_data["t1"]
            t2_raw = npz_data["t2"]

            if t1_raw.shape != (256, 256, 4) or t2_raw.shape != (256, 256, 4):
                return {
                    "valid": False,
                    "error": f"Invalid array dimensions: t1 shape is {t1_raw.shape}, t2 shape is {t2_raw.shape}. Expected (256, 256, 4).",
                }

            # Normalize to [0.0, 1.0]
            t1 = np.clip(t1_raw.astype(np.float32) / (REFLECTANCE_SCALE if np.max(t1_raw) > 1.5 else 1.0), 0.0, 1.0)
            t2 = np.clip(t2_raw.astype(np.float32) / (REFLECTANCE_SCALE if np.max(t2_raw) > 1.5 else 1.0), 0.0, 1.0)
            x_8ch = np.concatenate([t1, t2], axis=-1)

        elif "x" in npz_data:
            x_raw = npz_data["x"]
            if x_raw.shape != (256, 256, 8):
                return {
                    "valid": False,
                    "error": f"Invalid x dimension: {x_raw.shape}. Expected (256, 256, 8).",
                }
            x_8ch = np.clip(x_raw.astype(np.float32) / (REFLECTANCE_SCALE if np.max(x_raw) > 1.5 else 1.0), 0.0, 1.0)
            t1 = x_8ch[..., :4]
            t2 = x_8ch[..., 4:]
        else:
            return {
                "valid": False,
                "error": f"Missing required arrays. Uploaded .npz contains keys {keys}, but requires either ('t1', 't2') or ('x').",
            }

        # Check for optional reference mask and exclude mask
        has_reference = False
        y_ref = None
        w_mask = np.ones((256, 256, 1), dtype=np.float32)

        if "mask" in npz_data and "exclude" in npz_data:
            mask_raw = npz_data["mask"]
            exclude_raw = npz_data["exclude"]
            if mask_raw.shape == (256, 256) and exclude_raw.shape == (256, 256):
                has_reference = True
                y_ref = ((mask_raw == 1) & (exclude_raw == 0)).astype(np.float32)[..., np.newaxis]
                w_mask = (exclude_raw == 0).astype(np.float32)[..., np.newaxis]

        label = int(npz_data["label"]) if "label" in npz_data else None

        return {
            "valid": True,
            "t1": t1,
            "t2": t2,
            "x": x_8ch,
            "y": y_ref,
            "w": w_mask,
            "has_reference": has_reference,
            "label": label,
            "keys_found": keys,
        }

    except Exception as e:
        logger.exception("Error validating uploaded .npz file")
        return {
            "valid": False,
            "error": f"Could not parse uploaded file as valid NumPy .npz archive: {e}",
        }


def compute_utm_center_coordinates(bounds_epsg32720: List[float]) -> Dict[str, Any]:
    """Calculate bounding box and approximate center in UTM 20S."""
    minx, miny, maxx, maxy = bounds_epsg32720
    center_x = (minx + maxx) / 2.0
    center_y = (miny + maxy) / 2.0
    width_m = maxx - minx
    height_m = maxy - miny
    area_ha = (width_m * height_m) / 10000.0

    return {
        "utm_easting": round(center_x, 1),
        "utm_northing": round(center_y, 1),
        "width_m": int(width_m),
        "height_m": int(height_m),
        "area_ha": round(area_ha, 2),
        "crs": "EPSG:32720 (WGS 84 / UTM zone 20S)",
    }
