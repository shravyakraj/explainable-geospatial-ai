"""Data loading and metadata utilities for Streamlit dashboard."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import streamlit as st

from geospatial_ai.dataset import load_npz_patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "split_manifest.json"


@st.cache_data
def load_manifest() -> Dict[str, Any]:
    """Load and cache dataset split manifest."""
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Dataset manifest not found at: {MANIFEST_PATH}")
    with open(MANIFEST_PATH, "r") as f:
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
    
    # Select prominent test deforestation, test intact, validation, and training examples
    curated_ids = [
        # Test split deforestation (verified high change)
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
    
    # If any not found, fill with first 6 deforest and first 6 intact
    if len(curated) < 6:
        curated = usable[:12]
        
    return curated


@st.cache_data
def load_patch_data(patch_file_path: str) -> Tuple[np.ndarray, np.ndarray, int]:
    """Load and cache 4-band normalized Sentinel-2 patch data."""
    path = Path(patch_file_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Patch file not found: {path}")
    return load_npz_patch(path)


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
