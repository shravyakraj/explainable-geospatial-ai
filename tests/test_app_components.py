"""Unit tests for Streamlit application components and utilities."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pytest

from app.inference.predictor import get_classifier
from app.utils.data_loader import (
    compute_utm_center_coordinates,
    get_curated_demonstration_patches,
    get_usable_patches,
    load_patch_data,
)
from app.visualization.image_renderers import (
    array_to_pil_image,
    create_comparison_grid_figure,
    create_spectral_profile_figure,
)


def test_data_loader_usable_patches():
    """Verify usable patch loading and metadata integrity."""
    usable = get_usable_patches()
    assert len(usable) > 0, "No usable patches found"
    
    first = usable[0]
    required_keys = ["patch_id", "split", "label", "patch_bounds_epsg32720", "absolute_path"]
    for k in required_keys:
        assert k in first, f"Missing key '{k}' in usable patch record"
    assert first["label"] in (0, 1), f"Invalid label: {first['label']}"


def test_curated_demonstrations():
    """Verify curated demonstrations include both Deforestation and Intact Forest samples."""
    curated = get_curated_demonstration_patches()
    assert len(curated) >= 6, f"Expected at least 6 curated samples, found {len(curated)}"

    labels = [int(p["label"]) for p in curated]
    assert 0 in labels, "Curated demonstrations missing Intact Forest sample"
    assert 1 in labels, "Curated demonstrations missing Deforestation sample"


def test_utm_coordinates_calc():
    """Verify UTM bounding box and area computation."""
    bounds = [412080.0, 9057860.0, 414640.0, 9060420.0]
    info = compute_utm_center_coordinates(bounds)

    assert info["width_m"] == 2560
    assert info["height_m"] == 2560
    assert np.isclose(info["area_ha"], 655.36, atol=0.1)
    assert info["utm_easting"] == 413360.0
    assert info["utm_northing"] == 9059140.0


def test_image_renderers():
    """Verify PIL conversion and Matplotlib figure creation."""
    dummy_rgb = np.random.rand(256, 256, 3).astype(np.float32)
    pil_img = array_to_pil_image(dummy_rgb)

    assert isinstance(pil_img, Image.Image)
    assert pil_img.size == (256, 256)

    dummy_t1 = np.random.rand(256, 256, 4).astype(np.float32)
    dummy_t2 = np.random.rand(256, 256, 4).astype(np.float32)
    dummy_ndvi1 = np.random.rand(256, 256).astype(np.float32) * 0.8
    dummy_ndvi2 = np.random.rand(256, 256).astype(np.float32) * 0.8
    dummy_cam = np.random.rand(256, 256).astype(np.float32)

    grid_fig = create_comparison_grid_figure(
        rgb_t1=dummy_rgb,
        rgb_t2=dummy_rgb,
        cam_t1=dummy_cam,
        cam_t2=dummy_cam,
        change_exp=dummy_cam,
        delta_ndvi=dummy_ndvi2 - dummy_ndvi1,
        overlay_t1=dummy_rgb,
        overlay_t2=dummy_rgb,
        overlay_change=dummy_rgb,
    )
    assert isinstance(grid_fig, plt.Figure)
    plt.close(grid_fig)

    spec_fig = create_spectral_profile_figure(dummy_t1, dummy_t2, dummy_ndvi1, dummy_ndvi2)
    assert isinstance(spec_fig, plt.Figure)
    plt.close(spec_fig)


def test_production_classifier_singleton():
    """Verify cached classifier loader returns valid, working model instance."""
    clf = get_classifier()
    assert clf is not None
    assert clf.threshold == 0.2700
    assert clf.get_model() is not None
