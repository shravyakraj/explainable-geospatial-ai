"""Unit tests for Streamlit application components, predictor interfaces, and utilities."""

import io
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pytest

from app.inference.predictor import (
    get_classifier,
    get_segmentation_model,
    run_segmentation_inference,
)
from app.utils.data_loader import (
    compute_utm_center_coordinates,
    get_curated_demonstration_patches,
    get_usable_patches,
    load_official_test_metrics,
    load_patch_data,
    load_segmentation_patch_data,
    validate_uploaded_npz,
)
from app.visualization.image_renderers import (
    array_to_pil_image,
    create_annotated_candidate_regions_figure,
    create_bitemporal_comparison_figure,
    create_comparison_grid_figure,
    create_confusion_map_figure,
    create_display_rgb,
    create_integrated_gradients_figure,
    create_lime_figure,
    create_overlay_figure,
    create_prediction_mask_figure,
    create_segmentation_figure,
    create_spectral_profile_figure,
)
from app.utils.geo_regions import (
    CandidateRegion,
    export_regions_to_geojson,
    extract_candidate_deforestation_areas,
)
from app.utils.input_validator import (
    validate_bitemporal_multispectral_files,
    validate_npz_archive,
    validate_rgb_imagery,
)
from app.utils.report_exporter import (
    generate_deforestation_analysis_html_report,
    generate_evidence_validation_html_report,
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

    # Verify Phase 5 benchmark patches are present
    curated_ids = [p["patch_id"] for p in curated]
    assert "patch_r13_c26" in curated_ids, "Missing benchmark patch_r13_c26"
    assert "patch_r02_c24" in curated_ids, "Missing benchmark patch_r02_c24"


def test_load_segmentation_patch_data():
    """Verify segmentation patch loading returns correct shapes and normalization."""
    curated = get_curated_demonstration_patches()
    sample_path = curated[0]["absolute_path"]

    x, y, w = load_segmentation_patch_data(sample_path)
    assert x.shape == (256, 256, 8)
    assert y.shape == (256, 256, 1)
    assert w.shape == (256, 256, 1)
    assert x.dtype == np.float32
    assert np.all(x >= 0.0) and np.all(x <= 1.0)


def test_load_official_test_metrics():
    """Verify official evaluation benchmark JSON loads correctly."""
    metrics = load_official_test_metrics()
    assert metrics is not None, "Failed to load official test metrics"
    assert "primary_test_set_74_eligible" in metrics
    assert "secondary_test_set_60_classification_clean" in metrics
    prim_unet = metrics["primary_test_set_74_eligible"]["champion_unet"]
    assert np.isclose(prim_unet["global_metrics"]["exact_iou"], 0.6451, atol=1e-3)


def test_validate_uploaded_npz():
    """Verify strict validation of uploaded .npz files."""
    # 1. Valid archive with t1 and t2
    buf = io.BytesIO()
    t1_fake = (np.random.rand(256, 256, 4) * 10000).astype(np.uint16)
    t2_fake = (np.random.rand(256, 256, 4) * 10000).astype(np.uint16)
    mask_fake = np.zeros((256, 256), dtype=np.uint8)
    excl_fake = np.zeros((256, 256), dtype=np.uint8)
    np.savez(buf, t1=t1_fake, t2=t2_fake, mask=mask_fake, exclude=excl_fake, label=1)
    buf.seek(0)

    val_res = validate_uploaded_npz(buf)
    assert val_res["valid"] is True
    assert val_res["x"].shape == (256, 256, 8)
    assert val_res["has_reference"] is True

    # 2. Invalid dimensions
    bad_buf = io.BytesIO()
    np.savez(bad_buf, t1=np.zeros((128, 128, 4), dtype=np.uint16), t2=t2_fake)
    bad_buf.seek(0)
    bad_res = validate_uploaded_npz(bad_buf)
    assert bad_res["valid"] is False
    assert "Invalid array dimensions" in bad_res["error"]

    # 3. Missing keys
    missing_buf = io.BytesIO()
    np.savez(missing_buf, random_key=np.zeros((10, 10)))
    missing_buf.seek(0)
    missing_res = validate_uploaded_npz(missing_buf)
    assert missing_res["valid"] is False
    assert "Missing required arrays" in missing_res["error"]


def test_utm_coordinates_calc():
    """Verify UTM bounding box and area computation."""
    bounds = [412080.0, 9057860.0, 414640.0, 9060420.0]
    info = compute_utm_center_coordinates(bounds)

    assert info["width_m"] == 2560
    assert info["height_m"] == 2560
    assert np.isclose(info["area_ha"], 655.36, atol=0.1)
    assert info["utm_easting"] == 413360.0
    assert info["utm_northing"] == 9059140.0


def test_segmentation_model_and_inference():
    """Verify cached segmentation model forward pass and area computation."""
    model = get_segmentation_model()
    assert model is not None
    assert model.name == "EF_UNet_4B"

    dummy_x = np.random.rand(256, 256, 8).astype(np.float32)
    dummy_w = np.ones((256, 256, 1), dtype=np.float32)

    res = run_segmentation_inference(dummy_x, threshold=0.500, w_mask=dummy_w)
    assert res["probability_map"].shape == (256, 256)
    assert res["binary_mask"].shape == (256, 256)
    assert res["predicted_loss_ha"] >= 0.0
    assert res["valid_area_ha"] == 655.36


def test_image_renderers():
    """Verify PIL conversion and Matplotlib figure creation across all views."""
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

    # Test Segmentation figure
    seg_fig = create_segmentation_figure(dummy_rgb, dummy_cam, dummy_cam > 0.5)
    assert isinstance(seg_fig, plt.Figure)
    plt.close(seg_fig)

    # Test Confusion map figure
    dummy_ref = (np.random.rand(256, 256, 1) > 0.8).astype(np.float32)
    dummy_w = np.ones((256, 256, 1), dtype=np.float32)
    conf_fig, stats = create_confusion_map_figure(dummy_rgb, dummy_ref, dummy_cam > 0.5, dummy_w)
    assert isinstance(conf_fig, plt.Figure)
    assert "iou" in stats and "dice" in stats
    plt.close(conf_fig)

    # Test Integrated Gradients figure
    per_band_attr = {b: {"t1": 10.0, "t2": 20.0} for b in ["B02", "B03", "B04", "B08"]}
    ig_fig = create_integrated_gradients_figure(dummy_cam, dummy_cam, dummy_cam, dummy_w, per_band_attr)
    assert isinstance(ig_fig, plt.Figure)
    plt.close(ig_fig)

    # Test LIME figure
    segments = np.random.randint(0, 35, size=(256, 256))
    lime_fig = create_lime_figure(
        dummy_rgb, segments, dummy_cam > 0.5, dummy_cam < 0.2, [(1, 0.05), (2, -0.03)]
    )
    assert isinstance(lime_fig, plt.Figure)
    plt.close(lime_fig)


def test_production_classifier_singleton():
    """Verify cached classifier loader returns valid, working model instance."""
    clf = get_classifier()
    assert clf is not None
    assert clf.threshold == 0.2700
    assert clf.get_model() is not None


def test_comparison_grid_with_and_without_title():
    """Verify create_comparison_grid_figure accepts title parameter without raising TypeError."""
    dummy_rgb = np.random.rand(256, 256, 3).astype(np.float32)
    dummy_cam = np.random.rand(256, 256).astype(np.float32)

    # 1. Without title
    fig1 = create_comparison_grid_figure(
        rgb_t1=dummy_rgb,
        rgb_t2=dummy_rgb,
        cam_t1=dummy_cam,
        cam_t2=dummy_cam,
        change_exp=dummy_cam,
        delta_ndvi=dummy_cam,
        overlay_t1=dummy_rgb,
        overlay_t2=dummy_rgb,
        overlay_change=dummy_rgb,
    )
    assert isinstance(fig1, plt.Figure)
    plt.close(fig1)

    # 2. With title (reproducing and verifying the fix for the runtime crash)
    fig2 = create_comparison_grid_figure(
        rgb_t1=dummy_rgb,
        rgb_t2=dummy_rgb,
        cam_t1=dummy_cam,
        cam_t2=dummy_cam,
        change_exp=dummy_cam,
        delta_ndvi=dummy_cam,
        overlay_t1=dummy_rgb,
        overlay_t2=dummy_rgb,
        overlay_change=dummy_rgb,
        title="Bi-Temporal Multi-Spectral & Attribution Matrix",
    )
    assert isinstance(fig2, plt.Figure)
    plt.close(fig2)


def test_new_report_visualizers():
    """Verify dedicated report figure renderers."""
    dummy_rgb1 = np.random.rand(256, 256, 3).astype(np.float32)
    dummy_rgb2 = np.random.rand(256, 256, 3).astype(np.float32)
    dummy_prob = np.random.rand(256, 256).astype(np.float32)
    dummy_mask = (dummy_prob >= 0.500).astype(np.float32)
    dummy_w = np.ones((256, 256, 1), dtype=np.float32)
    dummy_w[0:50, 0:50, 0] = 0.0  # Excluded area

    # 1. Bi-temporal side-by-side
    fig_bitemp = create_bitemporal_comparison_figure(dummy_rgb1, dummy_rgb2)
    assert isinstance(fig_bitemp, plt.Figure)
    plt.close(fig_bitemp)

    # 2. Prediction mask figure
    fig_pmask = create_prediction_mask_figure(dummy_prob, dummy_mask, w_mask=dummy_w)
    assert isinstance(fig_pmask, plt.Figure)
    plt.close(fig_pmask)

    # 3. Overlay figure with opacity
    fig_over = create_overlay_figure(dummy_rgb2, dummy_mask, w_mask=dummy_w, alpha=0.55)
    assert isinstance(fig_over, plt.Figure)
    plt.close(fig_over)


def test_input_validator_types():
    """Verify validation and routing across Input Types A, B, and C."""
    # Type A: NPZ archive
    buf_a = io.BytesIO()
    t1_fake = (np.random.rand(256, 256, 4) * 10000).astype(np.uint16)
    t2_fake = (np.random.rand(256, 256, 4) * 10000).astype(np.uint16)
    np.savez(buf_a, t1=t1_fake, t2=t2_fake)
    buf_a.seek(0)
    res_a = validate_npz_archive(buf_a)
    assert res_a["valid"] is True
    assert res_a["supported"] is True
    assert res_a["input_type"] == "TYPE_A_NPZ"
    assert res_a["x"].shape == (256, 256, 8)

    # Type B: Multispectral pair (.npy)
    buf_b1 = io.BytesIO()
    buf_b2 = io.BytesIO()
    np.save(buf_b1, (np.random.rand(256, 256, 4) * 10000).astype(np.uint16))
    np.save(buf_b2, (np.random.rand(256, 256, 4) * 10000).astype(np.uint16))
    buf_b1.seek(0)
    buf_b2.seek(0)
    res_b = validate_bitemporal_multispectral_files(buf_b1, buf_b2, "date1.npy", "date2.npy")
    assert res_b["valid"] is True
    assert res_b["supported"] is True
    assert res_b["input_type"] == "TYPE_B_MULTISPECTRAL"
    assert res_b["x"].shape == (256, 256, 8)

    # Type C: Single RGB Image (JPEG/PNG)
    img_c = Image.fromarray((np.random.rand(256, 256, 3) * 255).astype(np.uint8))
    buf_c = io.BytesIO()
    img_c.save(buf_c, format="PNG")
    buf_c.seek(0)
    res_c = validate_rgb_imagery([buf_c], ["sample.png"])
    assert res_c["valid"] is True
    assert res_c["supported"] is False  # Must NOT be marked supported for inference
    assert res_c["input_type"] == "TYPE_C_RGB"
    assert len(res_c["deficiencies"]) >= 2
    assert "Missing Temporal Baseline" in res_c["deficiencies"][0]


def test_html_report_generation():
    """Verify standalone HTML report generation for both analysis and evidence validation."""
    dummy_rgb = np.random.rand(256, 256, 3).astype(np.float32)
    dummy_fig = plt.figure()

    figures = {
        "bitemporal": dummy_fig,
        "pred_mask": dummy_fig,
        "overlay": dummy_fig,
    }
    meta = {
        "sensor": "Sentinel-2 MSI",
        "spatial_resolution": "10 m",
        "dimensions": "256 × 256 px",
    }
    clf_res = {"prediction": "Deforestation", "probability": 0.85, "threshold": 0.27}
    seg_res = {
        "predicted_loss_ha": 35.2,
        "predicted_loss_pixels": 3520,
        "total_valid_pixels": 65536,
    }

    # Generate Analysis Report HTML
    html = generate_deforestation_analysis_html_report(
        patch_id="test_patch_01",
        metadata=meta,
        clf_result=clf_res,
        seg_result=seg_res,
        figures=figures,
    )
    assert "<!DOCTYPE html>" in html
    assert "test_patch_01" in html
    assert "DEFORESTATION SIGNAL DETECTED" in html
    assert "35.20 ha" in html
    plt.close(dummy_fig)

    # Generate Evidence Validation Report HTML
    val_diag = {
        "input_type": "TYPE_C_RGB",
        "num_dates": 1,
        "deficiencies": ["Missing Near-Infrared band"],
        "requirements": "Provide 4-band Sentinel-2 data",
        "image_metadata": [{"filename": "sample.png", "format": "PNG", "width": 256, "height": 256, "channels": 3, "mode": "RGB"}],
    }
    val_html = generate_evidence_validation_html_report(val_diag)
    assert "<!DOCTYPE html>" in val_html
    assert "Evidence Insufficient for Reliable Deforestation Inference" in val_html


def test_smoke_held_out_patches():
    """Smoke test running complete classification and segmentation inference on held-out test patches."""
    curated = get_curated_demonstration_patches()
    assert len(curated) > 0

    # Pick a high-confidence deforestation patch and an intact forest patch
    patch_tp = next(p for p in curated if p["patch_id"] == "patch_r13_c26")
    patch_tn = next(p for p in curated if p["patch_id"] == "patch_r02_c24")

    # 1. Test Deforestation patch
    x_tp, y_tp, w_tp = load_segmentation_patch_data(patch_tp["absolute_path"])
    seg_tp = run_segmentation_inference(x_tp, threshold=0.500, w_mask=w_tp)
    assert seg_tp["predicted_loss_ha"] > 10.0, f"Expected clearing, got {seg_tp['predicted_loss_ha']}"

    # 2. Test Intact Forest patch
    x_tn, y_tn, w_tn = load_segmentation_patch_data(patch_tn["absolute_path"])
    seg_tn = run_segmentation_inference(x_tn, threshold=0.500, w_mask=w_tn)
    assert seg_tn["predicted_loss_ha"] < 2.0, f"Expected zero/negligible clearing, got {seg_tn['predicted_loss_ha']}"


def test_display_rgb_contrast_and_channel_mapping():
    """Verify create_display_rgb performs robust percentile contrast stretching and correct band mapping."""
    # Simulate dark Sentinel-2 forest reflectance values [0.015, 0.06]
    raw_s2 = np.random.uniform(0.015, 0.06, size=(256, 256, 4)).astype(np.float32)
    # Assign distinct markers to channels to verify mapping
    raw_s2[..., 0] = 0.020  # B02 (Blue)
    raw_s2[..., 1] = 0.035  # B03 (Green)
    raw_s2[..., 2] = 0.050  # B04 (Red)
    raw_s2[..., 3] = 0.350  # B08 (NIR) - must NOT be in display RGB

    # Add random variations to test percentile calculation
    raw_s2[..., :3] += np.random.normal(0, 0.005, size=(256, 256, 3)).astype(np.float32)

    raw_copy = raw_s2.copy()
    disp = create_display_rgb(raw_s2)

    # 1. Input array must NOT be mutated
    assert np.allclose(raw_s2, raw_copy), "create_display_rgb mutated input array!"

    # 2. Output shape and range
    assert disp.shape == (256, 256, 3)
    assert disp.dtype == np.float32
    assert np.all(np.isfinite(disp))
    assert np.min(disp) >= 0.0 and np.max(disp) <= 1.0

    # 3. High contrast: mean must be well within visible range [0.20, 0.80], not pitch black (<0.05)
    assert np.mean(disp) > 0.20, f"Display image is still too dark: mean={np.mean(disp)}"

    # 4. NIR channel is strictly excluded (if channel 3 was displayed as red, it would be ~1.0)
    # Red should correspond to B04
    assert np.mean(disp[..., 0]) < 0.90

    # 5. Safe handling of flat/zero dynamic range bands
    flat_arr = np.ones((256, 256, 4), dtype=np.float32) * 0.05
    flat_disp = create_display_rgb(flat_arr)
    assert np.all(np.isfinite(flat_disp))

    # 6. Safe handling of NaNs and Infs
    corrupt_arr = raw_s2.copy()
    corrupt_arr[0, 0, :] = np.nan
    corrupt_arr[0, 1, :] = np.inf
    corrupt_disp = create_display_rgb(corrupt_arr)
    assert np.all(np.isfinite(corrupt_disp))


def test_display_rgb_joint_scaling():
    """Verify joint contrast stretching produces identical radiometric baseline across two dates."""
    t1 = np.random.uniform(0.015, 0.05, size=(256, 256, 4)).astype(np.float32)
    # T2 has an area of high reflectance clearing in red and blue
    t2 = t1.copy()
    t2[100:150, 100:150, 2] = 0.120  # Cleared red reflectance
    t2[100:150, 100:150, 0] = 0.080  # Cleared blue reflectance

    joint = np.concatenate([t1, t2], axis=0)
    disp1 = create_display_rgb(t1, joint_reference=joint)
    disp2 = create_display_rgb(t2, joint_reference=joint)

    # In undisturbed area (e.g. top-left 0..50), disp1 and disp2 must match closely
    assert np.allclose(disp1[:50, :50, :], disp2[:50, :50, :], atol=1e-3)
    # In cleared area, disp2 must be substantially brighter in red channel than disp1
    assert np.mean(disp2[100:150, 100:150, 0]) > np.mean(disp1[100:150, 100:150, 0]) + 0.3


def test_display_rgb_preserves_natural_color_balance():
    """Verify create_display_rgb preserves physical band ratios (forest is green, soil is warm earth)."""
    # Create realistic canopy reflectance: Green (B03) > Blue (B02) > Red (B04)
    forest = np.zeros((64, 64, 4), dtype=np.float32)
    forest[..., 0] = 0.025  # B02 (Blue)
    forest[..., 1] = 0.045  # B03 (Green - chlorophyll peak)
    forest[..., 2] = 0.022  # B04 (Red - chlorophyll absorption)
    forest[..., 3] = 0.300  # B08 (NIR)

    # Create realistic cleared soil reflectance: Red (B04) > Green (B03) > Blue (B02)
    soil = np.zeros((64, 64, 4), dtype=np.float32)
    soil[..., 0] = 0.065   # B02 (Blue)
    soil[..., 1] = 0.080   # B03 (Green)
    soil[..., 2] = 0.095   # B04 (Red - warm bare earth)
    soil[..., 3] = 0.220   # B08 (NIR)

    joint = np.concatenate([forest, soil], axis=0)
    disp_forest = create_display_rgb(forest, joint_reference=joint, gamma=1.25)
    disp_soil = create_display_rgb(soil, joint_reference=joint, gamma=1.25)

    # 1. Forest Canopy must have Green (channel 1) greater than Red (channel 0) and Blue (channel 2)
    assert np.mean(disp_forest[..., 1]) > np.mean(disp_forest[..., 0])
    assert np.mean(disp_forest[..., 1]) > np.mean(disp_forest[..., 2])

    # 2. Cleared Soil must have Red (channel 0) greater than Blue (channel 2) - avoiding cyan/blue tint
    assert np.mean(disp_soil[..., 0]) > np.mean(disp_soil[..., 2])
    assert np.mean(disp_soil[..., 0]) > 0.60

    # 3. Gamma lift: with gamma=1.25, midtones are lifted compared to linear gamma=1.0
    disp_linear = create_display_rgb(forest, joint_reference=joint, gamma=1.0)
    assert np.mean(disp_forest) > np.mean(disp_linear)


def test_candidate_deforestation_areas_extraction():
    """Verify 8-connected candidate area extraction, ranking, and coordinate computation."""
    prob_map = np.zeros((256, 256), dtype=np.float32)
    binary_pred = np.zeros((256, 256), dtype=bool)

    # Create two discrete clearing regions:
    # Region 1: 100 pixels (1.0 ha at 10m)
    prob_map[10:20, 10:20] = 0.90
    binary_pred[10:20, 10:20] = True

    # Region 2: 400 pixels (4.0 ha at 10m)
    prob_map[100:120, 100:120] = 0.85
    binary_pred[100:120, 100:120] = True

    # Reference mask overlapping Region 2 only
    y_ref = np.zeros((256, 256, 1), dtype=np.float32)
    y_ref[100:120, 100:120, 0] = 1.0

    meta = {
        "patch_bounds_epsg32720": [412080.0, 9057860.0, 414640.0, 9060420.0],
    }

    regions = extract_candidate_deforestation_areas(
        prob_map=prob_map,
        binary_mask=binary_pred,
        y_ref=y_ref,
        metadata=meta,
        pixel_resolution_m=10.0,
    )

    # Must find 2 regions, ranked by size descending
    assert len(regions) == 2
    # Area 01 must be the 400-pixel region (4.0 ha)
    assert regions[0].region_id == "Area 01"
    assert regions[0].pixel_count == 400
    assert np.isclose(regions[0].area_ha, 4.0, atol=0.01)
    assert regions[0].ref_overlap_pixels == 400
    assert "Confirmed PRODES reference loss" in regions[0].ref_status

    # Area 02 must be the 100-pixel region (1.0 ha)
    assert regions[1].region_id == "Area 02"
    assert regions[1].pixel_count == 100
    assert np.isclose(regions[1].area_ha, 1.0, atol=0.01)
    assert regions[1].ref_overlap_pixels == 0

    # Georeferencing checks
    assert regions[0].georeferenced is True
    assert regions[0].crs == "EPSG:32720"
    assert regions[0].centroid_utm is not None
    assert regions[0].centroid_wgs84 is not None

    # Centroid coordinate verification:
    # Region 2 is at rows 100..119, cols 100..119 -> mean row = 109.5, mean col = 109.5
    # top_left_x = 412080.0, top_left_y = 9060420.0
    # Easting = 412080.0 + (109.5 + 0.5) * 10 = 413180.0
    # Northing = 9060420.0 - (109.5 + 0.5) * 10 = 9059320.0
    assert np.isclose(regions[0].centroid_utm[0], 413180.0, atol=1.0)
    assert np.isclose(regions[0].centroid_utm[1], 9059320.0, atol=1.0)


def test_candidate_areas_unprojected_handling():
    """Verify unprojected inputs report coordinates in pixel space and do NOT invent geographic coordinates."""
    prob_map = np.zeros((256, 256), dtype=np.float32)
    binary_pred = np.zeros((256, 256), dtype=bool)
    prob_map[50:70, 50:70] = 0.88
    binary_pred[50:70, 50:70] = True

    # No metadata / bounds
    regions = extract_candidate_deforestation_areas(
        prob_map=prob_map,
        binary_mask=binary_pred,
        metadata={},
    )
    assert len(regions) == 1
    reg = regions[0]
    assert reg.georeferenced is False
    assert reg.centroid_utm is None
    assert reg.centroid_wgs84 is None
    assert reg.centroid_pixel is not None


def test_geojson_export():
    """Verify GeoJSON FeatureCollection generation."""
    prob_map = np.zeros((256, 256), dtype=np.float32)
    binary_pred = np.zeros((256, 256), dtype=bool)
    prob_map[20:40, 20:40] = 0.92
    binary_pred[20:40, 20:40] = True

    meta = {"patch_bounds_epsg32720": [412080.0, 9057860.0, 414640.0, 9060420.0]}
    regions = extract_candidate_deforestation_areas(prob_map, binary_pred, metadata=meta)

    geojson_str = export_regions_to_geojson(regions, patch_id="patch_test_geojson")
    assert geojson_str is not None
    parsed = json.loads(geojson_str)
    assert parsed["type"] == "FeatureCollection"
    assert len(parsed["features"]) == 1
    feat = parsed["features"][0]
    assert feat["properties"]["region_id"] == "Area 01"
    assert feat["properties"]["patch_id"] == "patch_test_geojson"

    # For unprojected regions, export_regions_to_geojson must return None
    unproj_regions = extract_candidate_deforestation_areas(prob_map, binary_pred, metadata={})
    assert export_regions_to_geojson(unproj_regions, "patch_unproj") is None


def test_annotated_candidate_regions_figure():
    """Verify annotated candidate regions figure renderer."""
    dummy_rgb = np.random.rand(256, 256, 3).astype(np.float32)
    prob_map = np.zeros((256, 256), dtype=np.float32)
    binary_pred = np.zeros((256, 256), dtype=bool)
    prob_map[30:60, 30:60] = 0.95
    binary_pred[30:60, 30:60] = True

    regions = extract_candidate_deforestation_areas(prob_map, binary_pred, metadata={})
    fig = create_annotated_candidate_regions_figure(dummy_rgb, regions, binary_pred)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
