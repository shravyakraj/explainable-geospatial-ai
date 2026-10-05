"""Unit tests for Phase 2B modeling, dataset pipeline, metrics, and explainability."""

import json
from pathlib import Path
import numpy as np
import pytest
import tensorflow as tf

from geospatial_ai.dataset import get_split_records, load_npz_patch, REFLECTANCE_SCALE
from geospatial_ai.explainability import compute_gradcam, compute_ndvi
from geospatial_ai.metrics import compute_wilson_ci, compute_confusion_matrix, evaluate_predictions
from geospatial_ai.model import build_siamese_network, build_shared_feature_extractor


@pytest.fixture
def project_root():
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def manifest_path(project_root):
    return project_root / "data" / "processed" / "split_manifest.json"


def test_dataset_label_filtering(manifest_path, project_root):
    """Verify that get_split_records strictly excludes label -1 and includes only 0 and 1."""
    if not manifest_path.exists():
        pytest.skip("Manifest not generated yet")

    for split in ["train", "val", "test"]:
        records = get_split_records(manifest_path, split, project_root=project_root)
        assert len(records) > 0, f"Split {split} should have usable records"
        labels = [int(r["label"]) for r in records]
        assert -1 not in labels, f"Label -1 found in usable records for {split}"
        assert all(lbl in (0, 1) for lbl in labels), f"Invalid labels found in {split}"


def test_patch_normalization_and_shapes(manifest_path, project_root):
    """Verify that load_npz_patch returns arrays of shape (256, 256, 4) scaled to [0, 1]."""
    if not manifest_path.exists():
        pytest.skip("Manifest not generated yet")

    records = get_split_records(manifest_path, "train", project_root=project_root)
    t1, t2, lbl = load_npz_patch(records[0]["absolute_path"])

    assert t1.shape == (256, 256, 4)
    assert t2.shape == (256, 256, 4)
    assert t1.dtype == np.float32
    assert t2.dtype == np.float32
    assert 0.0 <= np.min(t1) <= np.max(t1) <= 1.0
    assert 0.0 <= np.min(t2) <= np.max(t2) <= 1.0
    assert lbl in (0, 1)


def test_siamese_architecture_forward_pass():
    """Verify forward pass of both M1 and M2 Siamese networks."""
    # M1: Difference only
    m1 = build_siamese_network(input_shape=(256, 256, 4), include_concatenation=False)
    assert m1.count_params() == 407681

    # M2: Difference + Concatenation
    m2 = build_siamese_network(input_shape=(256, 256, 4), include_concatenation=True)
    assert m2.count_params() == 442497

    batch_t1 = np.random.rand(2, 256, 256, 4).astype(np.float32)
    batch_t2 = np.random.rand(2, 256, 256, 4).astype(np.float32)

    out1 = m1([batch_t1, batch_t2]).numpy()
    out2 = m2([batch_t1, batch_t2]).numpy()

    assert out1.shape == (2, 1)
    assert out2.shape == (2, 1)
    assert np.all(out1 >= 0.0) and np.all(out1 <= 1.0)
    assert np.all(out2 >= 0.0) and np.all(out2 <= 1.0)


def test_siamese_zero_difference_symmetry():
    """Verify that when T1 == T2, absolute difference layer produces an exact zero vector."""
    m1 = build_siamese_network(input_shape=(256, 256, 4), include_concatenation=False)
    sample_t = np.random.rand(1, 256, 256, 4).astype(np.float32)

    backbone = m1.get_layer("shared_backbone")
    f1 = backbone(sample_t)
    f2 = backbone(sample_t)
    diff = m1.get_layer("abs_feature_difference")([f1, f2]).numpy()

    assert np.allclose(diff, 0.0, atol=1e-6)


def test_metrics_calculations():
    """Verify confusion matrix, accuracy, Wilson score CI, and edge cases."""
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.1, 0.2])

    res = evaluate_predictions(y_true, y_prob, threshold=0.5)
    assert res["accuracy"] == 1.0
    assert res["precision"] == 1.0
    assert res["recall"] == 1.0
    assert res["f1_score"] == 1.0
    assert res["confusion_matrix"] == {"tp": 2, "fp": 0, "tn": 2, "fn": 0, "total": 4}

    # Wilson CI for 4/4 should be [0.51, 1.0]
    ci = compute_wilson_ci(4, 4)
    assert ci[0] > 0.45 and ci[1] == 1.0


def test_gradcam_output_shape_and_range():
    """Verify Grad-CAM produces heatmaps of shape (256, 256) normalized to [0, 1]."""
    m1 = build_siamese_network(input_shape=(256, 256, 4), include_concatenation=False)
    t1 = np.random.rand(256, 256, 4).astype(np.float32) * 0.4
    t2 = np.random.rand(256, 256, 4).astype(np.float32) * 0.4

    cam1, cam2, prob = compute_gradcam(m1, t1, t2)
    assert cam1.shape == (256, 256)
    assert cam2.shape == (256, 256)
    assert 0.0 <= np.min(cam1) <= np.max(cam1) <= 1.0
    assert 0.0 <= np.min(cam2) <= np.max(cam2) <= 1.0
    assert 0.0 <= prob <= 1.0


def test_ndvi_computation():
    """Verify NDVI computation adheres to standard formula."""
    patch = np.zeros((256, 256, 4), dtype=np.float32)
    patch[..., 2] = 0.1  # Red = 0.1
    patch[..., 3] = 0.5  # NIR = 0.5

    ndvi = compute_ndvi(patch)
    expected = (0.5 - 0.1) / (0.5 + 0.1 + 1e-6)
    assert np.allclose(ndvi, expected, atol=1e-4)
