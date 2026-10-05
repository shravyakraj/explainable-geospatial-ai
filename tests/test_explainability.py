"""Comprehensive validation suite for Phase 3 Bi-Temporal Grad-CAM Explainability."""

from pathlib import Path
import numpy as np
import pytest
import tensorflow as tf

from geospatial_ai.dataset import get_split_records, load_npz_patch
from geospatial_ai.inference import ProductionDeforestationClassifier


@pytest.fixture(scope="module")
def classifier():
    return ProductionDeforestationClassifier()


@pytest.fixture(scope="module")
def sample_patches():
    manifest_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "split_manifest.json"
    project_root = Path(__file__).resolve().parent.parent
    records = get_split_records(manifest_path, "test", project_root=project_root)
    t1_a, t2_a, lbl_a = load_npz_patch(records[0]["absolute_path"])
    t1_b, t2_b, lbl_b = load_npz_patch(records[1]["absolute_path"])
    return {
        "patch_a": (t1_a, t2_a, lbl_a, records[0]),
        "patch_b": (t1_b, t2_b, lbl_b, records[1]),
    }


def test_gradcam_output_shape_and_finite(classifier, sample_patches):
    """Property 1, 2, 4: Grad-CAM heatmaps match input spatial dimensions and are strictly finite."""
    t1, t2, _, _ = sample_patches["patch_a"]
    exp = classifier.explain(t1, t2)

    cam_t1 = exp["cam_t1"]
    cam_t2 = exp["cam_t2"]
    change_exp = exp["change_explanation"]

    assert cam_t1.shape == (256, 256), f"Expected (256, 256), got {cam_t1.shape}"
    assert cam_t2.shape == (256, 256), f"Expected (256, 256), got {cam_t2.shape}"
    assert change_exp.shape == (256, 256), f"Expected (256, 256), got {change_exp.shape}"

    assert np.all(np.isfinite(cam_t1)), "T1 Grad-CAM contains non-finite values (NaN/Inf)"
    assert np.all(np.isfinite(cam_t2)), "T2 Grad-CAM contains non-finite values (NaN/Inf)"
    assert np.all(np.isfinite(change_exp)), "Change explanation contains non-finite values"


def test_gradcam_normalized_range(classifier, sample_patches):
    """Property 3: Heatmaps are strictly normalized within documented range [0.0, 1.0]."""
    t1, t2, _, _ = sample_patches["patch_a"]
    exp = classifier.explain(t1, t2)

    for name in ["cam_t1", "cam_t2", "change_explanation"]:
        m = exp[name]
        assert 0.0 <= np.min(m) <= np.max(m) <= 1.0, f"{name} values outside [0.0, 1.0]: [{np.min(m)}, {np.max(m)}]"


def test_gradcam_input_sensitivity(classifier, sample_patches):
    """Property 5: Changing the input patch significantly alters the resulting Grad-CAM heatmap."""
    t1_a, t2_a, _, _ = sample_patches["patch_a"]
    t1_b, t2_b, _, _ = sample_patches["patch_b"]

    exp_a = classifier.explain(t1_a, t2_a)
    exp_b = classifier.explain(t1_b, t2_b)

    # Heatmaps for two distinct geographic patches must not be identical
    diff_t1 = np.max(np.abs(exp_a["cam_t1"] - exp_b["cam_t1"]))
    diff_t2 = np.max(np.abs(exp_a["cam_t2"] - exp_b["cam_t2"]))

    assert diff_t1 > 0.10, f"Grad-CAM T1 failed sensitivity test (diff={diff_t1})"
    assert diff_t2 > 0.10, f"Grad-CAM T2 failed sensitivity test (diff={diff_t2})"


def test_model_weight_invariance_during_explanation(classifier, sample_patches):
    """Property 7: Model weights are strictly immutable during explanation generation."""
    model = classifier.get_model()
    weights_before = [w.numpy().copy() for w in model.weights]

    t1, t2, _, _ = sample_patches["patch_a"]
    _ = classifier.explain(t1, t2)

    weights_after = [w.numpy() for w in model.weights]

    for i, (w_b, w_a) in enumerate(zip(weights_before, weights_after)):
        assert np.array_equal(w_b, w_a), f"Model weight tensor {i} was modified during explain()!"


def test_repeated_inference_determinism(classifier, sample_patches):
    """Property 8: Repeated inference on the exact same input produces bitwise identical results."""
    t1, t2, _, _ = sample_patches["patch_a"]

    exp1 = classifier.explain(t1, t2)
    exp2 = classifier.explain(t1, t2)

    assert exp1["prediction"]["probability"] == exp2["prediction"]["probability"]
    assert np.allclose(exp1["cam_t1"], exp2["cam_t1"], atol=1e-6)
    assert np.allclose(exp1["cam_t2"], exp2["cam_t2"], atol=1e-6)
    assert np.allclose(exp1["change_explanation"], exp2["change_explanation"], atol=1e-6)


def test_production_inference_contract(classifier, sample_patches):
    """Verify structured output fields of inference contract."""
    t1, t2, _, _ = sample_patches["patch_a"]
    res = classifier.predict(t1, t2)

    expected_keys = [
        "prediction",
        "predicted_class",
        "probability",
        "decision_margin",
        "threshold",
        "model_name",
        "model_version",
        "input_shape",
        "bands",
    ]
    for k in expected_keys:
        assert k in res, f"Missing key '{k}' in prediction response"

    assert res["prediction"] in ("Deforestation", "Intact Forest")
    assert res["predicted_class"] in (0, 1)
    assert 0.0 <= res["probability"] <= 1.0
    assert 0.0 <= res["decision_margin"] <= 1.0
    assert res["threshold"] == 0.2700
