"""Unit tests for Phase 2D architecture escalation models."""

import numpy as np
import pytest
import tensorflow as tf

from geospatial_ai.early_fusion_model import build_early_fusion_network
from geospatial_ai.hybrid_fusion_model import build_hybrid_temporal_network
from geospatial_ai.pretrained_models import build_pretrained_siamese_network
from geospatial_ai.spatial_attention_model import build_spatial_attention_change_network


def test_pretrained_mobilenetv3_forward_pass():
    """Verify Pretrained MobileNetV3 Siamese network forward pass and input adapter."""
    model = build_pretrained_siamese_network(backbone_name="mobilenet_v3", freeze_backbone=True)
    assert model.count_params() == 978429

    b_t1 = np.random.rand(2, 256, 256, 4).astype(np.float32)
    b_t2 = np.random.rand(2, 256, 256, 4).astype(np.float32)

    preds = model({"input_t1": b_t1, "input_t2": b_t2}).numpy()
    assert preds.shape == (2, 1)
    assert np.all(preds >= 0.0) and np.all(preds <= 1.0)


def test_early_fusion_forward_pass():
    """Verify Early Fusion 8-channel CNN forward pass."""
    model = build_early_fusion_network(input_shape=(256, 256, 4))
    assert model.count_params() == 408833

    b_t1 = np.random.rand(2, 256, 256, 4).astype(np.float32)
    b_t2 = np.random.rand(2, 256, 256, 4).astype(np.float32)

    preds = model({"input_t1": b_t1, "input_t2": b_t2}).numpy()
    assert preds.shape == (2, 1)
    assert np.all(preds >= 0.0) and np.all(preds <= 1.0)


def test_spatial_attention_change_forward_pass():
    """Verify True Bi-Temporal Spatial Change Attention CNN forward pass."""
    model = build_spatial_attention_change_network(input_shape=(256, 256, 4), include_sum=True)
    assert model.count_params() == 544706

    b_t1 = np.random.rand(2, 256, 256, 4).astype(np.float32)
    b_t2 = np.random.rand(2, 256, 256, 4).astype(np.float32)

    preds = model({"input_t1": b_t1, "input_t2": b_t2}).numpy()
    assert preds.shape == (2, 1)
    assert np.all(preds >= 0.0) and np.all(preds <= 1.0)


def test_hybrid_temporal_fusion_forward_pass():
    """Verify Hybrid Temporal Fusion CNN forward pass."""
    model = build_hybrid_temporal_network(input_shape=(256, 256, 4))
    assert model.count_params() == 595906

    b_t1 = np.random.rand(2, 256, 256, 4).astype(np.float32)
    b_t2 = np.random.rand(2, 256, 256, 4).astype(np.float32)

    preds = model({"input_t1": b_t1, "input_t2": b_t2}).numpy()
    assert preds.shape == (2, 1)
    assert np.all(preds >= 0.0) and np.all(preds <= 1.0)
