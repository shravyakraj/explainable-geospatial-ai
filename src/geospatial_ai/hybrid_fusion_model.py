"""Hybrid Temporal Fusion Network for Bi-temporal Deforestation Detection.

Family E: Combines shared temporal encoder, absolute feature difference, signed feature difference,
feature concatenation, spatial attention, and global pooling.
"""

import logging
from typing import Tuple

import keras
from keras import layers, models, ops

from geospatial_ai.model import AbsoluteDifference, SignedDifference
from geospatial_ai.spatial_attention_model import build_spatial_feature_extractor

logger = logging.getLogger(__name__)


def build_hybrid_temporal_network(
    input_shape: Tuple[int, int, int] = (256, 256, 4),
    dense_units: int = 64,
    dropout_rate: float = 0.4,
    name: str = "hybrid_temporal_cnn",
) -> keras.Model:
    """Build Hybrid Temporal Fusion Network.

    Fusion = [F1, F2, |F1-F2|, F2-F1] -> Spatial Attention -> GAP + GMP -> Classifier
    """
    input_t1 = layers.Input(shape=input_shape, name="input_t1")
    input_t2 = layers.Input(shape=input_shape, name="input_t2")

    backbone = build_spatial_feature_extractor(input_shape=input_shape, name="shared_spatial_backbone")

    # 2D feature maps: (32, 32, 256)
    f1 = backbone(input_t1)
    f2 = backbone(input_t2)

    # Differential features
    diff_abs = AbsoluteDifference(name="hybrid_abs_diff")([f1, f2])
    diff_signed = SignedDifference(name="hybrid_signed_diff")([f1, f2])

    # Hybrid concatenation: [F1, F2, |F1-F2|, F2-F1]
    f_fusion = layers.Concatenate(axis=-1, name="hybrid_temporal_concat")([f1, f2, diff_abs, diff_signed])

    # Spatial Change Attention Gate
    att = layers.Conv2D(64, (1, 1), padding="same", use_bias=False, name="hybrid_att_conv1")(f_fusion)
    att = layers.BatchNormalization(name="hybrid_att_bn1")(att)
    att = layers.ReLU(name="hybrid_att_relu1")(att)
    att_map = layers.Conv2D(1, (3, 3), padding="same", activation="sigmoid", name="hybrid_attention_map")(att)

    # Weighted spatial features
    attended = layers.Multiply(name="hybrid_attended_features")([f_fusion, att_map])

    # Multi-scale spatial pooling
    gap = layers.GlobalAveragePooling2D(name="hybrid_gap")(attended)
    gmp = layers.GlobalMaxPooling2D(name="hybrid_gmp")(attended)
    latent = layers.Concatenate(name="hybrid_pool_concat")([gap, gmp])

    # Classification head
    h = layers.BatchNormalization(name="hybrid_head_bn")(latent)
    h = layers.Dense(dense_units, activation="relu", name="hybrid_head_dense")(h)
    h = layers.Dropout(dropout_rate, name="hybrid_head_dropout")(h)
    output = layers.Dense(1, activation="sigmoid", name="deforestation_prob")(h)

    model = models.Model(inputs=[input_t1, input_t2], outputs=output, name=name)
    logger.info("Built Hybrid Temporal Fusion CNN '%s' (parameters: %d)", name, model.count_params())
    return model
