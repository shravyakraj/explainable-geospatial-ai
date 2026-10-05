"""True Bi-Temporal Spatial Change Detection with Spatial Change Attention."""

import logging
from typing import Tuple

import keras
from keras import layers, models, ops

from geospatial_ai.model import build_shared_feature_extractor, AbsoluteDifference, SignedDifference

logger = logging.getLogger(__name__)


def build_spatial_feature_extractor(
    input_shape: Tuple[int, int, int] = (256, 256, 4),
    name: str = "shared_spatial_backbone",
) -> keras.Model:
    """Build shared feature extractor that outputs 2D spatial feature map (32x32x256) before pooling."""
    inputs = layers.Input(shape=input_shape, name="extractor_input")

    # Block 1: 256x256 -> 128x128
    x = layers.Conv2D(32, (3, 3), padding="same", use_bias=False, name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.ReLU(name="relu1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Block 2: 128x128 -> 64x64
    x = layers.Conv2D(64, (3, 3), padding="same", use_bias=False, name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.ReLU(name="relu2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # Block 3: 64x64 -> 32x32
    x = layers.Conv2D(128, (3, 3), padding="same", use_bias=False, name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.ReLU(name="relu3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)

    # Block 4: 32x32 -> target feature map for Grad-CAM
    x = layers.Conv2D(256, (3, 3), padding="same", use_bias=False, name="target_conv")(x)
    x = layers.BatchNormalization(name="target_bn")(x)
    target_relu = layers.ReLU(name="target_relu")(x)

    return models.Model(inputs=inputs, outputs=target_relu, name=name)


def build_spatial_attention_change_network(
    input_shape: Tuple[int, int, int] = (256, 256, 4),
    dense_units: int = 64,
    dropout_rate: float = 0.4,
    include_sum: bool = True,
    name: str = "spatial_attention_change_cnn",
) -> keras.Model:
    """Build True Bi-temporal Spatial Change Detection CNN with Spatial Attention.

    Preserves 2D spatial feature maps before pooling:
    F_change = [|F1 - F2|, (F2 - F1), (F1 + F2)]
    Spatial Attention = sigma(Conv(F_change))
    F_attended = F_change * Spatial Attention
    """
    input_t1 = layers.Input(shape=input_shape, name="input_t1")
    input_t2 = layers.Input(shape=input_shape, name="input_t2")

    backbone = build_spatial_feature_extractor(input_shape=input_shape, name="shared_spatial_backbone")

    # 2D feature maps: (32, 32, 256)
    f1 = backbone(input_t1)
    f2 = backbone(input_t2)

    # Spatial difference representations
    diff_abs = AbsoluteDifference(name="spatial_abs_diff")([f1, f2])
    diff_signed = SignedDifference(name="spatial_signed_diff")([f1, f2])

    if include_sum:
        f_sum = layers.Add(name="spatial_sum")([f1, f2])
        f_change = layers.Concatenate(axis=-1, name="spatial_change_fusion")([diff_abs, diff_signed, f_sum])
    else:
        f_change = layers.Concatenate(axis=-1, name="spatial_change_fusion")([diff_abs, diff_signed])

    # Spatial Change Attention Gate
    att = layers.Conv2D(64, (1, 1), padding="same", use_bias=False, name="att_conv1")(f_change)
    att = layers.BatchNormalization(name="att_bn1")(att)
    att = layers.ReLU(name="att_relu1")(att)
    att_map = layers.Conv2D(1, (3, 3), padding="same", activation="sigmoid", name="spatial_attention_map")(att)

    # Weight spatial features by attention: highlights 'WHERE did the change occur?'
    attended = layers.Multiply(name="attended_change_features")([f_change, att_map])

    # Multi-scale pooling: GAP captures context, GMP captures peak clearing anomaly
    gap = layers.GlobalAveragePooling2D(name="gap")(attended)
    gmp = layers.GlobalMaxPooling2D(name="gmp")(attended)
    latent = layers.Concatenate(name="pool_concat")([gap, gmp])

    # Classification head
    h = layers.BatchNormalization(name="head_bn")(latent)
    h = layers.Dense(dense_units, activation="relu", name="head_dense")(h)
    h = layers.Dropout(dropout_rate, name="head_dropout")(h)
    output = layers.Dense(1, activation="sigmoid", name="deforestation_prob")(h)

    model = models.Model(inputs=[input_t1, input_t2], outputs=output, name=name)
    logger.info("Built Spatial Attention Change CNN '%s' (parameters: %d)", name, model.count_params())
    return model
