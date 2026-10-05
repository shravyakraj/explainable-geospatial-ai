"""Siamese Convolutional Neural Network architectures for bi-temporal deforestation detection."""

import logging
from typing import Optional, Tuple

import keras
from keras import layers, models, ops, regularizers

logger = logging.getLogger(__name__)


@keras.saving.register_keras_serializable(package="geospatial_ai")
class AbsoluteDifference(layers.Layer):
    """Pairwise absolute feature difference layer: |f(T1) - f(T2)|."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise ValueError("AbsoluteDifference expects a list/tuple of two tensors [t1, t2].")
        return ops.abs(inputs[0] - inputs[1])

    def get_config(self):
        return super().get_config()


@keras.saving.register_keras_serializable(package="geospatial_ai")
class SignedDifference(layers.Layer):
    """Pairwise signed feature difference layer: (f(T2) - f(T1)) representing change direction."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise ValueError("SignedDifference expects a list/tuple of two tensors [t1, t2].")
        return inputs[1] - inputs[0]

    def get_config(self):
        return super().get_config()


def build_shared_feature_extractor(
    input_shape: Tuple[int, int, int] = (256, 256, 4),
    dropout_rate: float = 0.0,
    pooling: str = "gap",
    l2_reg: float = 0.0,
    name: str = "shared_feature_extractor",
) -> keras.Model:
    """Build compact shared convolutional feature extractor for single-date 4-band patch."""
    reg = regularizers.l2(l2_reg) if l2_reg > 0.0 else None
    inputs = layers.Input(shape=input_shape, name="extractor_input")

    # Block 1: 256x256 -> 128x128
    x = layers.Conv2D(32, (3, 3), padding="same", use_bias=False, kernel_regularizer=reg, name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.ReLU(name="relu1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Block 2: 128x128 -> 64x64
    x = layers.Conv2D(64, (3, 3), padding="same", use_bias=False, kernel_regularizer=reg, name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.ReLU(name="relu2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # Block 3: 64x64 -> 32x32
    x = layers.Conv2D(128, (3, 3), padding="same", use_bias=False, kernel_regularizer=reg, name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.ReLU(name="relu3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)

    # Block 4: 32x32 -> target feature map for Grad-CAM
    x = layers.Conv2D(256, (3, 3), padding="same", use_bias=False, kernel_regularizer=reg, name="target_conv")(x)
    x = layers.BatchNormalization(name="target_bn")(x)
    x = layers.ReLU(name="target_relu")(x)

    if dropout_rate > 0.0:
        x = layers.Dropout(dropout_rate, name="extractor_dropout")(x)

    # Multi-scale or standard pooling
    gap = layers.GlobalAveragePooling2D(name="gap")(x)
    if pooling == "gap_gmp":
        gmp = layers.GlobalMaxPooling2D(name="gmp")(x)
        embedding = layers.Concatenate(name="pool_concat")([gap, gmp])
    else:
        embedding = gap

    return models.Model(inputs=inputs, outputs=embedding, name=name)


def build_residual_feature_extractor(
    input_shape: Tuple[int, int, int] = (256, 256, 4),
    dropout_rate: float = 0.1,
    pooling: str = "gap",
    l2_reg: float = 1e-4,
    name: str = "shared_residual_backbone",
) -> keras.Model:
    """Build residual convolutional backbone for deep multi-scale feature representation."""
    reg = regularizers.l2(l2_reg) if l2_reg > 0.0 else None
    inputs = layers.Input(shape=input_shape, name="extractor_input")

    # Initial projection: 256x256 -> 128x128
    x = layers.Conv2D(32, (3, 3), strides=2, padding="same", use_bias=False, kernel_regularizer=reg, name="init_conv")(inputs)
    x = layers.BatchNormalization(name="init_bn")(x)
    x = layers.ReLU(name="init_relu")(x)

    def _res_stage(feat, filters, stride, stage_name):
        shortcut = feat
        if stride != 1 or feat.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, (1, 1), strides=stride, use_bias=False, kernel_regularizer=reg, name=f"{stage_name}_proj")(feat)
            shortcut = layers.BatchNormalization(name=f"{stage_name}_proj_bn")(shortcut)

        c1 = layers.Conv2D(filters, (3, 3), strides=stride, padding="same", use_bias=False, kernel_regularizer=reg, name=f"{stage_name}_c1")(feat)
        b1 = layers.BatchNormalization(name=f"{stage_name}_bn1")(c1)
        r1 = layers.ReLU(name=f"{stage_name}_r1")(b1)

        c2 = layers.Conv2D(filters, (3, 3), padding="same", use_bias=False, kernel_regularizer=reg, name=f"{stage_name}_c2")(r1)
        b2 = layers.BatchNormalization(name=f"{stage_name}_bn2")(c2)

        add = layers.Add(name=f"{stage_name}_add")([shortcut, b2])
        return layers.ReLU(name=f"{stage_name}_out")(add)

    # Stage 1: 128x128 -> 32 filters
    x = _res_stage(x, 32, stride=1, stage_name="s1")
    # Stage 2: 128x128 -> 64x64, 64 filters
    x = _res_stage(x, 64, stride=2, stage_name="s2")
    # Stage 3: 64x64 -> 32x32, 128 filters
    x = _res_stage(x, 128, stride=2, stage_name="s3")
    # Stage 4: 32x32, 256 filters (Target for Grad-CAM)
    x = _res_stage(x, 256, stride=1, stage_name="target_conv")
    # Alias target_relu layer for Grad-CAM
    x = layers.Activation("linear", name="target_relu")(x)

    if dropout_rate > 0.0:
        x = layers.Dropout(dropout_rate, name="extractor_dropout")(x)

    gap = layers.GlobalAveragePooling2D(name="gap")(x)
    if pooling == "gap_gmp":
        gmp = layers.GlobalMaxPooling2D(name="gmp")(x)
        embedding = layers.Concatenate(name="pool_concat")([gap, gmp])
    else:
        embedding = gap

    return models.Model(inputs=inputs, outputs=embedding, name=name)


def build_siamese_network(
    input_shape: Tuple[int, int, int] = (256, 256, 4),
    dense_units: int = 64,
    dropout_rate: float = 0.4,
    include_concatenation: bool = False,
    backbone_type: str = "conv",
    pooling: str = "gap",
    fusion_type: str = "abs_diff",
    l2_reg: float = 0.0,
    name: str = "siamese_deforestation_cnn",
) -> keras.Model:
    """Build complete Siamese CNN with configurable backbone and temporal fusion."""
    input_t1 = layers.Input(shape=input_shape, name="input_t1")
    input_t2 = layers.Input(shape=input_shape, name="input_t2")

    # Select feature extractor
    if backbone_type == "resnet":
        feature_extractor = build_residual_feature_extractor(
            input_shape=input_shape,
            dropout_rate=0.1,
            pooling=pooling,
            l2_reg=l2_reg,
            name="shared_backbone",
        )
    else:
        feature_extractor = build_shared_feature_extractor(
            input_shape=input_shape,
            dropout_rate=0.1 if include_concatenation else 0.0,
            pooling=pooling,
            l2_reg=l2_reg,
            name="shared_backbone",
        )

    feat_t1 = feature_extractor(input_t1)
    feat_t2 = feature_extractor(input_t2)

    # Absolute difference
    diff_abs = AbsoluteDifference(name="abs_feature_difference")([feat_t1, feat_t2])

    # Fusion strategy
    if include_concatenation or fusion_type == "concat_state":
        fused = layers.Concatenate(name="feature_fusion")([diff_abs, feat_t1, feat_t2])
    elif fusion_type == "directional":
        diff_signed = SignedDifference(name="signed_feature_difference")([feat_t1, feat_t2])
        fused = layers.Concatenate(name="feature_fusion")([diff_abs, diff_signed])
    else:
        fused = diff_abs

    # Classification head with BatchNormalization
    x = layers.BatchNormalization(name="fusion_bn")(fused)
    x = layers.Dense(dense_units, activation="relu", kernel_regularizer=regularizers.l2(l2_reg) if l2_reg > 0 else None, name="head_dense")(x)
    x = layers.Dropout(dropout_rate, name="head_dropout")(x)
    output = layers.Dense(1, activation="sigmoid", name="deforestation_prob")(x)

    model = models.Model(inputs=[input_t1, input_t2], outputs=output, name=name)
    logger.info("Built Siamese model '%s' (type=%s, pooling=%s, fusion=%s, params=%d)",
                name, backbone_type, pooling, fusion_type, model.count_params())
    return model
