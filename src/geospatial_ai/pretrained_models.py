"""Pretrained ImageNet backbones adapted for 4-band Sentinel-2 bi-temporal classification."""

import logging
from typing import Tuple

import keras
from keras import layers, models, ops
import numpy as np
import tensorflow as tf

from geospatial_ai.model import AbsoluteDifference

logger = logging.getLogger(__name__)


def build_pretrained_siamese_network(
    backbone_name: str = "efficientnet_b0",
    input_shape: Tuple[int, int, int] = (256, 256, 4),
    dense_units: int = 64,
    dropout_rate: float = 0.4,
    freeze_backbone: bool = True,
    unfreeze_top_n: int = 0,
    name: str = "pretrained_siamese",
) -> keras.Model:
    """Build complete Siamese CNN using adapted pretrained ImageNet backbone with sensible NIR initialization."""
    inp_t1 = layers.Input(shape=input_shape, name="input_t1")
    inp_t2 = layers.Input(shape=input_shape, name="input_t2")

    # Spectral adapter: 4 channels (B02, B03, B04, B08) -> 3 channels (R, G, B)
    adapter = layers.Conv2D(3, (1, 1), padding="same", use_bias=False, name="spectral_adapter_4to3")

    # Sensible initial weights: (1, 1, 4, 3)
    # Output 0 (Red)   <- Red (B04, ch2) + 0.3*NIR (B08, ch3)
    # Output 1 (Green) <- Green (B03, ch1) + 0.3*NIR (B08, ch3)
    # Output 2 (Blue)  <- Blue (B02, ch0)
    init_w = np.zeros((1, 1, 4, 3), dtype=np.float32)
    init_w[0, 0, 2, 0] = 1.0  # B04 -> Red
    init_w[0, 0, 3, 0] = 0.3  # B08 -> Red
    init_w[0, 0, 1, 1] = 1.0  # B03 -> Green
    init_w[0, 0, 3, 1] = 0.3  # B08 -> Green
    init_w[0, 0, 0, 2] = 1.0  # B02 -> Blue

    adapter.build((None, input_shape[0], input_shape[1], input_shape[2]))
    adapter.set_weights([init_w])
    # Adapter weights can remain trainable so network can fine-tune spectral mixture
    adapter.trainable = True

    # Pretrained base model
    if backbone_name == "efficientnet_b0":
        base = keras.applications.EfficientNetB0(
            weights="imagenet",
            include_top=False,
            input_shape=(input_shape[0], input_shape[1], 3),
        )
        rescaling = layers.Rescaling(255.0, name="rescale_to_255")
    else:
        base = keras.applications.MobileNetV3Small(
            weights="imagenet",
            include_top=False,
            input_shape=(input_shape[0], input_shape[1], 3),
        )
        rescaling = layers.Rescaling(255.0, name="rescale_to_255")

    if freeze_backbone and unfreeze_top_n == 0:
        base.trainable = False
    elif unfreeze_top_n > 0:
        base.trainable = True
        for layer in base.layers[:-unfreeze_top_n]:
            layer.trainable = False
        for layer in base.layers[-unfreeze_top_n:]:
            layer.trainable = True
    else:
        base.trainable = True

    # Build shared feature extractor
    inp_single = layers.Input(shape=input_shape, name="extractor_input")
    x = adapter(inp_single)
    x = rescaling(x)
    feat_map = base(x)
    target_relu = layers.Activation("linear", name="target_relu")(feat_map)
    gap = layers.GlobalAveragePooling2D(name="gap")(target_relu)

    backbone = models.Model(inputs=inp_single, outputs=gap, name="shared_backbone")

    f1 = backbone(inp_t1)
    f2 = backbone(inp_t2)

    diff = AbsoluteDifference(name="abs_feature_difference")([f1, f2])
    x = layers.BatchNormalization(name="fusion_bn")(diff)
    x = layers.Dense(dense_units, activation="relu", name="head_dense")(x)
    x = layers.Dropout(dropout_rate, name="head_dropout")(x)
    output = layers.Dense(1, activation="sigmoid", name="deforestation_prob")(x)

    model = models.Model(inputs=[inp_t1, inp_t2], outputs=output, name=name)
    trainable_count = sum(np.prod(p.shape) for p in model.trainable_weights)
    logger.info("Built Pretrained Siamese '%s' (%s, total params: %d, trainable: %d)",
                name, backbone_name, model.count_params(), trainable_count)
    return model
