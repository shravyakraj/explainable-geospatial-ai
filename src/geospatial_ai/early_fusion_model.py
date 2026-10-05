"""Early Temporal Fusion (8-channel joint input) Convolutional Neural Network."""

import logging
from typing import Tuple

import keras
from keras import layers, models

logger = logging.getLogger(__name__)


def build_early_fusion_network(
    input_shape: Tuple[int, int, int] = (256, 256, 4),
    dense_units: int = 64,
    dropout_rate: float = 0.4,
    name: str = "early_fusion_8ch_cnn",
) -> keras.Model:
    """Build Early Temporal Fusion CNN where both dates enter the first convolutional layer jointly."""
    input_t1 = layers.Input(shape=input_shape, name="input_t1")
    input_t2 = layers.Input(shape=input_shape, name="input_t2")

    # Early Temporal Fusion: Concatenate along spectral channel dimension -> (256, 256, 8)
    joint_input = layers.Concatenate(axis=-1, name="early_fusion_8ch")([input_t1, input_t2])

    # Block 1: 256x256 -> 128x128
    x = layers.Conv2D(32, (3, 3), padding="same", use_bias=False, name="conv1")(joint_input)
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
    x = layers.ReLU(name="target_relu")(x)

    gap = layers.GlobalAveragePooling2D(name="gap")(x)

    # Classification head
    h = layers.BatchNormalization(name="head_bn")(gap)
    h = layers.Dense(dense_units, activation="relu", name="head_dense")(h)
    h = layers.Dropout(dropout_rate, name="head_dropout")(h)
    output = layers.Dense(1, activation="sigmoid", name="deforestation_prob")(h)

    model = models.Model(inputs=[input_t1, input_t2], outputs=output, name=name)
    logger.info("Built Early Fusion CNN '%s' (parameters: %d)", name, model.count_params())
    return model
