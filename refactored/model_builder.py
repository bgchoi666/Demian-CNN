"""
model_builder.py - CNN 모델 팩토리 모듈

TF1/tf.layers API → TF2/Keras Sequential API 로 전면 재작성.
동일한 아키텍처 로직(동적 레이어 수, Dropout, L2 등)을 유지합니다.

Copyright 2018 Shinhan Financial Group / Bumghi Choi. All Rights Reserved.
Refactored for TensorFlow 2.x / Keras by Claude (2025)
"""

from __future__ import annotations

import logging
from typing import Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from config import ModelConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 공통 레이어 팩토리
# ---------------------------------------------------------------------------

def _dense_block(units: int, dropout: float, l2: float, name_prefix: str = ""):
    """Dense → ReLU → Dropout 블록."""
    return [
        layers.Dense(
            units,
            activation="relu",
            kernel_initializer="glorot_uniform",
            kernel_regularizer=regularizers.l2(l2),
            name=f"{name_prefix}dense_{units}",
        ),
        layers.Dropout(dropout, name=f"{name_prefix}drop_{units}"),
    ]


# ---------------------------------------------------------------------------
# 1D CNN (분류)
# ---------------------------------------------------------------------------

def build_1d_cnn_classifier(config: ModelConfig) -> keras.Model:
    """
    1D CNN 분류 모델.

    입력 형상: (batch, num_steps * input_size)
    출력: 이진 분류 확률 (Softmax 2 units)

    원본 BMS_DL_CNN_1D.py 의 동적 레이어 조건을 재현합니다:
    - Conv + Pool 블록을 flat_size > 3000 조건으로 최대 5회 반복
    - Dense 블록(2000/1000/500)을 flat_size 조건으로 선택적 추가
    """
    seq_len = config.num_steps * config.input_size

    inp = keras.Input(shape=(seq_len,), name="input")
    x = layers.Reshape((seq_len, 1), name="reshape")(inp)

    # ── Conv 블록 (최대 5개, flat_size > 3000 조건) ───────────────────
    flat_size = seq_len  # 초기 추정값
    for i in range(1, 6):
        if flat_size <= 3000 and i > 1:
            break
        x = layers.Conv1D(
            filters=config.filters,
            kernel_size=config.kernel_width,
            strides=config.kernel_stride,
            padding="valid",
            activation="relu",
            kernel_initializer="glorot_uniform",
            name=f"conv{i}",
        )(x)
        x = layers.MaxPooling1D(
            pool_size=config.pool_size,
            strides=config.pool_stride,
            name=f"pool{i}",
        )(x)
        # flat_size 재계산
        flat_size = int(x.shape[1] or 1) * config.filters
        logger.debug("Conv 블록 %d 이후 flat_size 추정: %d", i, flat_size)

    x = layers.Flatten(name="flatten")(x)
    x = layers.Dropout(config.dropout_rate, name="dropout_cnn")(x)

    # ── Dense 블록 (조건부) ────────────────────────────────────────────
    if flat_size > 3000:
        for layer in _dense_block(2000, config.dropout_rate, config.l2_reg, "d1_"):
            x = layer(x)
    if flat_size > 2000:
        for layer in _dense_block(1000, config.dropout_rate, config.l2_reg, "d2_"):
            x = layer(x)
    if flat_size > 1000:
        for layer in _dense_block(500, config.dropout_rate, config.l2_reg, "d3_"):
            x = layer(x)

    for layer in _dense_block(50, 0.2, config.l2_reg, "d4_"):
        x = layer(x)

    out = layers.Dense(2, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inp, outputs=out, name="CNN_1D_Classifier")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    logger.info("1D CNN 분류 모델 생성 완료\n%s", model.summary())
    return model


# ---------------------------------------------------------------------------
# 2D CNN (분류)
# ---------------------------------------------------------------------------

def build_2d_cnn_classifier(config: ModelConfig) -> keras.Model:
    """
    2D CNN 분류 모델.

    입력 형상: (batch, num_steps, input_size)
    출력: 이진 분류 확률 (Softmax 2 units)

    원본 BMS_DL_CNN.py 의 2개 Conv2D 레이어 아키텍처를 Keras 로 재현.
    """
    inp = keras.Input(shape=(config.num_steps, config.input_size, 1), name="input")

    x = layers.Conv2D(
        config.filters,
        kernel_size=(config.kernel_width, config.kernel_width),
        strides=(config.kernel_stride, 1),
        padding="valid",
        activation="relu",
        name="conv1",
    )(inp)
    x = layers.MaxPooling2D(pool_size=(config.pool_size, 1), strides=(config.pool_stride, 1), name="pool1")(x)

    x = layers.Conv2D(
        config.filters,
        kernel_size=(config.kernel_width, 1),
        strides=(config.kernel_stride, 1),
        padding="valid",
        activation="relu",
        name="conv2",
    )(x)
    x = layers.MaxPooling2D(pool_size=(config.pool_size, 1), strides=(config.pool_stride, 1), name="pool2")(x)

    x = layers.Flatten(name="flatten")(x)
    x = layers.Dropout(config.dropout_rate, name="dropout_cnn")(x)

    for layer in _dense_block(50, 0.2, config.l2_reg, "d1_"):
        x = layer(x)

    out = layers.Dense(2, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inp, outputs=out, name="CNN_2D_Classifier")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    logger.info("2D CNN 분류 모델 생성 완료")
    return model


# ---------------------------------------------------------------------------
# 2D CNN (회귀)
# ---------------------------------------------------------------------------

def build_2d_cnn_regressor(config: ModelConfig) -> keras.Model:
    """
    2D CNN 회귀 모델.

    입력 형상: (batch, num_steps, input_size)
    출력: 단일 스칼라 (Linear activation)

    원본 BMS_DL_CNN_v2.py 의 3 Conv2D 레이어 + MSE 손실 아키텍처를 재현.
    """
    inp = keras.Input(shape=(config.num_steps, config.input_size, 1), name="input")

    for i in range(1, 4):
        x = layers.Conv2D(
            config.filters,
            kernel_size=(2, config.kernel_width),
            strides=(1, 1),
            padding="valid",
            activation="relu",
            name=f"conv{i}",
        )(inp if i == 1 else x)
        x = layers.MaxPooling2D(pool_size=(2, 1), strides=(2, 1), name=f"pool{i}")(x)

    x = layers.Flatten(name="flatten")(x)
    x = layers.Dropout(config.dropout_rate, name="dropout_cnn")(x)

    for layer in _dense_block(500, config.dropout_rate, config.l2_reg, "d1_"):
        x = layer(x)
    for layer in _dense_block(50, 0.2, config.l2_reg, "d2_"):
        x = layer(x)

    out = layers.Dense(1, activation="linear", name="output")(x)

    model = keras.Model(inputs=inp, outputs=out, name="CNN_2D_Regressor")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    logger.info("2D CNN 회귀 모델 생성 완료")
    return model


# ---------------------------------------------------------------------------
# 통합 팩토리
# ---------------------------------------------------------------------------

def build_model(config: ModelConfig) -> keras.Model:
    """
    config.model_type 에 따라 적절한 모델을 생성합니다.

    model_type:
        '1d_cls' → 1D CNN 분류 (BMS_DL_CNN_1D.py 호환)
        '2d_cls' → 2D CNN 분류 (BMS_DL_CNN.py 호환)
        '2d_reg' → 2D CNN 회귀 (BMS_DL_CNN_v2.py 호환)
    """
    builders = {
        "1d_cls": build_1d_cnn_classifier,
        "2d_cls": build_2d_cnn_classifier,
        "2d_reg": build_2d_cnn_regressor,
    }
    if config.model_type not in builders:
        raise ValueError(
            f"지원하지 않는 model_type: '{config.model_type}'. "
            f"가능한 값: {list(builders)}"
        )
    return builders[config.model_type](config)


def _reshape_inputs(inputs: "np.ndarray", config: ModelConfig) -> "np.ndarray":
    """
    모델 타입에 맞게 입력 텐서 형상을 변환합니다.

    1d_cls : (N, steps*feats)        → 그대로 반환
    2d_cls : (N, steps, feats)       → (N, steps, feats, 1) 추가 채널
    2d_reg : (N, steps, feats)       → (N, steps, feats, 1) 추가 채널
    """
    import numpy as np
    if config.model_type.startswith("2d"):
        if inputs.ndim == 3:
            return inputs[..., np.newaxis]
        raise ValueError(f"2D 모델 입력은 3D 배열이어야 합니다. 현재: {inputs.shape}")
    return inputs
