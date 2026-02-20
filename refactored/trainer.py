"""
trainer.py - CNN 주식 지수 예측 시스템 학습/평가/예측 모듈

TF1 Estimator API → TF2 Keras fit/evaluate/predict API 로 전면 재작성.

Copyright 2018 Shinhan Financial Group / Bumghi Choi. All Rights Reserved.
Refactored for TensorFlow 2.x / Keras by Claude (2025)
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from config import ModelConfig
from model_builder import build_model, _reshape_inputs

logger = logging.getLogger(__name__)

Array = np.ndarray


# ---------------------------------------------------------------------------
# 학습
# ---------------------------------------------------------------------------

def train(
    config: ModelConfig,
    train_inputs: Array,
    train_labels: Array,
    val_inputs:   Optional[Array] = None,
    val_labels:   Optional[Array] = None,
) -> keras.Model:
    """
    모델을 학습하고 체크포인트를 저장합니다.

    Args:
        config        : ModelConfig
        train_inputs  : 학습 입력 (형상은 model_type 에 따라 다름)
        train_labels  : 학습 레이블
        val_inputs    : 검증 입력 (None이면 train의 20%를 자동 분할)
        val_labels    : 검증 레이블

    Returns:
        학습된 keras.Model
    """
    # 기존 체크포인트 초기화
    model_path = Path(config.model_dir)
    if config.model_reset and model_path.exists():
        shutil.rmtree(model_path)
        logger.info("기존 체크포인트 삭제: %s", model_path)
    model_path.mkdir(parents=True, exist_ok=True)

    # 모델 빌드 or 체크포인트 복원
    ckpt_file = model_path / "model.keras"
    if not config.model_reset and ckpt_file.exists():
        logger.info("체크포인트 복원: %s", ckpt_file)
        model = keras.models.load_model(str(ckpt_file))
    else:
        model = build_model(config)

    # 입력 형상 변환
    X_train = _reshape_inputs(train_inputs, config)
    if val_inputs is not None:
        X_val = _reshape_inputs(val_inputs, config)
        validation_data = (X_val, val_labels)
    else:
        validation_data = None

    # 콜백
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_file),
            save_best_only=True,
            monitor="val_loss" if validation_data else "loss",
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss" if validation_data else "loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss" if validation_data else "loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.TensorBoard(
            log_dir=str(model_path / "logs"),
            histogram_freq=0,
        ),
    ]

    history = model.fit(
        X_train,
        train_labels,
        batch_size=config.batch_size,
        epochs=config.epochs,
        validation_split=0.15 if validation_data is None else 0.0,
        validation_data=validation_data,
        callbacks=callbacks,
        shuffle=True,
        verbose=1,
    )

    logger.info(
        "학습 완료 — 최고 val_loss: %.4f",
        min(history.history.get("val_loss", history.history["loss"])),
    )
    return model


# ---------------------------------------------------------------------------
# 평가 및 예측
# ---------------------------------------------------------------------------

def evaluate_and_predict(
    model: keras.Model,
    config: ModelConfig,
    test_inputs: Array,
    test_labels: Array,
) -> Tuple[Dict[str, float], Array, Array]:
    """
    테스트 세트에 대한 평가 및 예측을 수행합니다.

    Returns:
        metrics      : {'accuracy': float, 'loss': float, ...}
        predictions  : 예측 확률 또는 회귀 값 (N,) or (N, 2)
        pred_classes : 분류 모델의 경우 예측 클래스 (N,), 회귀는 None
    """
    X_test = _reshape_inputs(test_inputs, config)

    raw_eval = model.evaluate(X_test, test_labels, verbose=0)
    metric_names = model.metrics_names
    metrics = dict(zip(metric_names, raw_eval))

    raw_pred = model.predict(X_test, verbose=0)

    if config.model_type.endswith("_reg"):
        predictions  = raw_pred.reshape(-1)
        pred_classes = None
        # RMSE
        rmse = float(np.sqrt(np.mean((predictions - test_labels) ** 2)))
        metrics["rmse"] = rmse
        logger.info("평가 — loss: %.4f  RMSE: %.4f", metrics.get("loss", 0), rmse)
    else:
        predictions  = raw_pred[:, 1]          # 상승 확률
        pred_classes = np.argmax(raw_pred, axis=1).astype(int)
        acc = calculate_metrics(test_labels.astype(int), pred_classes)
        metrics.update(acc)
        logger.info(
            "평가 — loss: %.4f  accuracy: %.4f  F1: %.4f",
            metrics.get("loss", 0),
            acc["accuracy"],
            acc["f1_score"],
        )

    return metrics, predictions, pred_classes


# ---------------------------------------------------------------------------
# 성능 지표 계산
# ---------------------------------------------------------------------------

def calculate_metrics(labels: Array, pred_classes: Array) -> Dict[str, float]:
    """
    정확도, 정밀도, 재현율, F1 점수를 계산합니다.

    Args:
        labels      : 실제 레이블 (0/1 정수)
        pred_classes: 예측 클래스 (0/1 정수)

    Returns:
        dict with keys: accuracy, precision, recall, f1_score,
                        true_positives, true_negatives, false_positives, false_negatives
    """
    tp = int(np.sum((pred_classes == 1) & (labels == 1)))
    tn = int(np.sum((pred_classes == 0) & (labels == 0)))
    fp = int(np.sum((pred_classes == 1) & (labels == 0)))
    fn = int(np.sum((pred_classes == 0) & (labels == 1)))

    total = tp + tn + fp + fn
    accuracy  = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score  = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
    }


# ---------------------------------------------------------------------------
# 결과 저장
# ---------------------------------------------------------------------------

def save_results(
    config: ModelConfig,
    metrics: Dict[str, float],
    predictions: Array,
    pred_classes: Optional[Array],
    test_labels: Array,
    today_dates: Array,
    target_dates: Array,
    today_prices: Array,
) -> str:
    """
    예측 결과와 성능 지표를 CSV 파일로 저장합니다.

    Returns:
        저장된 파일 경로
    """
    Path("results").mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = f"{config.result_prefix}_{timestamp}.csv"

    n = min(len(today_dates), len(predictions))
    result_df = pd.DataFrame({
        "date":        today_dates[:n],
        "pred_date":   target_dates[:n],
        "index_today": today_prices[:n],
        "real":        test_labels[:n].astype(int) if pred_classes is not None else test_labels[:n],
        "pred_class":  pred_classes[:n] if pred_classes is not None else ["N/A"] * n,
        "prediction":  predictions[:n],
    })
    result_df.to_csv(result_path, index=False)

    # 성능 지표 추가
    with open(result_path, "a", encoding="utf-8") as f:
        f.write("\n# === 성능 지표 ===\n")
        for k, v in metrics.items():
            f.write(f"# {k}: {v}\n")

    logger.info("결과 저장 완료: %s", result_path)
    return result_path
