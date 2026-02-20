"""
data_reader.py - CNN 주식 지수 예측 시스템 데이터 로딩/전처리 모듈

원본 reader.py / reader_1D.py / reader_v2.py 를 하나의 모듈로 통합하고
TF1 의존성을 제거하였습니다.

Copyright 2018 Shinhan Financial Group / Bumghi Choi. All Rights Reserved.
Refactored for TensorFlow 2.x / Keras by Claude (2025)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import ModelConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 타입 별칭
# ---------------------------------------------------------------------------
Array = np.ndarray
DataBundle = Tuple[Array, Array, Array, Array, Array, Array]
# (train_inputs, train_labels, test_inputs, test_labels, today_dates, target_dates)


# ---------------------------------------------------------------------------
# 정규화 유틸
# ---------------------------------------------------------------------------

def normalize(df: Array) -> Array:
    """StandardScaler(Z-score)로 피처 행렬을 정규화합니다."""
    scaler = StandardScaler()
    return scaler.fit_transform(df)


# ---------------------------------------------------------------------------
# 시계열 → CNN 입력 변환
# ---------------------------------------------------------------------------

def _make_sequences_2d(
    feature_matrix: Array,
    label_vector: Array,
    num_steps: int,
    step_interval: int,
) -> Tuple[Array, Array]:
    """
    2D CNN 용 시퀀스 생성.

    각 타임스텝의 피처를 행(row)으로 쌓아 (num_steps × input_size) 형태 행렬 생성.

    Args:
        feature_matrix : (T, input_size) 정규화된 피처
        label_vector   : (T,) 타겟 레이블
        num_steps      : CNN 입력 시퀀스 길이
        step_interval  : 스텝 간 간격 (일)

    Returns:
        X : (N, num_steps, input_size)
        y : (N,)
    """
    input_size = feature_matrix.shape[1]
    size = len(feature_matrix) - step_interval * (num_steps - 1)

    dataX, dataY = [], []
    for i in range(size):
        idx = list(range(i, i + num_steps * step_interval, step_interval))
        dataX.append(feature_matrix[idx, :input_size])          # (num_steps, input_size)
        dataY.append(label_vector[i + step_interval * (num_steps - 1)])

    X = np.array(dataX, dtype=np.float32)   # (N, num_steps, input_size)
    y = np.array(dataY, dtype=np.float32)   # (N,)
    return X, y


def _make_sequences_1d(
    feature_matrix: Array,
    label_vector: Array,
    num_steps: int,
    step_interval: int,
) -> Tuple[Array, Array]:
    """
    1D CNN 용 시퀀스 생성.

    각 피처의 시계열을 이어 붙여 (num_steps × input_size,) 1D 벡터 생성.
    원본 reader_1D.py 의 전치(Transpose) 방식과 동일합니다.

    Args:
        feature_matrix : (T, input_size) 정규화된 피처
        label_vector   : (T,) 타겟 레이블
        num_steps      : CNN 입력 시퀀스 길이
        step_interval  : 스텝 간 간격 (일)

    Returns:
        X : (N, num_steps * input_size)
        y : (N,)
    """
    input_size = feature_matrix.shape[1]
    size = len(feature_matrix) - step_interval * (num_steps - 1)

    dataX, dataY = [], []
    for i in range(size):
        idx = list(range(i, i + num_steps * step_interval, step_interval))
        # 전치: (input_size, num_steps) → flatten → (input_size * num_steps,)
        dataX.append(feature_matrix[idx, :input_size].T.reshape(-1))
        dataY.append(label_vector[i + step_interval * (num_steps - 1)])

    X = np.array(dataX, dtype=np.float32)   # (N, num_steps * input_size)
    y = np.array(dataY, dtype=np.float32)   # (N,)
    return X, y


# ---------------------------------------------------------------------------
# 레이블 생성
# ---------------------------------------------------------------------------

def _make_classification_labels(price_series: Array, far_predict: int) -> Array:
    """
    분류 레이블 생성: far_predict 이후 가격 > 현재 가격 → 1(상승), 그 외 → 0(하락).

    Args:
        price_series : 원본(비정규화) 가격 시계열 (T,)
        far_predict  : 예측 기간 (영업일)

    Returns:
        labels : (T - far_predict,)  0 또는 1
    """
    n = len(price_series) - far_predict
    labels = np.where(
        price_series[far_predict: far_predict + n] > price_series[:n],
        1, 0
    ).astype(np.float32)
    return labels


def _make_regression_labels(
    price_series: Array,
    far_predict: int,
    conversion: str = "diff",
) -> Array:
    """
    회귀 레이블 생성.

    Args:
        price_series : 원본(비정규화) 가격 시계열 (T,)
        far_predict  : 예측 기간 (영업일)
        conversion   : 변환 방식
            'diff' → 절대 차이값
            'rate' → 변화율(%)
            'norm' → 고정 정규화 (원본 코드 호환)

    Returns:
        labels : (T - far_predict,)
    """
    n = len(price_series) - far_predict
    current = price_series[:n].astype(np.float64)
    future  = price_series[far_predict: far_predict + n].astype(np.float64)

    if conversion == "diff":
        labels = (future - current).astype(np.float32)
    elif conversion == "rate":
        labels = ((future - current) / current * 100).astype(np.float32)
    elif conversion == "norm":
        labels = ((future - 67) / 30).astype(np.float32)
    else:
        raise ValueError(f"알 수 없는 conversion 방식: '{conversion}'")
    return labels


# ---------------------------------------------------------------------------
# 메인 데이터 로더
# ---------------------------------------------------------------------------

def load_data(config: ModelConfig, data_dir: str = ".") -> DataBundle:
    """
    CSV 파일을 읽어 CNN 학습/테스트 데이터를 반환합니다.

    CSV 파일 형식:
        컬럼 0    : date (YYYY-MM-DD)
        컬럼 1~N  : 기술적 지표 피처 (N = config.input_size)
        컬럼 N+1  : 종가 지수 (price / close)

    Args:
        config   : ModelConfig 인스턴스
        data_dir : CSV 파일이 위치한 디렉토리

    Returns:
        train_inputs  : (N_train, ...) 학습 입력
        train_labels  : (N_train,)    학습 레이블
        test_inputs   : (N_test,  ...) 테스트 입력
        test_labels   : (N_test,)     테스트 레이블
        today_dates   : (N_test,)     예측 시도일 문자열 배열
        target_dates  : (N_test,)     예측 대상일 문자열 배열
    """
    csv_path = Path(data_dir) / f"raw_data_{config.file_name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"데이터 파일을 찾을 수 없습니다: {csv_path}\n"
            "raw_data_<file_name>.csv 형식으로 파일을 준비하세요."
        )

    raw_df = pd.read_csv(csv_path, encoding="ISO-8859-1")
    logger.info("데이터 로드 완료: %s  rows=%d", csv_path, len(raw_df))

    # 피처 정규화
    feature_cols = raw_df.values[:, 1 : config.input_size + 1].astype(np.float64)
    features_normalized = normalize(feature_cols).astype(np.float32)

    # 가격(종가) 시계열 — 레이블 생성에 사용
    price_series = raw_df.values[:, config.input_size + 1].astype(np.float64)

    # 테스트 구간 인덱스 계산
    test_start_idx = len(raw_df[raw_df["date"] <= config.test_start]) - 1 - config.far_predict
    test_end_idx   = len(raw_df[raw_df["date"] <= config.test_end])   - 1 - config.far_predict

    if test_start_idx < 0:
        raise ValueError("test_start 날짜가 데이터 범위를 벗어납니다.")

    # ------------------------------------------------------------------
    # 학습 데이터
    # ------------------------------------------------------------------
    train_feature = features_normalized[: test_start_idx - config.far_predict]
    train_price   = price_series        [: test_start_idx]

    if config.model_type.endswith("_reg"):
        train_label_seq = _make_regression_labels(train_price, config.far_predict, config.conversion)
    else:
        train_label_seq = _make_classification_labels(train_price, config.far_predict)

    # ------------------------------------------------------------------
    # 테스트 데이터
    # ------------------------------------------------------------------
    lead = config.step_interval * (config.num_steps - 1)
    test_feature_raw = features_normalized[test_start_idx - lead : test_end_idx + 1]
    test_price_raw   = price_series        [test_start_idx - lead : test_end_idx + 1 + config.far_predict]

    if config.model_type.endswith("_reg"):
        test_label_seq = _make_regression_labels(test_price_raw, config.far_predict, config.conversion)
    else:
        test_label_seq = _make_classification_labels(test_price_raw, config.far_predict)

    # ------------------------------------------------------------------
    # 시퀀스 생성
    # ------------------------------------------------------------------
    seq_fn = _make_sequences_1d if config.model_type.startswith("1d") else _make_sequences_2d

    train_inputs, train_labels = seq_fn(
        np.concatenate([train_feature, train_label_seq.reshape(-1, 1)], axis=1)
        if False  # 레이블 열을 합치지 않는 방식으로 수정
        else _concat_labels(train_feature, train_label_seq),
        train_label_seq,
        config.num_steps,
        config.step_interval,
    )
    # 위 방식 대신 피처/레이블을 분리하여 직접 생성
    train_inputs, train_labels = _build_sequences(
        train_feature, train_label_seq, config, seq_fn
    )
    test_inputs, test_labels = _build_sequences(
        test_feature_raw, test_label_seq, config, seq_fn
    )

    # ------------------------------------------------------------------
    # 날짜 / 지수 메타데이터
    # ------------------------------------------------------------------
    today_dates  = raw_df.values[test_start_idx : test_end_idx + 1, 0]
    target_dates = raw_df.values[
        test_start_idx + config.far_predict : test_end_idx + config.far_predict + 1, 0
    ]
    today_prices = price_series[test_start_idx : test_end_idx + 1]

    logger.info(
        "데이터 분할 완료 — train: %d  test: %d",
        len(train_labels), len(test_labels),
    )

    return (
        train_inputs, train_labels,
        test_inputs,  test_labels,
        today_dates,  target_dates,
        today_prices,
    )


# ---------------------------------------------------------------------------
# 내부 헬퍼
# ---------------------------------------------------------------------------

def _concat_labels(features: Array, labels: Array) -> Array:
    """피처 행렬과 레이블을 마지막 열로 합칩니다."""
    return np.concatenate([features, labels.reshape(-1, 1)], axis=1)


def _build_sequences(
    features: Array,
    labels: Array,
    config: ModelConfig,
    seq_fn,
) -> Tuple[Array, Array]:
    """
    (features, labels) 쌍에서 CNN 시퀀스를 생성합니다.
    labels 배열의 길이는 features 길이와 같아야 합니다.
    """
    num_steps    = config.num_steps
    step_interval = config.step_interval

    size = len(features) - step_interval * (num_steps - 1)
    if size <= 0:
        raise ValueError(
            f"데이터가 너무 짧습니다. "
            f"필요: {step_interval * (num_steps - 1) + 1}  실제: {len(features)}"
        )

    if config.model_type.startswith("1d"):
        return _make_sequences_1d(features, labels, num_steps, step_interval)
    else:
        return _make_sequences_2d(features, labels, num_steps, step_interval)
