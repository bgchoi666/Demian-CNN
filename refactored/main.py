"""
main.py - CNN 주식 지수 예측 시스템 메인 실행 스크립트

사용법:
    # 기본 실행 (config.py 의 기본값 사용)
    python main.py

    # 모델 타입 / 종목 / 기간 직접 지정
    python main.py --model 1d_cls --file kospi --test_start 2016-07-01 --test_end 2017-06-30

    # 회귀 모드
    python main.py --model 2d_reg --file kospi200f_all --conversion diff

    # 멀티기간 테스트 (병렬 실행)
    python main.py --multi_period --file kospi --model 1d_cls

Copyright 2018 Shinhan Financial Group / Bumghi Choi. All Rights Reserved.
Refactored for TensorFlow 2.x / Keras by Claude (2025)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import concurrent.futures
from datetime import datetime
from typing import List, Tuple

import numpy as np
import tensorflow as tf

# 로컬 모듈
from config import ModelConfig
from data_reader import load_data
from trainer import train, evaluate_and_predict, save_results

# ---------------------------------------------------------------------------
# 로깅 설정
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 단일 기간 실행
# ---------------------------------------------------------------------------

def run_single(config: ModelConfig, data_dir: str = ".") -> str:
    """
    단일 테스트 기간에 대해 데이터 로드 → 학습 → 평가 → 저장을 수행합니다.

    Args:
        config   : ModelConfig
        data_dir : CSV 파일 디렉토리

    Returns:
        저장된 결과 파일 경로
    """
    logger.info(
        "=== 실행 시작 | 모델: %s | 파일: %s | 예측기간: %d일 | 테스트: %s ~ %s ===",
        config.model_type, config.file_name, config.far_predict,
        config.test_start, config.test_end,
    )

    # 1. 데이터 로드
    (
        train_inputs, train_labels,
        test_inputs,  test_labels,
        today_dates,  target_dates,
        today_prices,
    ) = load_data(config, data_dir=data_dir)

    logger.info(
        "데이터 로드 완료 — train: %s  test: %s",
        train_inputs.shape, test_inputs.shape,
    )

    # 2. 학습
    model = train(
        config,
        train_inputs, train_labels,
    )

    # 3. 평가 및 예측
    metrics, predictions, pred_classes = evaluate_and_predict(
        model, config, test_inputs, test_labels,
    )

    # 4. 결과 저장
    result_path = save_results(
        config,
        metrics,
        predictions,
        pred_classes,
        test_labels,
        today_dates,
        target_dates,
        today_prices,
    )

    logger.info("=== 실행 완료 | 결과: %s ===", result_path)
    return result_path


# ---------------------------------------------------------------------------
# 멀티기간 실행
# ---------------------------------------------------------------------------

def run_multi_period(config: ModelConfig, data_dir: str = ".") -> List[str]:
    """
    config.test_terms 의 여러 기간을 병렬로 실행합니다.
    원본 BMS_DL_CNN_1D_2014_17.py 의 threading 방식을 ThreadPoolExecutor로 대체.

    Args:
        config   : ModelConfig (test_terms 항목 사용)
        data_dir : CSV 파일 디렉토리

    Returns:
        각 기간별 결과 파일 경로 리스트
    """
    results = []

    def _run_one(term: Tuple[str, str]) -> str:
        cfg = ModelConfig(
            file_name=config.file_name,
            model_type=config.model_type,
            step_interval=config.step_interval,
            num_steps=config.num_steps,
            input_size=config.input_size,
            far_predict=config.far_predict,
            batch_size=config.batch_size,
            epochs=config.epochs,
            filters=config.filters,
            kernel_width=config.kernel_width,
            kernel_stride=config.kernel_stride,
            pool_size=config.pool_size,
            pool_stride=config.pool_stride,
            dropout_rate=config.dropout_rate,
            learning_rate=config.learning_rate,
            l2_reg=config.l2_reg,
            model_reset=True,
            test_start=term[0],
            test_end=term[1],
        )
        return run_single(cfg, data_dir=data_dir)

    # GPU 공유 시 직렬 실행이 안전; CPU만이면 병렬 가능
    use_parallel = len(tf.config.list_physical_devices("GPU")) == 0
    if use_parallel:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(_run_one, t): t for t in config.test_terms}
            for future in concurrent.futures.as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.error("기간 %s 실행 실패: %s", futures[future], e)
    else:
        for term in config.test_terms:
            try:
                results.append(_run_one(term))
            except Exception as e:
                logger.error("기간 %s 실행 실패: %s", term, e)

    return results


# ---------------------------------------------------------------------------
# CLI 인터페이스
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CNN 기반 주식 지수 중장기 예측 시스템 (TF2/Keras)"
    )
    parser.add_argument("--model",       default="1d_cls",
                        choices=["1d_cls", "2d_cls", "2d_reg"],
                        help="모델 타입 (기본: 1d_cls)")
    parser.add_argument("--file",        default="kospi",
                        help="raw_data_<file>.csv 의 <file> 부분 (기본: kospi)")
    parser.add_argument("--test_start",  default="2016-07-01",
                        help="테스트 시작일 (기본: 2016-07-01)")
    parser.add_argument("--test_end",    default="2017-06-30",
                        help="테스트 종료일 (기본: 2017-06-30)")
    parser.add_argument("--far_predict", type=int, default=260,
                        help="예측 기간 영업일 수 (기본: 260 ≈ 1년)")
    parser.add_argument("--step_interval", type=int, default=30,
                        help="시계열 스텝 간격 (기본: 30)")
    parser.add_argument("--num_steps",   type=int, default=100,
                        help="시계열 스텝 수 (기본: 100)")
    parser.add_argument("--input_size",  type=int, default=38,
                        help="입력 피처 수 (기본: 38)")
    parser.add_argument("--batch_size",  type=int, default=30,
                        help="배치 크기 (기본: 30)")
    parser.add_argument("--epochs",      type=int, default=50,
                        help="학습 에폭 수 (기본: 50)")
    parser.add_argument("--filters",     type=int, default=32,
                        help="CNN 필터 수 (기본: 32)")
    parser.add_argument("--lr",          type=float, default=1e-3,
                        help="학습률 (기본: 0.001)")
    parser.add_argument("--conversion",  default="diff",
                        choices=["diff", "rate", "norm"],
                        help="회귀 레이블 변환 방식 (기본: diff)")
    parser.add_argument("--no_reset",    action="store_true",
                        help="기존 체크포인트 유지 (이어서 학습)")
    parser.add_argument("--multi_period", action="store_true",
                        help="config.test_terms 의 여러 기간을 순차/병렬 실행")
    parser.add_argument("--data_dir",    default="data",
                        help="CSV 파일 디렉토리 (기본: data)")
    return parser.parse_args()


def main():
    args = parse_args()

    config = ModelConfig(
        model_type   = args.model,
        file_name    = args.file,
        test_start   = args.test_start,
        test_end     = args.test_end,
        far_predict  = args.far_predict,
        step_interval= args.step_interval,
        num_steps    = args.num_steps,
        input_size   = args.input_size,
        batch_size   = args.batch_size,
        epochs       = args.epochs,
        filters      = args.filters,
        learning_rate= args.lr,
        conversion   = args.conversion,
        model_reset  = not args.no_reset,
    )

    # GPU 설정 (메모리 증가 허용)
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logger.info("사용 가능 GPU: %d개", len(gpus))

    if args.multi_period:
        paths = run_multi_period(config, data_dir=args.data_dir)
        logger.info("멀티기간 실행 완료. 결과 파일:\n%s", "\n".join(paths))
    else:
        run_single(config, data_dir=args.data_dir)


if __name__ == "__main__":
    main()
