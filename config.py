"""
config.py - CNN 주식 지수 예측 시스템 설정 모듈

Copyright 2018 Shinhan Financial Group / Bumghi Choi. All Rights Reserved.
Refactored for TensorFlow 2.x / Keras by Claude (2025)
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class ModelConfig:
    """
    CNN 모델 하이퍼파라미터 및 데이터 설정.

    Attributes:
        file_name       : CSV 파일명 접두사. 예) 'kospi' → raw_data_kospi.csv
        test_start      : 테스트 기간 시작일 (YYYY-MM-DD)
        test_end        : 테스트 기간 종료일 (YYYY-MM-DD)
        step_interval   : 시계열 스텝 간격 (영업일 기준)
        num_steps       : CNN에 입력할 시계열 스텝 수
        input_size      : 입력 피처(기술적 지표) 컬럼 수
        far_predict     : 예측 대상 기간 (영업일). 예) 65 ≈ 3개월, 260 ≈ 1년
        batch_size      : 학습 미니배치 크기
        epochs          : 학습 에폭 수 (iter_steps 대체)
        filters         : CNN Conv 레이어 필터 수
        kernel_width    : 1D Conv 커널 너비
        kernel_stride   : 1D Conv 스트라이드
        pool_size       : MaxPool 풀링 크기
        pool_stride     : MaxPool 스트라이드
        dropout_rate    : Dropout 비율 (0.0 ~ 1.0)
        learning_rate   : Adam 옵티마이저 학습률
        l2_reg          : L2 정규화 계수
        model_type      : '1d_cls'(분류) | '2d_cls'(분류) | '2d_reg'(회귀)
        model_reset     : True이면 기존 체크포인트 삭제 후 재학습
        conversion      : 회귀 모드 타겟 변환 방식 ('diff'|'rate'|'norm')
        test_terms      : 멀티기간 테스트용 (시작일, 종료일) 리스트
    """
    # 데이터 설정
    file_name: str = "kospi"
    test_start: str = "2016-07-01"
    test_end: str = "2017-06-30"

    # 시계열 설정
    step_interval: int = 30
    num_steps: int = 100
    input_size: int = 38

    # 예측 기간
    far_predict: int = 260  # 1년 (260 영업일)

    # 학습 설정
    batch_size: int = 30
    epochs: int = 50          # 원본 iter_steps(2000) 대비 epoch 기반으로 변경
    learning_rate: float = 1e-3

    # 아키텍처
    filters: int = 32
    kernel_width: int = 5
    kernel_stride: int = 3
    pool_size: int = 2
    pool_stride: int = 2
    dropout_rate: float = 0.4
    l2_reg: float = 1e-3

    # 모델 종류
    model_type: str = "1d_cls"   # '1d_cls' | '2d_cls' | '2d_reg'

    # 체크포인트
    model_reset: bool = True

    # 회귀 전용
    conversion: str = "diff"     # 'diff' | 'rate' | 'norm'

    # 멀티기간 테스트 전용
    test_terms: List[Tuple[str, str]] = field(default_factory=lambda: [
        ("2014-07-01", "2015-06-30"),
        ("2015-07-01", "2016-06-30"),
        ("2016-07-01", "2017-06-30"),
        ("2017-07-01", "2018-06-30"),
    ])

    @property
    def model_dir(self) -> str:
        """체크포인트 저장 경로"""
        return (
            f"model_checkpoints/{self.model_type}_{self.file_name}"
            f"_{self.far_predict}_{self.step_interval}_{self.num_steps}"
            f"_f{self.filters}_k{self.kernel_width}.{self.kernel_stride}"
            f"_p{self.pool_size}.{self.pool_stride}"
            f"_b{self.batch_size}_{self.test_start}"
        )

    @property
    def result_prefix(self) -> str:
        """결과 CSV 파일명 접두사"""
        return (
            f"results/{self.model_type}_{self.file_name}"
            f"_{self.far_predict}_{self.step_interval}_{self.num_steps}"
            f"_k{self.kernel_width}.{self.kernel_stride}"
            f"_p{self.pool_size}.{self.pool_stride}"
            f"_{self.batch_size}_{self.test_start}"
        )


# ---------------------------------------------------------------------------
# 빠른 실행을 위한 사전 정의 설정들
# ---------------------------------------------------------------------------

CONFIG_KOSPI_1YEAR = ModelConfig(
    file_name="kospi",
    test_start="2016-07-01",
    test_end="2017-06-30",
    step_interval=30,
    num_steps=100,
    input_size=38,
    far_predict=260,
    model_type="1d_cls",
)

CONFIG_FTSE_3MONTH = ModelConfig(
    file_name="ftse",
    test_start="2016-07-01",
    test_end="2017-06-30",
    step_interval=20,
    num_steps=20,
    input_size=38,
    far_predict=65,
    model_type="2d_cls",
)

CONFIG_NIKKEI_REGRESSION = ModelConfig(
    file_name="nikkei225",
    test_start="2017-01-01",
    test_end="2018-10-01",
    step_interval=20,
    num_steps=20,
    input_size=39,
    far_predict=65,
    model_type="2d_reg",
    conversion="diff",
)
