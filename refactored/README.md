# Demian-CNN v2  
**CNN 기반 주식 시장 지수 중장기 예측 시스템 (TF2/Keras)**

> 원본: Copyright 2018 Shinhan Financial Group / Bumghi Choi  
> 리팩토링: TensorFlow 2.x / Keras (2025)

---

## 개요

CNN(Convolutional Neural Network)으로 각국 주식 지수(KOSPI, S&P500, FTSE, Nikkei 등)의  
**중장기(1개월 ~ 1년) 상승/하락 확률** 또는 **실제 지수 변화량**을 예측하는 딥러닝 시스템입니다.

### 주요 변경사항 (v1 → v2)

| 항목 | 원본 (v1) | 리팩토링 (v2) |
|------|-----------|--------------|
| TensorFlow | 1.x (`tf.layers`, `tf.estimator`) | 2.x (Keras API) |
| 모델 정의 | `cnn_model_fn` 함수 | `keras.Model` 서브클래스 |
| 학습 루프 | `Estimator.train()` | `model.fit()` + 콜백 |
| 조기 종료 | 없음 | `EarlyStopping` 콜백 |
| LR 스케줄 | 없음 | `ReduceLROnPlateau` 콜백 |
| TensorBoard | 수동 `tf.summary` | Keras TensorBoard 콜백 |
| 멀티기간 병렬 | `threading.Thread` | `concurrent.futures` |
| 설정 관리 | 파일마다 별도 `Config` 클래스 | `dataclass` 기반 단일 `ModelConfig` |
| 데이터 로더 | `reader.py`, `reader_1D.py`, `reader_v2.py` (3개) | `data_reader.py` (1개 통합) |
| 진입점 | 파일별 `tf.app.run()` | `main.py` CLI 통합 |

---

## 파일 구조

```
Demian-CNN-v2/
├── main.py           # 메인 실행 스크립트 (CLI)
├── config.py         # ModelConfig 설정 클래스
├── data_reader.py    # 데이터 로드 / 전처리 / 시퀀스 생성
├── model_builder.py  # Keras 모델 팩토리 (1D_cls / 2D_cls / 2D_reg)
├── trainer.py        # 학습 / 평가 / 결과 저장
├── requirements.txt  # 의존 라이브러리
├── data/             # raw_data_<name>.csv 파일 위치
├── results/          # 예측 결과 CSV 저장
└── model_checkpoints/# 모델 가중치 저장
```

---

## 빠른 시작

### 1. 환경 설치

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 데이터 준비

`data/` 폴더에 아래 형식의 CSV 파일을 배치합니다:

```
data/raw_data_kospi.csv
data/raw_data_ftse.csv
...
```

**CSV 컬럼 형식:**

| 위치 | 컬럼명 | 설명 |
|------|--------|------|
| 0 | `date` | 날짜 (YYYY-MM-DD) |
| 1 ~ N | `feat_1` … `feat_N` | 기술적 지표 피처 (N = `input_size`) |
| N+1 | `close` | 종가 지수 |

### 3. 실행

```bash
# 1D CNN 분류 (1년 예측, KOSPI)
python main.py --model 1d_cls --file kospi --far_predict 260

# 2D CNN 분류 (3개월 예측, FTSE)
python main.py --model 2d_cls --file ftse --far_predict 65 \
               --step_interval 20 --num_steps 20

# 2D CNN 회귀 (Nikkei)
python main.py --model 2d_reg --file nikkei225 --conversion diff

# 멀티기간 테스트 (2014~2018)
python main.py --model 1d_cls --file kospi --multi_period
```

---

## 모델 타입

| `--model` | 설명 | 원본 파일 |
|-----------|------|----------|
| `1d_cls` | 1D CNN 이진 분류 (상승↑/하락↓ 확률) | `BMS_DL_CNN_1D.py` |
| `2d_cls` | 2D CNN 이진 분류 | `BMS_DL_CNN.py` |
| `2d_reg` | 2D CNN 회귀 (지수 변화량 예측) | `BMS_DL_CNN_v2.py` |

---

## 주요 하이퍼파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `--far_predict` | 260 | 예측 기간(영업일). 65≈3개월, 130≈6개월, 260≈1년 |
| `--step_interval` | 30 | 시계열 스텝 간격(일) |
| `--num_steps` | 100 | 입력 시퀀스 길이 |
| `--filters` | 32 | CNN 필터 수 |
| `--epochs` | 50 | 최대 학습 에폭 (EarlyStopping 적용) |
| `--batch_size` | 30 | 미니배치 크기 |
| `--lr` | 0.001 | Adam 학습률 |

---

## 결과 파일

학습 완료 후 `results/` 폴더에 CSV 파일이 생성됩니다:

```csv
date,pred_date,index_today,real,pred_class,prediction
2016-07-01,2017-06-30,2015.32,1,1,0.87
2016-07-04,2017-07-03,2018.45,0,0,0.21
...
# === 성능 지표 ===
# accuracy: 0.6234
# precision: 0.6100
# recall: 0.5900
# f1_score: 0.5998
# rmse: 0.4821
```

---

## TensorBoard

```bash
tensorboard --logdir model_checkpoints/
```

---

## 라이선스

Copyright 2018 Shinhan Financial Group / Bumghi Choi. All Rights Reserved.
