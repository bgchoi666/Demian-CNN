# CNN 기반 주식 시장 지수 예측 프로그램 문서

## 목차
1. [프로젝트 개요](#프로젝트-개요)
2. [시스템 아키텍처](#시스템-아키텍처)
3. [프로그램 구성](#프로그램-구성)
4. [데이터 처리](#데이터-처리)
5. [모델 상세](#모델-상세)
6. [사용 방법](#사용-방법)
7. [성능 평가](#성능-평가)

---

## 프로젝트 개요

### 목적
CNN(Convolutional Neural Networks)을 활용하여 각국의 주식 시장 지수 및 채권 금리를 예측하는 딥러닝 시스템입니다. 시계열 데이터를 이미지 처리 방식으로 학습하여 미래 시장 움직임을 예측합니다.

### 주요 특징
- **다양한 CNN 아키텍처**: 1D CNN, 2D CNN 지원
- **유연한 예측 기간**: 1일부터 1년까지 조정 가능
- **다중 시장 지원**: KOSPI, S&P500, FTSE, Nikkei 등
- **두 가지 예측 모드**: 
  - 분류(Classification): 상승/하락 확률 예측
  - 회귀(Regression): 실제 지수 값 예측

### 지원 시장
- KOSPI 200 선물
- S&P 500
- FTSE
- Nikkei 225
- 기타 주요 지수

---

## 시스템 아키텍처

### 전체 구조도

```
┌─────────────────────────────────────────────────────────────┐
│                      데이터 입력 계층                          │
│  (CSV 파일: 기술적 지표, 가격 데이터, 거래량 등)                │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                   데이터 전처리 계층                           │
│  - 정규화 (StandardScaler)                                   │
│  - 시계열 시퀀스 생성                                          │
│  - Train/Test 분할                                           │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    CNN 모델 계층                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   1D CNN     │  │   2D CNN     │  │  Hybrid CNN  │      │
│  │  (시계열)     │  │  (이미지형)   │  │  (혼합형)     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    출력 계층                                  │
│  - 분류: 상승(1) / 하락(0) 확률                               │
│  - 회귀: 예측 지수 값                                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 프로그램 구성

### 1. BMS_DL_CNN.py
**2D CNN 기반 확률 예측 시스템**

#### 특징
- 2D CNN을 1D CNN처럼 활용
- Width: 변수들, Height: 시계열 스텝
- Softmax를 통한 상승/하락 확률 산출

#### 핵심 설정
```python
class Config:
    step_interval = 20      # 스텝 간 간격
    num_steps = 20          # 시계열 스텝 수
    input_size = 38         # 입력 변수 개수
    far_predict = 65        # 예측 기간 (일)
    batch_size = 50         # 배치 크기
    iter_steps = 2000       # 학습 반복 횟수
    filters = 32            # CNN 필터 수
    kernel_height = 5       # 커널 높이
    kernel_width = 5        # 커널 너비
```

#### 네트워크 구조
```
Input Layer (20 x 38 x 1)
    ↓
Conv2D Layer 1 (32 filters, 5x5 kernel)
    ↓
MaxPooling2D (2x1 pool size)
    ↓
Conv2D Layer 2 (32 filters, 5x1 kernel)
    ↓
MaxPooling2D (2x1 pool size)
    ↓
Flatten
    ↓
Dropout (0.4)
    ↓
Dense Layer (50 units)
    ↓
Dropout (0.2)
    ↓
Output Layer (2 units - Softmax)
```

---

### 2. BMS_DL_CNN_1D.py
**1D CNN 기반 확률 예측 시스템 (유연한 레이어)**

#### 특징
- 순수 1D CNN 구조
- 레이어 수 자동 조정 (2~5개)
- 입력 크기에 따라 동적으로 네트워크 깊이 결정

#### 핵심 설정
```python
class Config:
    step_interval = 30      # 스텝 간 간격
    num_steps = 100         # 시계열 스텝 수
    input_size = 38         # 입력 변수 개수
    far_predict = 260       # 예측 기간 (1년)
    batch_size = 30         # 배치 크기
    filters = 32            # CNN 필터 수
    kernel_width = 5        # 커널 너비
    kernel_stride_width = 3 # 스트라이드
```

#### 네트워크 구조 (동적)
```
Input Layer (100 x 38 = 3800)
    ↓
Conv1D Layer 1 (32 filters, kernel_size=5, stride=3)
    ↓
MaxPooling1D (pool_size=2, stride=2)
    ↓
[조건부: flat_size > 3000]
Conv1D Layer 2-5 (반복)
    ↓
[조건부: flat_size > 3000/2000/1000]
Dense Layers (2000 → 1000 → 500 units)
    ↓
Dense Layer (50 units)
    ↓
Output Layer (2 units - Softmax)
```

#### 레이어 조건부 추가 로직
```python
if pool.shape[1] * pool.shape[2] > 3000:
    # Conv + Pool 레이어 추가
    
if flat_size > 3000:
    # Dense 2000 레이어 추가
    
if flat_size > 2000:
    # Dense 1000 레이어 추가
    
if flat_size > 1000:
    # Dense 500 레이어 추가
```

---

### 3. BMS_DL_CNN_1D_origin.py
**1D CNN 기반 확률 예측 시스템 (고정 레이어)**

#### 특징
- 고정된 2개 Conv 레이어
- 간단한 네트워크 구조
- 빠른 학습 속도

#### 네트워크 구조
```
Input Layer (num_steps × input_size)
    ↓
Conv1D Layer 1 (32 filters)
    ↓
MaxPooling1D
    ↓
Conv1D Layer 2 (32 filters)
    ↓
MaxPooling1D
    ↓
Flatten
    ↓
Dropout (0.4)
    ↓
Dense Layer (50 units)
    ↓
Dropout (0.2)
    ↓
Output Layer (2 units)
```

---

### 4. BMS_DL_CNN_1D_2014_17.py
**다중 기간 테스트 시스템**

#### 특징
- 여러 테스트 기간 동시 실행
- 멀티스레딩 지원
- 기간별 성능 비교 분석

#### 테스트 기간 설정
```python
test_terms = [
    ('2014-07-01', '2015-06-30'),
    ('2015-07-01', '2016-06-30'),
    ('2016-07-01', '2017-06-30'),
    ('2017-07-01', '2018-06-30')
]
```

#### 멀티스레딩 구조
```python
t = ['' for i in range(4)]
for i in range(4):
    test_start_end = list(config.test_terms[i])
    t[i] = threading.Thread(
        target=model_create_train_predict,
        args=(test_start_end[0], test_start_end[1])
    )
    t[i].start()
```

---

### 5. BMS_DL_CNN_v2.py
**회귀 기반 2D CNN 시스템**

#### 특징
- 상승/하락이 아닌 실제 지수 값 예측
- 테스트 세트가 데이터 중간에 위치 가능
- 다중 학습 데이터 지원

#### 출력 변환 방식
```python
# 'diff': 차이값
train_target.append(
    train_target_raw[i+far_predict] - train_target_raw[i]
)

# 'rate': 변화율(%)
train_target.append(
    (train_target_raw[i+far_predict] - train_target_raw[i]) 
    / train_target_raw[i] * 100
)

# 'norm': 정규화 값
train_target.append(
    (train_target_raw[i+far_predict] - 67) / 30
)
```

#### 네트워크 구조
```
Input Layer (20 x 39 x 1)
    ↓
Conv2D Layers (3개, 32 filters each)
    ↓
Flatten
    ↓
Dropout (0.4)
    ↓
Dense Layer 1 (500 units)
    ↓
Dropout (0.4)
    ↓
Dense Layer 2 (50 units)
    ↓
Dense Layer 3 (3 units)
    ↓
Output Layer (1 unit - Linear)
```

---

### 6. IMG_CNN.py
**이미지 분류 CNN 시스템**

#### 특징
- 이미지 파일 기반 학습
- 1D CNN으로 이미지 처리
- 흑백 이미지 지원

#### 네트워크 구조
```
Input Layer (25 x 25 = 625)
    ↓
Conv1D Layer 1 (16 filters, kernel=5)
    ↓
MaxPooling1D
    ↓
Conv1D Layer 2 (16 filters, kernel=5)
    ↓
MaxPooling1D
    ↓
Conv1D Layer 3 (16 filters, kernel=5)
    ↓
MaxPooling1D
    ↓
Flatten
    ↓
Dense Layer 1 (1024 units)
    ↓
Dense Layer 2 (100 units)
    ↓
Output Layer (2 units)
```

---

## 데이터 처리

### 데이터 리더 모듈

#### 1. reader.py (2D CNN용)
**시계열 데이터를 2D 형태로 변환**

```python
def producer(raw_data, input_size, output_size, far_predict, 
             interval, steps):
    """
    시계열 데이터를 CNN 입력 형태로 변환
    
    Args:
        raw_data: 원본 데이터
        input_size: 입력 변수 개수
        output_size: 출력 크기
        far_predict: 예측 기간
        interval: 스텝 간 간격
        steps: 시계열 스텝 수
    
    Returns:
        x: [batch_size, steps * input_size]
        y: [batch_size]
    """
    dataX, dataY = [], []
    size = len(raw_data) - interval * (steps - 1)
    
    for i in range(size):
        # i, i+interval, i+2*interval, ..., i+(steps-1)*interval
        input_list = list(range(i, i + steps * interval, interval))
        a = np.reshape(raw_data[input_list, :input_size], 
                      [steps * input_size])
        dataX.append(a)
        
        # 마지막 날짜의 타겟 값
        b = np.reshape(raw_data[i + interval * (steps - 1), input_size], [1])
        dataY.append(b)
    
    return np.array(dataX), np.array(dataY).reshape(-1)
```

#### 2. reader_1D.py (1D CNN용)
**시계열 데이터를 1D 형태로 변환**

```python
def producer(raw_data, input_size, output_size, far_predict, 
             interval, steps):
    """
    각 변수의 시계열을 한 줄로 연결
    
    변환 방식:
    [var1_t0, var2_t0, ..., varN_t0, 
     var1_t1, var2_t1, ..., varN_t1,
     ...
     var1_tM, var2_tM, ..., varN_tM]
    """
    dataX, dataY = [], []
    size = len(raw_data) - interval * (steps - 1)
    
    for i in range(size):
        input_list = list(range(i, i + steps * interval, interval))
        # 전치(Transpose)하여 변수별 시계열을 연결
        a = np.reshape(raw_data[input_list, :input_size].T, 
                      [steps * input_size])
        dataX.append(a)
        
        b = np.reshape(raw_data[i + interval * (steps - 1), input_size], [1])
        dataY.append(b)
    
    return np.array(dataX), np.array(dataY).reshape(-1)
```

### 데이터 정규화

#### StandardScaler 사용
```python
from sklearn.preprocessing import StandardScaler

def normalize(df):
    """
    Z-score 정규화
    normalized_value = (value - mean) / std
    """
    normal_proc = StandardScaler().fit(df)
    transformed_df = normal_proc.transform(df)
    return transformed_df

def denormalize(df):
    """
    정규화 역변환
    """
    normal_proc = StandardScaler().fit(df)
    inverse_trans_df = normal_proc.inverse_transform(df)
    return inverse_trans_df
```

### 데이터 분할

```python
# 테스트 시작/종료 인덱스 계산
test_start_index = len(raw_df[raw_df['date'] <= config.test_start]) - 1 - config.far_predict
test_end_index = len(raw_df[raw_df['date'] <= config.test_end]) - 1 - config.far_predict

# 학습 데이터 생성
train_data = df[0: test_start_index - config.far_predict, :config.input_size]

# 테스트 데이터 생성
test_data = df[test_start_index - config.step_interval * (config.num_steps - 1): 
               test_end_index + 1, :config.input_size]
```

### 타겟 레이블 생성

#### 분류 (상승/하락)
```python
train_target = []
for i in range(len(train_target_raw) - config.far_predict):
    if train_target_raw[i + config.far_predict] - train_target_raw[i] > 0:
        train_target.append(1)  # 상승
    else:
        train_target.append(0)  # 하락
```

#### 회귀 (실제 값)
```python
train_target = []
for i in range(len(train_target_raw) - config.far_predict):
    # 차이값
    if config.conversion == 'diff':
        train_target.append(
            train_target_raw[i + config.far_predict] - train_target_raw[i]
        )
    
    # 변화율
    if config.conversion == 'rate':
        train_target.append(
            (train_target_raw[i + config.far_predict] - train_target_raw[i]) 
            / train_target_raw[i] * 100
        )
    
    # 정규화
    if config.conversion == 'norm':
        train_target.append(
            (train_target_raw[i + config.far_predict] - 67) / 30
        )
```

---

## 모델 상세

### 손실 함수 및 최적화

#### 분류 모델
```python
# Sparse Softmax Cross Entropy
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=tf.cast(labels, tf.int32),
    logits=outputs,
    name='cross_entropy_per_example'
)
loss = tf.reduce_mean(cross_entropy, name='cross_entropy')

# Adam Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train_op = optimizer.minimize(
    loss=loss,
    global_step=tf.train.get_global_step()
)
```

#### 회귀 모델
```python
# Mean Squared Error
labels = tf.cast(tf.reshape(labels, [-1]), tf.float32)
loss = tf.losses.mean_squared_error(labels=labels, predictions=logits)

# Adam Optimizer (동일)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
```

### 정규화 기법

#### Dropout
```python
# CNN 레이어 후
dropout_cnn = tf.layers.dropout(
    inputs=pool_flat,
    rate=0.4,  # 40% 드롭
    training=mode == tf.estimator.ModeKeys.TRAIN
)

# Dense 레이어 후
dropout = tf.layers.dropout(
    inputs=dense,
    rate=0.2,  # 20% 드롭
    training=mode == tf.estimator.ModeKeys.TRAIN
)
```

#### L2 Regularization
```python
dense = tf.layers.dense(
    inputs=dropout,
    units=50,
    activation=tf.nn.relu,
    kernel_initializer=tf.contrib.layers.xavier_initializer(),
    kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001)
)
```

### 초기화 방법

#### Xavier Initialization
```python
kernel_initializer = tf.contrib.layers.xavier_initializer()
```
- 입력과 출력 뉴런 수를 고려한 가중치 초기화
- 그래디언트 소실/폭발 방지

---

## 사용 방법

### 1. 환경 설정

#### 필수 라이브러리
```bash
pip install tensorflow==1.x
pip install numpy
pip install pandas
pip install scikit-learn
pip install pillow  # 이미지 처리용
```

### 2. 데이터 준비

#### CSV 파일 형식
```csv
date,feature1,feature2,...,feature38,close_price
2018-01-01,0.5,0.3,...,0.8,2500.5
2018-01-02,0.6,0.4,...,0.7,2505.2
...
```

#### 필수 컬럼
- `date`: 날짜 (YYYY-MM-DD)
- `feature1~featureN`: 기술적 지표
- `close_price`: 종가 지수

### 3. 설정 변경

#### Config 클래스 수정
```python
class Config:
    # 데이터 설정
    file_name = "kospi"          # 파일명
    test_start = '2016-07-01'    # 테스트 시작일
    test_end = '2017-08-31'      # 테스트 종료일
    
    # 시계열 설정
    step_interval = 20           # 스텝 간격 (일)
    num_steps = 20               # 입력 스텝 수
    far_predict = 65             # 예측 기간 (일)
    
    # 모델 설정
    input_size = 38              # 입력 변수 개수
    batch_size = 50              # 배치 크기
    iter_steps = 2000            # 학습 반복
    filters = 32                 # 필터 수
    
    # 커널 설정
    kernel_width = 5
    kernel_height = 5
    kernel_stride_width = 1
    kernel_stride_height = 1
    
    # 풀링 설정
    pool_width = 2
    pool_height = 2
    pool_stride_width = 2
    pool_stride_height = 2
    
    # 기타
    model_reset = True           # 모델 초기화 여부
```

### 4. 실행

#### 기본 실행
```bash
python BMS_DL_CNN_1D.py
```

#### 다중 기간 테스트
```bash
python BMS_DL_CNN_1D_2014_17.py
```

### 5. 결과 확인

#### 결과 파일 위치
```
results/
├── CNN_1D_prob_kospi_65_20_20_k5.3_p2.2_50_2016-07-01_2024-02-16.csv
└── ...
```

#### 결과 파일 형식
```csv
date,pred_date,index_today,pred_class,real,prediction
2016-07-01,2016-09-04,2015.32,1,1,0.85
2016-07-04,2016-09-07,2018.45,0,0,0.23
...

accuracy, RMSE
0.6234, 0.4821
```

---

## 성능 평가

### 평가 지표

#### 1. 정확도 (Accuracy)
```python
accuracy = (TP + TN) / (TP + TN + FP + FN)
```
- TP: True Positive (상승 예측 & 실제 상승)
- TN: True Negative (하락 예측 & 실제 하락)
- FP: False Positive (상승 예측 & 실제 하락)
- FN: False Negative (하락 예측 & 실제 상승)

#### 2. RMSE (Root Mean Squared Error)
```python
rmse = sqrt(mean((predicted - actual)^2))
```
- 회귀 모델의 예측 오차 측정

#### 3. Precision & Recall
```python
precision = TP / (TP + FP)  # 예측의 정확성
recall = TP / (TP + FN)     # 실제 양성의 검출률
f1_score = 2 / ((1/precision) + (1/recall))
```

### 성능 계산 함수

```python
def calculate_recall_precision(label, prediction):
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    
    for i in range(len(label)):
        if prediction[i] == 1:
            if label[i] == 1:
                true_positives += 1
            else:
                false_positives += 1
        else:
            if label[i] == 0:
                true_negatives += 1
            else:
                false_negatives += 1
    
    accuracy = (true_positives + true_negatives) / \
               (true_positives + true_negatives + 
                false_positives + false_negatives)
    
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 / ((1 / precision) + (1 / recall))
    
    return accuracy, precision, recall, f1_score
```

### 결과 저장

```python
def save_results(predictions, config, target, index_today, 
                today, date, eval_results):
    pred_values = []
    target_values = []
    class_values = []
    
    for i, p in enumerate(predictions):
        target_values.append(target[i])
        k = p["probabilities"]
        pred_values.append(k[1])  # 상승 확률
        class_values.append(p["classes"])
    
    accuracy, _, _, _ = calculate_recall_precision(
        target_values, class_values
    )
    
    comp_results = {
        "date": today,
        "pred_date": date,
        "index_today": index_today,
        "pred_class": class_values,
        "real": target_values,
        "prediction": pred_values
    }
    
    result_file = f"results/CNN_1D_prob_{config.file_name}_" \
                  f"{config.far_predict}_{config.step_interval}_" \
                  f"{config.num_steps}_k{config.kernel_width}." \
                  f"{config.kernel_stride_width}_p{config.pool_width}." \
                  f"{config.pool_stride_width}_{config.batch_size}_" \
                  f"{config.test_start}_{timestamp}.csv"
    
    pd.DataFrame.from_dict(comp_results).to_csv(result_file, index=False)
    
    with open(result_file, 'a') as r:
        r.write(f"accuracy, RMSE\n{accuracy}, {eval_results['rmse']}")
```

---

## 고급 기능

### 1. TensorBoard 통합

```python
# TensorBoard용 변수 로깅
tv = tf.trainable_variables()
tf.summary.scalar("first_filter_weight", tv[0][0, 0, 0])
tf.summary.scalar("second_filter_weight", outputs[2, 1])
```

실행:
```bash
tensorboard --logdir=model_dir/
```

### 2. 체크포인트 관리

```python
my_model_config = tf.estimator.RunConfig(
    keep_checkpoint_max=1  # 최대 1개 체크포인트 유지
)

cnn = tf.estimator.Estimator(
    model_fn=cnn_model_fn,
    model_dir=model_dir,
    config=my_model_config
)
```

### 3. 로깅 훅

```python
tensors_to_log = {"classes": "classes"}
logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log,
    every_n_iter=50  # 50 반복마다 로깅
)

cnn.train(
    input_fn=train_input_fn,
    hooks=[logging_hook],
    steps=config.iter_steps
)
```

---

## 주의사항 및 팁

### 1. 과적합 방지
- **Dropout 사용**: 0.2~0.4 비율 권장
- **L2 Regularization**: 0.001 계수 권장
- **조기 종료**: Validation loss 모니터링
- **데이터 증강**: 더 많은 학습 데이터 확보

### 2. 하이퍼파라미터 튜닝
```python
# 실험해볼 조합
configurations = [
    # (far_predict, step_interval, num_steps, filters)
    (65, 20, 20, 32),
    (65, 30, 30, 64),
    (130, 20, 50, 32),
    (260, 30, 100, 64),
]
```

### 3. 메모리 관리
```python
# 배치 크기 조정
if memory_limited:
    config.batch_size = 10  # 작은 배치
else:
    config.batch_size = 100  # 큰 배치
```

### 4. GPU 활용
```python
# GPU 메모리 증가 허용
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_options)
```

---

## 트러블슈팅

### 문제 1: 모델이 수렴하지 않음
**해결책:**
- Learning rate 감소 (0.001 → 0.0001)
- 배치 크기 증가
- 더 많은 학습 반복
- 데이터 정규화 확인

### 문제 2: 과적합 발생
**해결책:**
- Dropout 비율 증가
- L2 regularization 강화
- 학습 데이터 증가
- 모델 복잡도 감소

### 문제 3: 메모리 부족
**해결책:**
- 배치 크기 감소
- num_steps 감소
- 필터 수 감소
- GPU 메모리 설정 조정

### 문제 4: 학습 속도 느림
**해결책:**
- GPU 사용 확인
- 배치 크기 증가
- 데이터 파이프라인 최적화
- 불필요한 레이어 제거

---

## 확장 가능성

### 1. 다른 모델 통합
- LSTM + CNN 하이브리드
- Attention 메커니즘 추가
- Transformer 기반 모델

### 2. 추가 기능
- 실시간 예측 API
- 자동 매매 시스템 연동
- 앙상블 모델
- 멀티 태스크 학습

### 3. 성능 개선
- 배치 정규화 (Batch Normalization)
- 잔차 연결 (Residual Connections)
- 데이터 증강 기법
- 교차 검증

---

## 라이선스 및 연락처

### 라이선스
Copyright 2018 Shinhan Financial Group / Bumghi Choi. All Rights Reserved.

### 참고 문헌
- TensorFlow 공식 문서
- "Time Series Forecasting using Deep Learning"
- "CNN for Financial Time Series Analysis"

### 기술 지원
문제가 발생하거나 질문이 있는 경우 GitHub Issues를 통해 문의해주세요.

---

## 버전 히스토리

### v1.0 (2018)
- 초기 릴리스
- 2D CNN, 1D CNN 지원
- 분류 및 회귀 모델

### v2.0 (개선 사항)
- 동적 레이어 조정
- 멀티스레딩 지원
- 성능 최적화

---

**마지막 업데이트**: 2024년 2월
**문서 버전**: 1.0
