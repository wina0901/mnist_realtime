# MNIST Realtime — 손글씨 숫자 실시간 인식

> 브라우저 캔버스에 숫자를 그리면 필기 도중 실시간으로 예측 결과와 확률 분포를 보여주는 딥러닝 데모 시스템

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-latest-green)

---

## 프로젝트 목적

단순히 MNIST 분류 모델을 학습시키는 데 그치지 않고, 학습된 모델을 실제 서비스 형태로 배포하는 전 과정을 직접 구현하는 것을 목표로 했습니다.

모델 정확도보다 사용자 경험 측면의 실시간성과 백엔드 서빙 아키텍처 설계에 집중했습니다. 

구체적으로는 아래 두 가지 문제를 해결하는 데 초점을 맞췄습니다.

- 손으로 글씨를 쓰는 도중에도 끊임없이 추론 결과가 갱신될 것
- FastAPI의 async event loop를 블로킹하지 않으면서 PyTorch 동기 추론을 서빙할 것

---

## 기술 스택

| 영역 | 기술 | 선택 이유 |
|---|---|---|
| 모델 학습 | PyTorch 2.x | 유연한 커스텀 모델 구현, CUDA 자동 감지 |
| 모델 서빙 | FastAPI + Uvicorn | async 지원, 자동 API 문서, 빠른 프로토타이핑 |
| 비동기 처리 | run_in_threadpool | PyTorch 동기 추론을 threadpool에서 실행해 event loop 블로킹 방지 |
| 이미지 처리 | Pillow + NumPy | 전처리 파이프라인 구현 (알파채널·흰 배경 이중 대응) |
| 프론트엔드 | Vanilla JS + Canvas API | 외부 프레임워크 없이 드로잉·debounce·throttle 직접 구현 |
| 동시성 제어 | threading.Lock (double-checked locking) | 멀티스레드 환경에서 모델 중복 로딩 방지 |

---

## 아키텍처

```
브라우저 (Canvas)
    │  PNG 이미지  (150ms throttle + mouseup debounce)
    ▼
FastAPI (server.py)
    │  run_in_threadpool → 동기 추론을 별도 스레드에서 실행
    ▼
simple_service.py
    │
    ├── core/preprocess.py   # 이미지 → 28×28 텐서
    └── core/model_runner.py # SimpleCNN 추론 (lazy load + thread-safe)
    │
    ▼
JSON 응답 { pred, conf, probs }
    │
    ▼
브라우저 — 예측 숫자 + 확률 분포 바 실시간 갱신
```

---

## 핵심 구현 포인트

### 1. 실시간 추론 전략 — throttle + debounce 이중 구조

단순 debounce만 사용하면 손을 멈추기 전까지 추론이 시작되지 않아 실시간처럼 느껴지지 않습니다. 이를 해결하기 위해 두 가지 타이머를 병행했습니다.

- throttle (150ms) — 그리는 도중 최소 150ms마다 추론 발동
- debounce (0ms) — 손을 떼는 순간 즉시 추론 발동
- 추론이 진행 중일 때 새 요청이 오면 `pendingInfer` 플래그로 완료 후 즉시 재추론

### 2. 전처리 파이프라인 — 학습 데이터 분포 재현

실제 사용자 입력(Canvas PNG)과 MNIST 학습 데이터 사이의 도메인 갭을 줄이기 위해 LeCun et al. 방식을 따랐습니다.

```
입력 이미지
  └─ 알파채널 유무 감지
       ├─ 투명 배경 PNG  → 알파값으로 필기 픽셀 추출
       └─ 흰 배경 이미지 → 반전 밝기로 필기 픽셀 추출 (JPEG 대응)
  └─ Bounding box crop
  └─ 정사각형 패딩 (비율 유지)
  └─ 20×20 리사이즈 → 28×28 캔버스 중앙 배치
  └─ 정규화 [0, 1]
```

### 3. thread-safe 모델 로딩 — double-checked locking

FastAPI는 멀티스레드로 요청을 처리하므로, 초기 요청이 동시에 들어올 경우 모델이 두 번 로딩될 수 있습니다. `threading.Lock`과 double-checked locking 패턴으로 이를 방지했습니다.

```python
def get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:   # lock 획득 후 재확인
                _model = load()
    return _model
```

### 4. 모델 구조 — SimpleCNN

```
입력 (N, 1, 28, 28)
  └─ Conv2d(1→16, k=3, p=1) + ReLU + MaxPool2d → (N, 16, 14, 14)
  └─ Conv2d(16→32, k=3, p=1) + ReLU + MaxPool2d → (N, 32, 7, 7)
  └─ Flatten → Linear(1568→128) + ReLU
  └─ Linear(128→10)
출력: logits (N, 10)
```

MNIST 테스트셋 기준 정확도 **약 99%** (8 epoch, Adam, lr=1e-3)

---

## 프로젝트 구조

```
mnist_realtime/
├── core/
│   ├── preprocess.py       # 이미지 전처리 파이프라인
│   └── model_runner.py     # 추론 실행 (lazy load, thread-safe)
├── services/
│   └── simple_service.py   # 추론 서비스 진입점
├── demo/
│   └── index.html          # 웹 데모 UI (Vanilla JS)
├── weights/
│   └── baseline.pt         # 학습된 가중치 (train_mnist.py 실행 후 생성)
├── model.py                # SimpleCNN / SimpleMLP 정의
├── server.py               # FastAPI 엔드포인트
├── app.py                  # API 클라이언트 헬퍼
└── train_mnist.py          # 학습 스크립트
```

---

## 실행 방법

### 1. 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. 모델 학습 (최초 1회)

```bash
python train_mnist.py
# 완료 시 weights/baseline.pt 생성 (CPU 기준 약 3~5분)
```

옵션 조정:

```bash
python train_mnist.py --epochs 10 --lr 5e-4 --batch_size 256
```

### 3. 서버 실행

```bash
uvicorn server:app --reload
```

### 4. 데모 UI 실행

서버가 실행된 상태에서 `demo/index.html`을 브라우저로 열면 됩니다.

### API 직접 테스트

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@your_image.png"
```

응답 예시:

```json
{
  "pred": 3,
  "conf": 0.987,
  "probs": [0.001, 0.002, 0.003, 0.987, 0.001, 0.001, 0.001, 0.001, 0.001, 0.002]
}
```

---

## 학습 환경

- Python 3.10+
- PyTorch 2.x (CPU / CUDA 자동 감지)
- MNIST 데이터셋 자동 다운로드 (`train_mnist.py` 실행 시)