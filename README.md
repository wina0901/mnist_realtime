
# MNIST Real-time Feedback Learning

PyTorch 기반 실시간 손글씨 숫자 인식 + 사용자 피드백 기반 파인튜닝 프로젝트입니다.

---

## 📌 프로젝트 개요

이 프로젝트는 다음 흐름을 구현합니다:

1. **공식 MNIST 데이터셋으로 Baseline 모델 학습**
2. 웹 캔버스에서 사용자가 숫자를 직접 드로잉
3. 모델이 실시간으로 0~9 확률 출력
4. 사용자가 ✅ 맞음 / ❌ 틀림 피드백 제공
5. 틀린 샘플을 커스텀 데이터로 저장
6. 일정량 데이터 축적 후 Fine-tuning
7. Baseline vs Finetuned 성능 비교

---

## 🧠 핵심 개념

- **Domain Gap 문제 해결**
  - MNIST는 정제된 데이터
  - 실제 웹 드로잉은 선 두께, 위치, 안티앨리어싱 등 분포 차이 존재
  - 사용자 피드백을 통해 실제 입력 분포에 적응

- **Human-in-the-loop Learning**
  - 사용자가 직접 학습 데이터 생성
  - 피드백을 데이터로 전환

---

## 📂 프로젝트 구조

```
mnist-realtime-feedback/
├─ README.md
├─ requirements.txt
├─ python/
│  ├─ app.py
│  ├─ train_mnist.py
│  ├─ model.py
│  ├─ datasets.py
│  ├─ preprocess.py
│  └─ feedback_log.py
├─ python/weights/
│  ├─ baseline.pt
│  └─ finetuned.pt
└─ data/
   └─ custom/
      ├─ 0/ ... 9/
      └─ feedback.jsonl
```

---

## 🚀 실행 방법

### 1️⃣ 가상환경 및 패키지 설치

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

---

### 2️⃣ Baseline 학습 (MNIST)

```bash
python python/train_mnist.py --mode baseline --epochs 5 --out python/weights/baseline.pt
```

---

### 3️⃣ 실시간 데모 실행

```bash
python python/app.py
```

브라우저가 열리면 숫자를 직접 그려서 테스트하세요.

---

### 4️⃣ Fine-tuning (커스텀 데이터 활용)

```bash
python python/train_mnist.py   --mode finetune   --ckpt python/weights/baseline.pt   --custom_dir data/custom   --epochs 3   --out python/weights/finetuned.pt
```

---

## 📊 기대 효과

- Baseline: MNIST 테스트셋 정확도 높음
- Finetuned: 실제 웹 드로잉 입력에서 체감 정확도 개선

---

## 🔬 기술 스택

- Python
- PyTorch
- Gradio
- NumPy
- PIL

---

## 📎 확장 아이디어

- 혼동행렬 시각화
- Baseline vs Finetuned 비교 리포트 자동 생성
- ONNX export 및 웹 배포
- 사용자별 개인화 모델 저장

---

## 📘 License

MIT License
