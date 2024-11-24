# HuggingFace와 TRL을 사용한 LLaVa-1.5-7B 미세 조정

이 프로젝트는 **HuggingFace**의 **Transformers**, **PEFT**, **TRL** 라이브러리를 사용하여 **LLaVa-1.5-7B** 모델을 이미지 기반 질의응답 작업에 대해 미세 조정하는 과정을 보여줍니다. 목표는 시각-언어 모델(VLM)의 성능을 개선하여 사용자 맞춤 이미지 데이터셋을 처리하고 관련 있는 상세한 텍스트 응답을 생성하는 것입니다.

## 개요

이 저장소에는 사용자 맞춤 데이터셋을 이용하여 시각-언어 모델을 미세 조정하는 스크립트가 포함되어 있습니다. 미세 조정 과정은 **LoRA (Low-Rank Adaptation)**와 **4-bit 양자화**를 사용하여 자원 요구 사항을 줄여 일반 하드웨어에서도 대형 모델을 학습할 수 있게 합니다.

주요 단계는 다음과 같습니다:
- 4-bit 양자화를 사용하여 LLaVa-1.5-7B 모델 로드.
- Q&A 대화를 처리하기 위한 사용자 지정 채팅 템플릿 사용.
- `DataCollator` 클래스를 사용하여 이미지 및 Q&A 데이터셋 준비.
- **SFTTrainer**를 사용하여 하이퍼파라미터로 모델 학습.
- **Tensorboard**로 학습 진행 상황 모니터링.

## 요구 사항

- **Google Colab** (코드를 클라우드 환경에서 실행하는 것을 권장)
- **Python 3.7+**
- **PyTorch**
- **Transformers 4.39+**
- **TRL 0.8.3+**
- **PEFT**
- **BitsAndBytes**
- **Google Drive** (데이터 저장용)

필수 라이브러리 설치:
```bash
!pip install -U "transformers>=4.39.0"
!pip install peft bitsandbytes
!pip install -U "trl>=0.8.3"
```

## 데이터 준비

훈련 과정에서 세 가지 유형의 데이터가 사용됩니다:

1. **이미지 데이터**: 쇼핑몰 CCTV에서 캡처한 이미지로, `frames` 폴더에 저장되어 있습니다.
2. **텍스트 데이터 (`messages.txt`)**: 사용자와 AI 어시스턴트 간의 예시 대화가 포함된 파일로, Q&A 형식을 정의합니다.
3. **CSV 메타데이터 (`labels.csv`)**: 각 이미지에 등장하는 사람 수와 같은 메타데이터가 포함되어 있으며, 텍스트 데이터를 보강하는 데 사용됩니다.

이 데이터는 **`LLavaDataCollator`** 클래스를 사용해 학습 파이프라인에 로드되며, 이미지와 텍스트 데이터를 처리해 모델 미세 조정에 적합한 형태로 만듭니다.

## 학습 과정

### 모델 로드
모델은 Hugging Face의 허브에서 **4-bit 양자화** 설정으로 로드됩니다. 이를 통해 메모리 사용량을 최적화합니다:
```python
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    torch_dtype=torch.float16
)
```

### LoRA를 활용한 미세 조정
학습 스크립트는 **LoRA (Low-Rank Adaptation)**을 사용하여 모델의 특정 부분만 선택적으로 미세 조정함으로써 학습 효율성을 높입니다.

### 하이퍼파라미터
학습에 사용된 주요 하이퍼파라미터는 다음과 같습니다:
- **학습률**: `1.4e-5`
- **배치 크기**: `8`
- **에폭 수**: `1`
- **Gradient Accumulation Steps**: `1`

학습 인자는 다음과 같이 지정됩니다:
```python
training_args = TrainingArguments(
    output_dir="llava-1.5-7b-hf-ft-mix-vsft",
    report_to="tensorboard",
    learning_rate=1.4e-5,
    per_device_train_batch_size=8,
    num_train_epochs=1,
    gradient_checkpointing=True,
    fp16=True,
)
```

### 학습 및 모니터링
**SFTTrainer**를 사용하여 학습을 수행합니다. 학습 진행 상황은 **Tensorboard**를 통해 손실 및 성능 지표를 시각화하며 모니터링할 수 있습니다:
```python
%load_ext tensorboard
%tensorboard --logdir /content/llava-1.5-7b-hf-ft-mix-vsft
```

### 결과
- 모델의 학습 손실 값이 시간이 지남에 따라 감소하였으며, 이는 성공적인 수렴을 나타냅니다.
- 미세 조정된 모델은 CCTV 영상에서 사람 수를 추정하는 등의 이미지 기반 질문에 대해 문맥에 맞는 응답을 제공하는 능력을 보였습니다.

## 사용 방법
1. 이 저장소를 클론합니다:
   ```bash
   git clone https://github.com/yourusername/fine-tune-llava.git
   ```
2. 필요한 패키지를 설치합니다.
3. **Google Colab**에서 `fine_tune_VLM_LlaVa.ipynb` 파일을 엽니다.
4. 데이터를 액세스하기 위해 Google Drive를 마운트합니다.
5. 노트북의 지침에 따라 학습 과정을 진행합니다.

## 결론
이 프로젝트는 **LLaVa-1.5-7B**와 같은 대형 시각-언어 모델을 **LoRA**와 **4-bit 양자화**를 사용하여 사용자 맞춤 데이터셋에 효율적으로 미세 조정하는 방법을 보여줍니다. 이 접근법은 제한된 하드웨어 리소스에서도 강력한 모델을 실사용 환경에 맞게 조정할 수 있게 합니다.

## 향후 작업
- 더 다양한 유형의 이미지와 질문을 포함하도록 학습 데이터셋을 확장합니다.
- 추론 시 지연 시간을 줄이기 위해 모델을 추가 최적화합니다.
- **효율적인 미세 조정 방법**을 실험하여 성능을 개선합니다.

## 기여 방법
프로젝트에 기여하고 싶다면 다음 절차를 따라주세요:
1. 이 저장소를 포크합니다.
2. 새로운 기능이나 버그 수정을 위한 브랜치를 만듭니다:
   ```bash
   git checkout -b feature/새로운-기능-이름
   ```
3. 변경 사항을 커밋합니다:
   ```bash
   git commit -m "새로운 기능 추가 설명"
   ```
4. 브랜치에 푸시합니다:
   ```bash
   git push origin feature/새로운-기능-이름
   ```
5. 풀 리퀘스트를 만듭니다.

기여는 언제나 환영입니다! 새로운 기능 제안이나 버그 리포트를 통해도 기여할 수 있습니다.

## 라이선스
이 프로젝트는 MIT 라이선스 하에 라이선스가 부여되었습니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 감사의 말씀
- Hugging Face 팀의 오픈 소스 모델과 도구들에 감사드립니다.
- 원본 **Colab** 설정 스크립트를 제공해 주신 **@mrm8488**님께 감사드립니다.

기여를 원하시면 풀 리퀘스트 제출 또는 이슈 보고를 통해 참여해 주세요!

