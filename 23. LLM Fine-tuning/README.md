- 구글의 Colab은 LLM을 튜닝하기에는 현실적으로 리소스 문제가 있습니다.
- [runpod](https://www.runpod.io/console/gpu-cloud) 서비스를 사용하기를 권장합니다.

```
!wget https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/23.%20LLM%20Fine-tuning/sft-trainer.py
pip install -r requirements.txt
```

## SFT-LoRA
### 1. 학습 스크립트
```
!torchrun --nproc_per_node=4 --master_port=1234 sft-trainer.py \
    --base_model beomi/llama-2-ko-7b \
    --data-path daekeun-ml/naver-news-summarization-ko \
    --output_dir ./title-generator \
    --batch_size 8 \
    --micro_batch_size 1 \
    --num_epochs 3 \
    --learning_rate 1e-5 \
    --cutoff_len 2048 \
    --val_set_size 0 \
    --lora_r 16 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[gate_proj, down_proj, up_proj]' \
    --train_on_inputs False \
    --add_eos_token True \
    --group_by_length False \
    --lr_scheduler 'linear' \
    --warmup_steps 100
```

### 2. 알파카 템플릿
- sft-trainer.py안에 있는 `generate_and_tokenize_prompt` 함수를 수정하여 여러분만의 프롬프트 템플릿을 만들어야 합니다.

#### 2-1. 템플릿 예시
- 예를 들어서 알파카 데이터셋을 사용한다면 아래와 같은 템플릿이 가능합니다.
- 템플릿
```
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}
```

#### 2-2. 입력과 출력 완성
- 허깅페이스에 있는 [Bingsu/ko_alpaca_data](https://huggingface.co/datasets/Bingsu/ko_alpaca_data) 데이터셋을 보면 데이터가 instruction, input, output 열이 존재하는 것을 볼 수 있습니다.
- 위 템플릿에 따라서 만들어진 데이터는 다음과 같습니다.
- 모델의 입력
```
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
이 문장에 철자와 문법 오류가 있는지 평가하세요.

### Input:
그는 식사를 마치고 식당을 나섰습니다.

### Response:
```
- 모델의 출력(학습 대상)
```
그 문장에는 철자나 문법에 대한 오류가 없습니다.
```

### 3. 커스텀 템플릿
- 허깅페이스 데이터셋 중에는 뉴스의 제목, 본문, 요약문이 존재하는 [daekeun-ml/naver-news-summarization-ko](https://huggingface.co/datasets/daekeun-ml/naver-news-summarization-ko) 데이터셋이 있습니다.
- 저는 커스텀 템플릿을 만들어 본문이 주어지면 제목을 만드는 모델을 만들고자 합니다.
- 데이터셋을 보면 document, title 열이 존재하는 것을 볼 수 있습니다.

#### 3-1. 본문으로부터 제목을 생성하는 템플릿 구성
- 템플릿을 만들 때 너무 고민하지 마세요.
- 가이드1: LLM에게 당신은 ~한 역할을 하는 ~입니다.라고 하는 것은 가장 널리 알려진 시스템 프롬프트입니다.
- 가이드2: 그 후 각 입력이 들어갈 자리와 모델이 작성할 자리를 적절하게 배치하세요.
- 최종 템플릿은 다음과 같습니다. 모델은 본문을 주면 제목을 생성할 것입니다.
```
당신은 주어진 본문으로부터 적절한 제목을 생성하는 제목 생성기입니다. 본문이 주어지면 제목을 만드세요.

### 본문:
{document}

### 제목:
{title}
```

### 3-2. 입력과 출력 완성
- 허깅페이스에 있는 [daekeun-ml/naver-news-summarization-ko](https://huggingface.co/datasets/daekeun-ml/naver-news-summarization-ko) 
- 위 템플릿에 따라서 만들어진 데이터는 다음과 같습니다.

- 모델의 입력
```
당신은 주어진 본문으로부터 적절한 제목을 생성하는 제목 생성기입니다. 본문이 주어지면 제목을 만드세요.

### 본문:
앵커 정부가 올해 하반기 우리 경제의 버팀목인 수출 확대를 위해 총력을 기울이기로 했습니다. 특히 수출 중소기업의 물류난 해소를 위해 무역금융 규모를 40조 원 이상 확대하고 물류비 지원과 임시선박 투입 등을 추진하기로 했습니다. 류환홍 기자가 보도합니다. 기자 수출은 최고의 실적을 보였지만 수입액이 급증하면서 올해 상반기 우리나라 무역수지는 역대 최악인 103억 달러 적자를 기록했습니다. 정부가 수출확대에 총력을 기울이기로 한 것은 원자재 가격 상승 등 대외 리스크가 가중되는 상황에서 수출 증가세 지속이야말로 한국경제의 회복을 위한 열쇠라고 본 것입니다. 추경호 경제부총리 겸 기획재정부 장관 정부는 우리 경제의 성장엔진인 수출이 높은 증가세를 지속할 수 있도록 총력을 다하겠습니다. 우선 물류 부담 증가 원자재 가격 상승 등 가중되고 있는 대외 리스크에 대해 적극 대응하겠습니다. 특히 중소기업과 중견기업 수출 지원을 위해 무역금융 규모를 연초 목표보다 40조 원 늘린 301조 원까지 확대하고 물류비 부담을 줄이기 위한 대책도 마련했습니다. 이창양 산업통상자원부 장관 국제 해상운임이 안정될 때까지 월 4척 이상의 임시선박을 지속 투입하는 한편 중소기업 전용 선복 적재 용량 도 현재보다 주당 50TEU 늘려 공급하겠습니다. 하반기에 우리 기업들의 수출 기회를 늘리기 위해 2 500여 개 수출기업을 대상으로 해외 전시회 참가를 지원하는 등 마케팅 지원도 벌이기로 했습니다. 정부는 또 이달 중으로 반도체를 비롯한 첨단 산업 육성 전략을 마련해 수출 증가세를 뒷받침하고 에너지 소비를 줄이기 위한 효율화 방안을 마련해 무역수지 개선에 나서기로 했습니다. YTN 류환홍입니다.

### 제목:
```

- 모델의 출력(학습 대상)
```
추경호 중기 수출지원 총력 무역금융 40조 확대
```

## DPO
