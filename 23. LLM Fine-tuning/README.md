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
- 입력 템플릿
```
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}
```

### 2-2. 입력과 출력 완성
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

## DPO
