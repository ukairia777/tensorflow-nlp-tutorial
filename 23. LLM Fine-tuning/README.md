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

### 2. 템플릿
- sft-trainer.py안에 있는 `generate_and_tokenize_prompt` 함수를 수정하여 여러분만의 프롬프트 템플릿을 만들어야 합니다.

#### 2-1. 템플릿 예시
- 예를 들어서 알파카 데이터셋을 사용한다면 아래와 같은 템플릿이 있을 수 있습니다.
입력 템플릿
```
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"
```

## DPO
