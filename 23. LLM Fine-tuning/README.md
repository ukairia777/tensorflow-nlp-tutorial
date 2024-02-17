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
- 위 템플릿에 따라서 만들어진 데이터 예시는 다음과 같습니다.
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

#### 3-2. 입력과 출력 완성
- 위 템플릿에 따라서 만들어진 데이터 예시는 다음과 같습니다.
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

### 4. Merge
```
!wget https://raw.githubusercontent.com/ukairia777/LLM-Finetuning-tutorial/main/merge.py

!python merge.py \
    --base_model_name_or_path beomi/llama-2ko-7b \
    --peft_model_path ./title-generator/checkpoint-2000 \
    --output_dir ./output_dir
```


### 5. Inference
- 학습한 모델을 불러와서 테스트합니다.
```
import torch.nn as nn
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained('./output_dir')
tokenizer = AutoTokenizer.from_pretrained('./output_dir')

model.eval()
inputs = tokenizer(input_text, return_tensors="pt")
model = nn.DataParallel(model)
model.cuda()

eos_token_id = tokenizer.eos_token_id

with torch.no_grad():
    outputs = model.module.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=512, eos_token_id=eos_token_id)
    print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0])
```


### 6. 실제 학습 전, 후 비교
- 다음은 2000 step(약 1 Epoch)을 학습하고나서 llama-2-ko-7b의 출력 결과 비교입니다.
- 입력은 다음과 같았습니다.
- 모델의 입력은 `### 제목:`까지 주어져야 합니다. 그래야만 모델이 제목을 작성할 차례임을 인지합니다.
```
당신은 주어진 본문으로부터 적절한 제목을 생성하는 제목 생성기입니다. 본문이 주어지면 제목을 만드세요.

### 본문:
주말 포근한 날씨 속에 나들이객 발길이 전국 유명산과 유원지, 명소 등으로 이어졌다. 봄의 전령인 홍매화가 피기 시작한 전남 순천 매곡동 탐매마을, 금둔사, 낙안읍성마을에도 시민과 관광객이 몰렸다. 봄이면 매화나무로 절경을 이루는 광양 섬진강 변에도 하얀 매화꽃이 서서히 자태를 드러냈다. 제주 서귀포시 산방산과 성산일출봉 일대에는 노란 유채꽃이 활짝 펴 하루 종일 관광객의 발길이 끊이지 않았다. 협재, 함덕, 월정 등 도내 주요 해변은 모처럼 따사로운 햇살을 즐기려는 이들로 북적였고 주요 오름과 한라산에도 탐방객이 이어졌다. 국립공원공단 지리산국립공원전남사무소는 예년보다 봄이 빨리 찾아오면서 화엄사 일대에 복수초와 히어리가 개화했다고 16일 밝혔다. 연합뉴스 viewer 국립공원도 인산인해를 이뤘다. 속리산국립공원에는 이날 오후 1시 30분 기준 3900여명의 탐방객이 입장해 법주사와 세심정을 잇는 세조길을 거닐었다. 월악산 국립공원에도 900여명이 방문해 절경을 감상했다. 계룡산 국립공원을 찾은 등산객들 역시 포근한 날씨 속에 봄기운을 즐겼다. 무주 덕유산, 정읍 내장산, 완주 모악산, 대구 팔공산과 비슬산, 청송 주왕산, 가지산과 신불산 등이 있는 울산 영남알프스 등에도 많은 등산객이 찾았다. 부산 해운대와 광안리 해수욕장에는 낮 최고 12도까지 올라갔다. 전국의 주요 유원지도 북적였다. 용인 에버랜드를 찾은 시민들은 판다 '바오 패밀리'를 구경하며 즐거운 한때를 보냈다. 인근 한국민속촌에서도 방문객들이 각종 민속 체험을 하고 고궁을 거닐었다. 경북 경주 보문단지, 경주월드를 찾은 관광객들은 호숫가를 산책하거나 놀이기구를 타며 즐거워했다. 전주 한옥마을에는 평소보다 많은 관광객이 찾아 경기전과 전동성당, 향교 등을 둘러보고 기념사진을 찍었다. 16일 강원 산간과 동해안에 전날 폭설이 내려 강원 강릉시 경포호 너머로 백두대간이 흰눈에 덮여 설경을 자랑하고 있다. 연합뉴스 viewer 강원도 평창 용평스키장과 정선 하이원 스키장에는 각각 3000여명과 9000여명이 방문하는 등 이날 도내 스키장에는 2만여명의 인파가 몰렸다. 설악산, 태백산, 오대산, 치악산 등 주요 국립공원을 찾은 탐방객들은 눈 쌓인 산을 걸으며 마지막 겨울 낭만에 흠뻑 빠졌다. 근교의 복합쇼핑몰과 대형 아웃렛에도 쇼핑과 식사하러 온 많은 시민이 몰려 주변 도로에 차량 정체가 빚어지기도 했다.

### 제목:
```

#### 6-1. 기본 모델
- 제목이라기에는 다소 어색한 평문을 생성.
```
주말 포근한 날씨 속에 전국 유명산과 유원지, 명소 등에 봄 정취를 느끼려는 나들이객 발길이 이어졌습니다.
```

#### 6-2. 튜닝된 모델
- 뉴스 기사 제목과 매우 유사한 형태의 텍스트를 생성.
```
범의 전령 매화부터 유채꽃까지... 전국 유명산·유원지 북적
```

### 7. Huggingface Submit
```
from huggingface_hub import HfApi
api = HfApi()
username = "허깅페이스 ID"

MODEL_NAME = 'llama-2ko-7b-title-generator'

api.create_repo(
    token="허깅페이스 토큰",
    repo_id=f"{username}/{MODEL_NAME}",
    repo_type="model"
)

api.upload_folder(
    token="허깅페이스 토큰",
    repo_id=f"{username}/{MODEL_NAME}",
    folder_path="merged",
)
```

## DPO
```
!wget https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/23.%20LLM%20Fine-tuning/dpo-trainer.py
!python dpo-trainer.py \
    --base_model Qwen/Qwen1.5-72B \
    --data-path  Intel/orca_dpo_pairs \
    --output_dir ./lora \
    --num_epochs 3 \
    --batch_size 16 \
    --micro_batch_size 2 \
    --learning_rate 1e-5 \
    --lora_r 16 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    # --lora_target_modules ["embed_tokens", "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"] \
    --lora_target_modules ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"] \
    --lr_scheduler 'linear' \
    --warmup_ratio 0.1 \
    --cutoff_len 4096 \
```
