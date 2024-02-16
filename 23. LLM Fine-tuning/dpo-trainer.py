import os
import sys
from typing import List
import fire
import torch
from datasets import load_dataset #, Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer
import bitsandbytes as bnb

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return list(lora_module_names)

def print_trainable_parameters(model):
  """
  Prints the number of trainable parameters in the model.
  """
  trainable_params = 0
  all_param = 0
  for _, param in model.named_parameters():
    all_param += param.numel()
    if param.requires_grad:
      trainable_params += param.numel()
  print(
      f"trainable params: {trainable_params} || all params: {all_param} || trainables%: {100 * trainable_params / all_param}"
  )

def train(
    # model/data params
    base_model: str = "",
    data_path: str = "",
    output_dir: str = "",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 8,
    num_epochs: int = 1,
    learning_rate: float = 3e-4,
    cutoff_len: int = 4096,
    lr_scheduler: str = "cosine",
    warmup_ratio: float = 0.1, 
    # lora hyperparams
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    # from peft docs: ["q_proj", "k_proj", "v_proj", "o_proj", "fc_in", "fc_out", "wte", "gate_proj", "down_proj", "up_proj"]
    lora_target_modules: List[str] = ["gate_proj", "down_proj", "up_proj"],
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"lr_scheduler: {lr_scheduler}\n"
            f"warmup_ratio: {warmup_ratio}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    # from huggingface_hub import login
    # login(token='[...your_token...]')
    
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"

    # 1. Define policy and reference models
    model = AutoModelForCausalLM.from_pretrained(
        base_model, # location of saved SFT model
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map = device_map
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    print(type(model))
    print(model)
    print("length of tokenizer:",len(tokenizer))

    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    print("pre-trained model's BOS EOS and PAD token id:",bos,eos,pad," => It should be 1 2 None")

    # tokenizer.pad_token_id = 0
    tokenizer.padding_side = "right"

    # 2. Define dataset
    def return_prompt_and_responses(samples):
        return {
            "prompt": "### User:\n" + samples["question"] + "\n\n### Assistant:\n",
            "chosen": samples["chosen"],
            "rejected": samples["rejected"],
        }

    dataset = load_dataset(data_path)
    train_dataset = dataset.map(return_prompt_and_responses)
    train_dataset = train_dataset.filter(
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= cutoff_len
        and len(x["prompt"]) + len(x["rejected"]) <= cutoff_len
    )
    train_dataset = train_dataset["train"].shuffle()
    print('### 첫번째 샘플 출력 ###')
    print(train_dataset['prompt'][0])
    print(train_dataset['chosen'][0])
    print(train_dataset['rejected'][0])
    
    # 3. Define hyperparameters
    training_args = TrainingArguments(
        num_train_epochs= num_epochs,
        per_device_train_batch_size=micro_batch_size,
        logging_steps=1,
        save_steps=10,
        save_total_limit=2,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        output_dir=output_dir,
        lr_scheduler_type=lr_scheduler,
        warmup_ratio=warmup_ratio,
        optim='paged_adamw_32bit',
        bf16=True,
        # remove_unused_columns=False,
        run_name="dpo_trainer",
    )

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        # target_modules=lora_target_modules,
        target_modules=find_all_linear_names(model),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    print_trainable_parameters(model)
    
    # DPO trainer
    dpo_trainer = DPOTrainer(
        model,
        ref_model=None,
        args=training_args,
        beta=0.1,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    # train
    dpo_trainer.train()
    dpo_trainer.save_model(output_dir)

    # save
    output_dir = os.path.join(output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)

if __name__ == "__main__":
    torch.cuda.empty_cache() 
    fire.Fire(train)