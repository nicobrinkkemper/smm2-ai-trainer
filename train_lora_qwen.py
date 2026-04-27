from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import os

# Disable buggy high-speed downloader
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'

max_seq_length = 2048

print("Loading Qwen 3 Coder 30B MoE base model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-Coder-30B-A3B-Instruct",
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

print("Loading dataset...")
dataset = load_dataset("json", data_files="dataset_v3_chatml.jsonl", split="train")

def formatting_prompts_func(examples):
    conversations = examples["messages"]
    texts = [tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False) for conv in conversations]
    return { "text" : texts }

dataset = dataset.map(formatting_prompts_func, batched = True, num_proc=None) # num_proc=None prevents OOM on 85k dataset

print(f"Starting Qwen 3 Coder LoRA fine-tuning on {len(dataset)} examples...")
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = None,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

trainer.train()

print("Training complete! LoRA adapters saved to ./lora_model_qwen")
model.save_pretrained("./lora_model_qwen")
tokenizer.save_pretrained("./lora_model_qwen")
