from unsloth import FastLanguageModel
import torch
import os

os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'

print("1. Loading base model and adapters...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "principled-intelligence/gemma-4-E4B-it-text-only",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = False,
)

model.load_adapter("./lora_model_gemma4")

print("2. Spoofing architecture configuration for llama.cpp...")
model.config.architectures = ["Gemma2ForCausalLM"]
model.config.attn_logit_softcapping = 50.0
model.config.final_logit_softcapping = 30.0
model.config.sliding_window = 4096

print("3. Saving fully merged 16-bit model to disk...")
model.save_pretrained_merged("./smm2-gemma-4-merged", tokenizer, save_method = "merged_16bit")

print("Merge complete! Run ./convert_manual_gemma4.sh on the ./smm2-gemma-4-merged directory now.")
