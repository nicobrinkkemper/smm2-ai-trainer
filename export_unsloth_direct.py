from unsloth import FastLanguageModel
import torch
import os

# Bypass the HF transfer bug
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'

print("Loading base model and applying adapters directly for GGUF export...")
# We must load the base model and apply the adapters IN THIS SCRIPT so Unsloth sees the LoRA modules!
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "principled-intelligence/gemma-4-E4B-it-text-only",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

model.load_adapter("./lora_model_gemma4")

print("Exporting to GGUF using Unsloth native wrapper...")
model.save_pretrained_gguf("smm2-gemma-4-textonly", tokenizer, quantization_method = "q4_k_m")
print("Done! GGUF created.")
