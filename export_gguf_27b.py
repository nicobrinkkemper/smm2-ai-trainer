from unsloth import FastLanguageModel
import torch
import os

# We bypass the buggy hf-transfer just in case
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'

# Ensure HuggingFace uses the big persistent volume for its massive 27B model downloads
os.environ['HF_HOME'] = '/workspace/work/hf-cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/workspace/work/hf-cache'

max_seq_length = 1024
model_name = "./lora_model_27b"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)

# Export to Ollama directly in q4_k_m (4-bit quantization, optimal for Ollama/local GPUs)
print("Starting GGUF export for 27B model...")
model.save_pretrained_gguf("smm2-gemma-27b", tokenizer, quantization_method = "q4_k_m")
print("Export complete!")
