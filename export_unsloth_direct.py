from unsloth import FastLanguageModel
import torch
import os

# Bypass the HF transfer bug
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'

print("Loading merged 16-bit model directly via Unsloth...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "./merged_gemma4_16bit",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = False,
)

# We temporarily spoof the config inside Unsloth's memory so its internal 
# llama.cpp wrapper doesn't panic about "Gemma4TextModel"
model.config.architectures = ["Gemma2ForCausalLM"]
model.config.attn_logit_softcapping = 50.0
model.config.final_logit_softcapping = 30.0
model.config.sliding_window = 4096

print("Exporting to GGUF using Unsloth native wrapper...")
model.save_pretrained_gguf("smm2-gemma-4-Q4_K_M", tokenizer, quantization_method = "q4_k_m")
print("Done! GGUF created.")
