from unsloth import FastLanguageModel
import os

# Bypass the HF transfer bug
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'

print("Loading Qwen base model and adapters...")
# Qwen3 is natively supported by llama.cpp and unsloth, so we CAN use the built-in wrapper!
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-Coder-30B-A3B-Instruct",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = False, # 16-bit for clean merge
)

model.load_adapter("./lora_model_qwen")

print("Exporting Qwen3 Coder directly to GGUF Q4_K_M via Unsloth...")
model.save_pretrained_gguf("smm2-qwen3-coder", tokenizer, quantization_method = "q4_k_m")
print("Export complete! Find your GGUF file in the smm2-qwen3-coder folder.")
