from unsloth import FastLanguageModel
import torch
import os

# Bypass the HF transfer bug
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'

print("1. Loading base model and adapters...")
# We load the base model AND apply the adapters so the graph explicitly contains the PEFT (LoRA) modules
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "principled-intelligence/gemma-4-E4B-it-text-only",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = False,
)

model.load_adapter("./lora_model_gemma4")

print("2. Mathematically merging adapters into base weights...")
# This forces Unsloth/Transformers to physically merge the weights in VRAM so the output is a standard model
model = model.merge_and_unload()

print("3. Spoofing architecture configuration for llama.cpp...")
# Now that the weights are merged, we spoof the config so llama.cpp doesn't crash on "Gemma4TextModel"
model.config.architectures = ["Gemma2ForCausalLM"]
model.config.attn_logit_softcapping = 50.0
model.config.final_logit_softcapping = 30.0
model.config.sliding_window = 4096

print("4. Saving fully merged 16-bit model to disk...")
# We save the raw 16-bit tensors and config.json to disk using standard HF serialization
model.save_pretrained("./smm2-gemma-4-merged", safe_serialization=True)
tokenizer.save_pretrained("./smm2-gemma-4-merged")

# ALSO copy the tokenizer.model from the original cache because Unsloth forgets it sometimes
os.system("cp /workspace/work/hf-cache/models--principled-intelligence--gemma-4-E4B-it-text-only/snapshots/*/tokenizer.model ./smm2-gemma-4-merged/ || true")

print("Merge complete! Run ./convert_manual_gemma4.sh on the ./smm2-gemma-4-merged directory now.")
