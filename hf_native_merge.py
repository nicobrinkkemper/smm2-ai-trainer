from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

print("1. Loading raw base model via HF...")
base_model = AutoModelForCausalLM.from_pretrained(
    "principled-intelligence/gemma-4-E4B-it-text-only",
    torch_dtype="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("principled-intelligence/gemma-4-E4B-it-text-only")

print("2. Loading and merging LoRA adapters...")
model = PeftModel.from_pretrained(base_model, "./lora_model_gemma4")
merged_model = model.merge_and_unload()

print("3. Spoofing architecture to Gemma2ForCausalLM for llama.cpp compatibility...")
merged_model.config.architectures = ["Gemma2ForCausalLM"]
merged_model.config.model_type = "gemma2"
merged_model.config.attn_logit_softcapping = 50.0
merged_model.config.final_logit_softcapping = 30.0
merged_model.config.sliding_window = 4096

print("4. Saving pure HF tensors to disk...")
merged_model.save_pretrained("./smm2-gemma-4-merged", safe_serialization=True)
tokenizer.save_pretrained("./smm2-gemma-4-merged")
print("Native merge complete!")
