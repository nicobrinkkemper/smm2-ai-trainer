from unsloth import FastLanguageModel
import torch

max_seq_length = 2048
# Load the base model and immediately apply your trained LoRA adapters
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "principled-intelligence/gemma-4-E4B-it-text-only",
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = False, # We load it in 16-bit to get a clean merge
)

# Load your custom adapters
model.load_adapter("./lora_model_gemma4")

print("Merging LoRA adapters into base model and saving to ./merged_gemma4_16bit...")
# This physically merges the weights and saves the complete model + config.json
model.save_pretrained_merged("./merged_gemma4_16bit", tokenizer, save_method = "merged_16bit")
print("Merge complete!")
