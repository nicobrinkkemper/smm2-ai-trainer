import os
import glob
from safetensors.torch import load_file, save_file

print("Scanning for rogue tensor...")
model_dir = "/workspace/work/smm2-ai-trainer/smm2-gemma-4-merged"
safetensor_files = glob.glob(f"{model_dir}/*.safetensors")

for file_path in safetensor_files:
    print(f"Processing {file_path}...")
    tensors = load_file(file_path)
    
    rogue_key = "model.embed_tokens_per_layer.weight"
    if rogue_key in tensors:
        print(f"Found rogue tensor in {file_path}! Deleting...")
        del tensors[rogue_key]
        
        # Save the file back without the rogue tensor
        save_file(tensors, file_path)
        print("Tensor successfully deleted and file re-saved.")
    else:
        print("No rogue tensor found in this shard.")

print("All files processed!")
