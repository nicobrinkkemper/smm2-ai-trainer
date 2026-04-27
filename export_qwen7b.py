from unsloth import FastLanguageModel
import os
from huggingface_hub import HfApi

os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'

print("Loading Qwen 2.5 Coder 7B base model and adapters...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-Coder-7B-Instruct",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = False,
)

model.load_adapter("./lora_model_qwen7b")

print("Exporting directly to GGUF Q4_K_M via Unsloth...")
model.save_pretrained_gguf("smm2-qwen2.5-coder-7b", tokenizer, quantization_method = "q4_k_m")
print("Export complete!")

print("Pushing GGUF to HuggingFace Hub...")
try:
    api = HfApi()
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN environment variable not set.")
    else:
        import glob
        gguf_files = glob.glob("smm2-qwen2.5-coder-7b/*Q4_K_M.gguf")
        
        if gguf_files:
            file_path = gguf_files[0]
            file_name = os.path.basename(file_path)
            repo_id = "Geitje1/smm2-qwen2.5-coder-7b"
            
            print(f"Creating repo {repo_id}...")
            api.create_repo(repo_id=repo_id, exist_ok=True, token=token)
            
            print(f"Uploading {file_name}...")
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=file_name,
                repo_id=repo_id,
                repo_type="model",
                token=token
            )
            print(f"Successfully uploaded {file_name} to https://huggingface.co/{repo_id}")
        else:
            print("ERROR: Could not find generated GGUF file.")
except Exception as e:
    print(f"Failed to upload to HuggingFace: {e}")
