import json
import glob

# Find the config.json in the HF cache
cache_pattern = "/workspace/work/hf-cache/models--principled-intelligence--gemma-4-E4B-it-text-only/snapshots/*/config.json"
cache_files = glob.glob(cache_pattern)

if not cache_files:
    print("Could not find original config.json in HF cache!")
    exit(1)

with open(cache_files[0], "r") as f:
    d = json.load(f)

# Spoof everything to make llama.cpp think it's a standard Gemma 2 model
d["architectures"] = ["Gemma2ForCausalLM"]
d["model_type"] = "gemma2"
d["attn_logit_softcapping"] = 50.0
d["final_logit_softcapping"] = 30.0
d["sliding_window"] = 4096

with open("/workspace/work/smm2-ai-trainer/smm2-gemma-4-merged/config.json", "w") as f:
    json.dump(d, f, indent=2)

print("Config manually spoofed and written to ./smm2-gemma-4-merged/config.json")
