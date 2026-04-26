#!/bin/bash
set -e

# Hardcoded to the 27B model right now, could be parameterized later
MODEL_DIR="/workspace/work/smm2-ai-trainer/smm2-gemma-27b"
F16_GGUF="/workspace/work/smm2-ai-trainer/smm2-gemma-27b-f16.gguf"
Q4_GGUF="/workspace/work/smm2-ai-trainer/smm2-gemma-27b-Q4_K_M.gguf"

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Base model directory $MODEL_DIR not found!"
    echo "Did Unsloth successfully export the 16-bit merged weights?"
    exit 1
fi

echo "Setting up llama.cpp..."
cd /workspace/work/smm2-ai-trainer
if [ ! -d "llama.cpp" ]; then
  git clone https://github.com/ggerganov/llama.cpp.git
fi
cd llama.cpp

# Pull latest in case repo is stale
git pull origin master

echo "Building llama.cpp..."
rm -rf build
cmake -B build -G Ninja
cmake --build build -j $(nproc)

echo "Installing converter dependencies..."
pip install -r requirements.txt

# Only convert if the f16 file doesn't already exist (saves 15 minutes if you re-run)
if [ ! -f "$F16_GGUF" ]; then
    echo "Converting F16 model..."
    python3 convert_hf_to_gguf.py "$MODEL_DIR" --outfile "$F16_GGUF"
else
    echo "F16 GGUF already exists, skipping conversion..."
fi

# Only quantize if the Q4 file doesn't already exist
if [ ! -f "$Q4_GGUF" ]; then
    echo "Quantizing to Q4_K_M..."
    ./build/bin/llama-quantize "$F16_GGUF" "$Q4_GGUF" Q4_K_M
else
    echo "Q4_K_M GGUF already exists, skipping quantization..."
fi

echo "Done! Your final model is at: $Q4_GGUF"
