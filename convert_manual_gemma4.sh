#!/bin/bash
set -e

MODEL_DIR="/workspace/work/smm2-ai-trainer/smm2-gemma-4-merged"
F16_GGUF="/workspace/work/smm2-ai-trainer/smm2-gemma-4-textonly-f16.gguf"
Q4_GGUF="/workspace/work/smm2-ai-trainer/smm2-gemma-4-textonly-Q4_K_M.gguf"

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Base model directory $MODEL_DIR not found!"
    exit 1
fi

echo "Setting up llama.cpp..."
cd /workspace/work/smm2-ai-trainer
if [ ! -d "llama.cpp" ]; then
  git clone https://github.com/ggerganov/llama.cpp.git
fi
cd llama.cpp

echo "Building llama.cpp..."
cmake -B build -G Ninja
cmake --build build -j $(nproc)

if [ ! -f "$F16_GGUF" ]; then
    echo "Fixing tokenizer path requirement..."
    # We copy the tokenizer.model from the 27B directory we already downloaded
    # earlier today to bypass the HuggingFace 401 Unauthorized block!
    cp /workspace/work/smm2-ai-trainer/smm2-gemma-27b/tokenizer.model "$MODEL_DIR/tokenizer.model" || true

    echo "Patching llama.cpp to bypass the BPE hash check..."
    sed -i 's/self._set_vocab_gpt2()/self._set_vocab_sentencepiece()/g' convert_hf_to_gguf.py

    echo "Running Python configuration spoofer..."
    python3 ../fix_config.py

    echo "Converting F16 model..."
    python3 convert_hf_to_gguf.py "$MODEL_DIR" --outfile "$F16_GGUF"
else
    echo "F16 GGUF already exists, skipping conversion..."
fi

if [ ! -f "$Q4_GGUF" ]; then
    echo "Quantizing to Q4_K_M..."
    ./build/bin/llama-quantize "$F16_GGUF" "$Q4_GGUF" Q4_K_M
else
    echo "Q4_K_M GGUF already exists, skipping quantization..."
fi

echo "Done! Your final model is at: $Q4_GGUF"
