#!/bin/bash
set -e

echo "Cloning and building llama.cpp using modern CMake..."
cd /workspace/work/smm2-ai-trainer
if [ ! -d "llama.cpp" ]; then
  git clone https://github.com/ggerganov/llama.cpp.git
fi
cd llama.cpp

# Clean and build
rm -rf build
cmake -B build -G Ninja
cmake --build build -j 9

echo "Installing converter dependencies..."
pip install -r requirements.txt

echo "Converting F16 model..."
python3 convert_hf_to_gguf.py /workspace/work/smm2-ai-trainer/smm2-gemma-27b \
  --outfile /workspace/work/smm2-ai-trainer/smm2-gemma-27b-f16.gguf

echo "Quantizing to Q4_K_M..."
./build/bin/llama-quantize /workspace/work/smm2-ai-trainer/smm2-gemma-27b-f16.gguf \
  /workspace/work/smm2-ai-trainer/smm2-gemma-27b-Q4_K_M.gguf Q4_K_M

echo "Done! Your model is at /workspace/work/smm2-ai-trainer/smm2-gemma-27b-Q4_K_M.gguf"
