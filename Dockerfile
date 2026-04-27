# Use the official Unsloth image as the base to guarantee CUDA/PyTorch compatibility
FROM unsloth/unsloth:latest

# Set environment variables for HuggingFace caching inside the image
ENV HF_HOME=/workspace/hf-cache
ENV HUGGINGFACE_HUB_CACHE=/workspace/hf-cache
ENV HF_HUB_ENABLE_HF_TRANSFER=0

# Create working directory
WORKDIR /workspace

# Clone the training repository
RUN git clone https://github.com/nicobrinkkemper/smm2-ai-trainer.git
WORKDIR /workspace/smm2-ai-trainer

# Pre-compile llama.cpp so it's permanently baked into the image
RUN echo "Pre-compiling llama.cpp..." && \
    git clone https://github.com/ggerganov/llama.cpp.git && \
    cd llama.cpp && \
    cmake -B build -G Ninja && \
    cmake --build build -j $(nproc)

# Pre-download the dataset from HuggingFace so it's baked into the image
RUN echo "Caching dataset..." && \
    python3 -c "from datasets import load_dataset; load_dataset('Geitje1/smm2-decomp-chatml', split='train')"

# WARNING: Pre-downloading the base model (Qwen 3) will make this Docker image >20GB!
# If you have fast upload speed, uncomment the line below to bake the model into the image:
# RUN echo "Caching base model..." && python3 -c "from unsloth import FastLanguageModel; FastLanguageModel.from_pretrained('unsloth/Qwen3-Coder-30B-A3B-Instruct', load_in_4bit=True)"

# Default command when the container boots
CMD ["/bin/bash"]
