#!/bin/bash
# Script pour construire le moteur TensorRT-LLM pour Qwen 2.5 1.5B sur Jetson
# À exécuter directement sur le Jetson Orin Nano

set -e

# Configuration
MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR="/workspace/trt_engines/qwen2.5-1.5b"
CHECKPOINT_DIR="/workspace/checkpoints/qwen2.5-1.5b"
MAX_BATCH_SIZE=4
MAX_INPUT_LEN=2048
MAX_OUTPUT_LEN=512
DTYPE="float16"  # float16 pour Jetson (pas de support int8 natif sur tous les modèles)

echo "=================================================="
echo "Building TensorRT-LLM Engine for Qwen 2.5 1.5B"
echo "=================================================="

# Créer les répertoires
mkdir -p ${OUTPUT_DIR}
mkdir -p ${CHECKPOINT_DIR}

# Étape 1 : Télécharger et convertir le modèle HuggingFace
echo "[1/3] Converting HuggingFace model to TensorRT-LLM checkpoint..."
python3 /app/tensorrt_llm/examples/qwen/convert_checkpoint.py \
    --model_dir ${MODEL_NAME} \
    --output_dir ${CHECKPOINT_DIR} \
    --dtype ${DTYPE} \
    --tp_size 1  # Tensor Parallelism = 1 (mono-GPU)

# Étape 2 : Construire le moteur TensorRT
echo "[2/3] Building TensorRT engine (this will take 10-20 minutes)..."
trtllm-build \
    --checkpoint_dir ${CHECKPOINT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --gemm_plugin auto \
    --max_batch_size ${MAX_BATCH_SIZE} \
    --max_input_len ${MAX_INPUT_LEN} \
    --max_output_len ${MAX_OUTPUT_LEN} \
    --max_beam_width 1 \
    --gpt_attention_plugin ${DTYPE} \
    --remove_input_padding enable \
    --paged_kv_cache enable \
    --use_custom_all_reduce disable

echo "[3/3] Engine built successfully!"
echo "Engine location: ${OUTPUT_DIR}"
echo ""
echo "Next steps:"
echo "  1. Start the TensorRT-LLM server with this engine"
echo "  2. Configure your ASR pipeline to use LLM_ENDPOINT=http://localhost:8000"