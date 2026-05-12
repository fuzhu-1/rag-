#!/bin/bash
# ============================================================
# Enterprise-RAG: llama.cpp GGUF 量化部署 Qwen2.5-7B
# 适用于资源受限环境 (CPU / 消费级 GPU)
# ============================================================
set -e

# ── Configuration ──
MODEL_NAME="${MODEL_NAME:-Qwen2.5-7B-Instruct}"
GGUF_QUANT="${GGUF_QUANT:-Q4_K_M}"        # 量化级别
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
N_CTX="${N_CTX:-8192}"                    # 上下文窗口
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"         # GPU 层数 (-1 = 全部)
THREADS="${THREADS:-8}"

echo "=========================================="
echo "  Enterprise-RAG: llama.cpp Deployment"
echo "=========================================="
echo "  Model:      $MODEL_NAME"
echo "  Quant:      $GGUF_QUANT"
echo "  Host:       $HOST"
echo "  Port:       $PORT"
echo "  Context:    $N_CTX"
echo "  GPU Layers: $N_GPU_LAYERS"
echo "=========================================="

# ── Clone llama.cpp if needed ──
LLAMA_DIR="./llama.cpp"
if [ ! -d "$LLAMA_DIR" ]; then
    echo "[INFO] Cloning llama.cpp..."
    git clone https://github.com/ggerganov/llama.cpp.git "$LLAMA_DIR"
    cd "$LLAMA_DIR"
    make -j "$THREADS"
    cd -
fi

# ── Download GGUF model ──
GGUF_FILE="./models/${MODEL_NAME}-${GGUF_QUANT}.gguf"
if [ ! -f "$GGUF_FILE" ]; then
    echo "[INFO] Downloading GGUF model: ${MODEL_NAME}-${GGUF_QUANT}.gguf"
    HF_REPO="Qwen/${MODEL_NAME}-GGUF"
    python "$LLAMA_DIR/scripts/hf.py" --repo "$HF_REPO" --file "*${GGUF_QUANT}*" --output ./models/
fi

# ── Launch server with OpenAI-compatible API ──
echo "[INFO] Starting llama.cpp server..."

CMD="$LLAMA_DIR/llama-server"
CMD="$CMD -m $GGUF_FILE"
CMD="$CMD --host $HOST"
CMD="$CMD --port $PORT"
CMD="$CMD -c $N_CTX"
CMD="$CMD -t $THREADS"
CMD="$CMD -ngl $N_GPU_LAYERS"

echo "[CMD] $CMD"
echo ""
echo "Once started, the API will be available at:"
echo "  - OpenAI Compatible: http://$HOST:$PORT/v1"
echo ""

exec $CMD
