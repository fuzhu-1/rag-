#!/bin/bash
# ============================================================
# Enterprise-RAG: vLLM 部署 Qwen2.5-7B-Instruct
# 暴露 OpenAI 兼容 API 接口
# ============================================================
set -e

# ── Configuration ──
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
GPU_MEMORY="${GPU_MEMORY:-0.9}"          # GPU 显存利用率
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"   # 最大上下文长度
DTYPE="${DTYPE:-auto}"                    # 数据类型: auto, float16, bfloat16
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"   # 张量并行度 (多卡)
QUANTIZATION="${QUANTIZATION:-}"          # 量化方法: awq, gptq, squeezellm

echo "=========================================="
echo "  Enterprise-RAG: vLLM Model Deployment"
echo "=========================================="
echo "  Model:     $MODEL_NAME"
echo "  Host:      $HOST"
echo "  Port:      $PORT"
echo "  GPU Memory: ${GPU_MEMORY}"
echo "  Max Length: $MAX_MODEL_LEN"
echo "  dtype:     $DTYPE"
echo "=========================================="

# ── Check vLLM installation ──
if ! python -c "import vllm" 2>/dev/null; then
    echo "[INFO] Installing vLLM..."
    pip install vllm -q
fi

# ── Download model if needed ──
MODEL_PATH="$MODEL_NAME"
if [ ! -d "$MODEL_NAME" ] && [[ "$MODEL_NAME" != *"/"* ]]; then
    echo "[INFO] Model not found locally, will download from HuggingFace..."
fi

# ── Build vLLM command ──
CMD="python -m vllm.entrypoints.openai.api_server"
CMD="$CMD --model $MODEL_NAME"
CMD="$CMD --host $HOST"
CMD="$CMD --port $PORT"
CMD="$CMD --gpu-memory-utilization $GPU_MEMORY"
CMD="$CMD --max-model-len $MAX_MODEL_LEN"
CMD="$CMD --dtype $DTYPE"
CMD="$CMD --trust-remote-code"

if [ -n "$TENSOR_PARALLEL" ] && [ "$TENSOR_PARALLEL" -gt 1 ]; then
    CMD="$CMD --tensor-parallel-size $TENSOR_PARALLEL"
fi

if [ -n "$QUANTIZATION" ]; then
    CMD="$CMD --quantization $QUANTIZATION"
fi

# ── Launch ──
echo "[INFO] Starting vLLM server..."
echo "[CMD] $CMD"
echo ""
echo "Once started, the API will be available at:"
echo "  - OpenAI Compatible: http://$HOST:$PORT/v1"
echo "  - API Docs:          http://$HOST:$PORT/docs"
echo ""

exec $CMD
