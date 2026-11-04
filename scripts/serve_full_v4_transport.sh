#!/usr/bin/env bash
set -euo pipefail
DEPLOY_DIR=/root/barbet-context-sft-data/inference/qwen38
export PATH="$DEPLOY_DIR/venv/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_LAUNCH_BLOCKING=1
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_CACHE_ROOT="$DEPLOY_DIR/cache/vllm"
export XDG_CACHE_HOME="$DEPLOY_DIR/cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export VLLM_PLE_CPU_OFFLOAD=0 VLLM_USE_RAY_V2_EXECUTOR_BACKEND=0 VLLM_BARBET_DISABLE_SHM_BROADCAST=1
export RAY_USAGE_STATS_ENABLED=0 NCCL_SHM_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_CUMEM_HOST_ENABLE=0
export NCCL_P2P_LEVEL=SYS NCCL_P2P_USE_CUDA_MEMCPY=0 OMP_NUM_THREADS=8
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN OPENAI_API_KEY DASHSCOPE_API_KEY
exec "$DEPLOY_DIR/venv/bin/python" -u "$DEPLOY_DIR/ray_serve.py" serve "$DEPLOY_DIR/model-fp8" \
 --served-model-name Qwen/Qwen3.8-Flash-Next-FP8 --host 127.0.0.1 --port 8002 \
 --distributed-executor-backend ray --tensor-parallel-size 8 --enable-expert-parallel --moe-backend triton \
 --dtype bfloat16 --gpu-memory-utilization 0.85 --enforce-eager \
 --max-model-len 131072 --max-num-seqs 2 --max-num-batched-tokens 8192 \
 --enable-chunked-prefill --enable-prefix-caching --no-enable-flashinfer-autotune \
 --limit-mm-per-prompt '{"image":0,"video":0}' --reasoning-parser qwen3 \
 --default-chat-template-kwargs '{"enable_thinking":false}'
