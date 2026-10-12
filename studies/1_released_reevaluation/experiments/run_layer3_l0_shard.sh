#!/bin/bash
# Layer 3 L0 re-run split across GPUs (execution only): bash run_layer3_l0_shard.sh <gpu> <port> <k> <n> <util>
GPU=$1; PORT=$2; K=$3; N=$4; UTIL=${5:-0.85}
ROOT=$(cd "$(dirname "$0")/.." && pwd); VPY=<VLLM_ENV>/bin; MODEL=meta-llama/Llama-3.1-8B-Instruct
source $ROOT/env.sh
echo "$(date +%H:%M:%S) [$GPU] start server $MODEL (128k) shard $K/$N"
CUDA_VISIBLE_DEVICES=$GPU nohup $VPY/vllm serve $MODEL --port $PORT --dtype bfloat16 --gpu-memory-utilization $UTIL \
    --max-model-len 131072 --generation-config vllm --disable-log-requests > $ROOT/logs/vllm_layer3_shard$K.log 2>&1 &
until curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1; do
  sleep 5; pgrep -f "^$VPY/python.*vllm serve $MODEL --port $PORT" >/dev/null || { echo "server died"; exit 1; }
done
echo "$(date +%H:%M:%S) [$GPU] ready"
export TMC_BASE_URL=http://127.0.0.1:$PORT/v1 TMC_MODEL_OVERRIDE=$MODEL
OUT=$ROOT/runs/layer3/L0/run1_shard$K.jsonl
cd $ROOT/external/MedLA
WANDB_MODE=offline TMC_RUN_ID=L3-L0-run1 MEDLA_CONDITION=L3-L0 MEDLA_OUT=$OUT MEDLA_SHARD=$K/$N \
  MEDLA_SKIP_FROM=$ROOT/runs/layer3/L0/run1.jsonl \
  python main.py --dataset medqa_us --model openai --llm_name gpt-4o-mini-2024-07-18 --baseline_bool 0 \
  --num_samples -1 --num_threads 64 --tag L3-L0-run1-shard$K >> $ROOT/runs/layer3/L0/run1_shard$K.log 2>&1
echo "$(date +%H:%M:%S) [$GPU] shard $K done: $(wc -l < $OUT) rows"
pkill -f "^$VPY/python.*vllm serve $MODEL --port $PORT"
while pgrep -f "^$VPY/python.*vllm serve $MODEL --port $PORT" >/dev/null; do sleep 3; done
echo "$(date +%H:%M:%S) [$GPU] server down"
