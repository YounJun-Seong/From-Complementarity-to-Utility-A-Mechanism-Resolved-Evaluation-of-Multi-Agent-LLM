#!/bin/bash
# Layer 3 L0 re-run of flagged questions with a 128k-context Llama server (Amendment 1).
ROOT=$(cd "$(dirname "$0")/.." && pwd); VPY=<VLLM_ENV>/bin
GPU=${1:-1}; PORT=${2:-8051}; MODEL=meta-llama/Llama-3.1-8B-Instruct
source $ROOT/env.sh
echo "$(date +%H:%M:%S) [$GPU] start server $MODEL (128k)"
CUDA_VISIBLE_DEVICES=$GPU nohup $VPY/vllm serve $MODEL --port $PORT --dtype bfloat16 --gpu-memory-utilization 0.85 \
    --max-model-len 131072 --generation-config vllm --disable-log-requests > $ROOT/logs/vllm_layer3_llama_128k.log 2>&1 &
until curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1; do
  sleep 5; pgrep -f "^$VPY/python.*vllm serve $MODEL --port $PORT" >/dev/null || { echo "server died"; exit 1; }
done
echo "$(date +%H:%M:%S) [$GPU] ready"
export TMC_BASE_URL=http://127.0.0.1:$PORT/v1 TMC_MODEL_OVERRIDE=$MODEL
cd $ROOT/external/MedLA
WANDB_MODE=offline TMC_RUN_ID=L3-L0-run1 MEDLA_CONDITION=L3-L0 MEDLA_OUT=$ROOT/runs/layer3/L0/run1.jsonl \
  python main.py --dataset medqa_us --model openai --llm_name gpt-4o-mini-2024-07-18 --baseline_bool 0 \
  --num_samples -1 --num_threads 64 --tag L3-L0-run1-resume128k >> $ROOT/runs/layer3/L0/run1_resume.log 2>&1
echo "$(date +%H:%M:%S) [$GPU] L0 resume done: $(wc -l < $ROOT/runs/layer3/L0/run1.jsonl)/1273"
pkill -f "^$VPY/python.*vllm serve $MODEL --port $PORT"
while pgrep -f "^$VPY/python.*vllm serve $MODEL --port $PORT" >/dev/null; do sleep 3; done
echo "$(date +%H:%M:%S) [$GPU] server down"
