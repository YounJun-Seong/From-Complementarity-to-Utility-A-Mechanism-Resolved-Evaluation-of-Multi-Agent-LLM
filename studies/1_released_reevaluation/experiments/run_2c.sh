#!/bin/bash
# Layer 2C driver (M1-expanded only): for each model on one GPU, bring up vLLM, run S x3 and M1 x3 (10 shards of 20
# questions each) on MedQA 0-199, wait, shut the server down, move to the next model.
#   bash experiments/run_2a.sh <gpu> <port> <model> [<model> ...]
GPU=$1; PORT=$2; shift 2
ROOT=$(cd "$(dirname "$0")/.." && pwd)
VPY=<VLLM_ENV>/bin
source $ROOT/env.sh
for MODEL in "$@"; do
  TAG=$(basename $MODEL); OUT=$ROOT/runs/layer2/2C/$TAG; mkdir -p $OUT/M1
  echo "$(date +%H:%M:%S) [$GPU] start server $MODEL"
  CUDA_VISIBLE_DEVICES=$GPU nohup $VPY/vllm serve $MODEL --port $PORT --dtype bfloat16 \
      --gpu-memory-utilization 0.85 --max-model-len 32768 --generation-config vllm \
      --disable-log-requests > $ROOT/logs/vllm_2c_$TAG.log 2>&1 &
  until curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1; do
    sleep 5
    pgrep -f "^$VPY/python.*vllm serve $MODEL --port $PORT" >/dev/null || { echo "server died: $MODEL"; continue 2; }
  done
  echo "$(date +%H:%M:%S) [$GPU] ready $MODEL"
  export TMC_BASE_URL=http://127.0.0.1:$PORT/v1 TMC_MODEL_OVERRIDE=$MODEL TMC_RUN_ID=2C-$TAG
  PIDS=""
  for r in 1 2 3; do
    for s in 0 1 2 3 4 5 6 7 8 9; do
      d=$OUT/M1/run$r/shard$s; mkdir -p $d
      ( cd $ROOT/external/MedAgents && TMC_NONCE_NS=2C:$TAG:M1x:run$r TMC_BUDGET_MAP=50:150,30:75,300:700,500:1000 python run.py --method syn_verif \
          --start_pos $((s*20)) --end_pos $((s*20+20)) --output_files_folder $d > $d/log.txt 2>&1 ) & PIDS="$PIDS $!"
    done
  done
  wait $PIDS
  echo "$(date +%H:%M:%S) [$GPU] jobs done $MODEL; M1=$(cat $OUT/M1/run*/shard*/chatgpt-syn_verif 2>/dev/null | wc -l)/600"
  pkill -f "^$VPY/python.*vllm serve $MODEL --port $PORT"
  while pgrep -f "^$VPY/python.*vllm serve $MODEL --port $PORT" >/dev/null; do sleep 3; done
  unset TMC_BASE_URL TMC_MODEL_OVERRIDE TMC_RUN_ID
  echo "$(date +%H:%M:%S) [$GPU] server down $MODEL"
done
echo "$(date +%H:%M:%S) [$GPU] ALL DONE"
