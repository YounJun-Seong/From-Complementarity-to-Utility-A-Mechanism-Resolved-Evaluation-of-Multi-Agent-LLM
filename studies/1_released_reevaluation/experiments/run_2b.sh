#!/bin/bash
# Layer 2B driver: per model on one GPU, vLLM up -> B1/B3 x 3 runs on MedQA 0-199 -> down.
#   bash experiments/run_2b.sh <gpu> <port> <model> [<model> ...]
GPU=$1; PORT=$2; shift 2
ROOT=$(cd "$(dirname "$0")/.." && pwd); VPY=<VLLM_ENV>/bin
source $ROOT/env.sh
for MODEL in "$@"; do
  TAG=$(basename $MODEL); OUT=$ROOT/runs/layer2/2B/$TAG; mkdir -p $OUT
  echo "$(date +%H:%M:%S) [$GPU] start server $MODEL"
  CUDA_VISIBLE_DEVICES=$GPU nohup $VPY/vllm serve $MODEL --port $PORT --dtype bfloat16 --gpu-memory-utilization 0.85 \
      --max-model-len 32768 --generation-config vllm --disable-log-requests > $ROOT/logs/vllm_2b_$TAG.log 2>&1 &
  until curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1; do sleep 5; done
  echo "$(date +%H:%M:%S) [$GPU] ready $MODEL"
  export TMC_BASE_URL=http://127.0.0.1:$PORT/v1 TMC_MODEL_OVERRIDE=$MODEL TMC_RUN_ID=2B-$TAG
  PIDS=""
  for r in 1 2 3; do
    python $ROOT/experiments/p0_sweep.py --roles empty --budgets 300,700 --n 200 --workers 32 \
        --run_label 2B:$TAG:run$r --out $OUT/run$r > $OUT/run$r.log 2>&1 & PIDS="$PIDS $!"
  done
  wait $PIDS
  echo "$(date +%H:%M:%S) [$GPU] jobs done $MODEL; files $(ls $OUT/run*/*.jsonl | wc -l)/6"
  pkill -f "^$VPY/python.*vllm serve $MODEL --port $PORT"
  while pgrep -f "^$VPY/python.*vllm serve $MODEL --port $PORT" >/dev/null; do sleep 3; done
  unset TMC_BASE_URL TMC_MODEL_OVERRIDE TMC_RUN_ID
  echo "$(date +%H:%M:%S) [$GPU] server down $MODEL"
done
echo "$(date +%H:%M:%S) [$GPU] ALL DONE"
