#!/bin/bash
# Amendment 1: HET runs on one local model.  bash studies/2_complementarity_router/code/run_het.sh <gpu> <port> <model> <pilot|heldout>
GPU=$1; PORT=$2; MODEL=$3; SPLIT=$4
ROOT=$(cd "$(dirname "$0")/../../.." && pwd); VPY=${VLLM_BIN:?set VLLM_BIN to the bin/ of a vLLM environment}
export PYTHONPATH=$ROOT/common${PYTHONPATH:+:$PYTHONPATH}; export TMC_CACHE=$ROOT/cache/phase2 HF_HUB_OFFLINE=1
TAG=$(basename $MODEL)
case $SPLIT in pilot) DATA=$ROOT/data/splits/MedQA_p2pilot/test.jsonl; N=300; RUNS="1 2 3";;
               *) echo "split?"; exit 1;; esac
N=${N_OVERRIDE:-$N}; RUNS=${RUNS_OVERRIDE:-$RUNS}
OUT=${OUT_OVERRIDE:-$ROOT/studies/2_complementarity_router/results/$SPLIT/$TAG/HET}; mkdir -p $OUT
CUDA_VISIBLE_DEVICES=$GPU nohup $VPY/vllm serve $MODEL --port $PORT --dtype bfloat16 --gpu-memory-utilization ${VLLM_UTIL:-0.80} \
    --max-model-len 32768 --generation-config vllm --disable-log-requests > $ROOT/logs/vllm_p2_het_${SPLIT}_$TAG.log 2>&1 &
until curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1; do
  sleep 5; pgrep -f "^$VPY/python.*vllm serve $MODEL --port $PORT" >/dev/null || { echo "server died"; exit 1; }
done
echo "$(date +%H:%M:%S) [$GPU] ready $MODEL"
export TMC_BASE_URL=http://127.0.0.1:$PORT/v1 TMC_MODEL_OVERRIDE=$MODEL TMC_RUN_ID=P2-het-$SPLIT-$TAG
PIDS=""
for r in $RUNS; do
  python $ROOT/studies/2_complementarity_router/code/hetero_sc.py --data $DATA --n $N --total 21 --seed $((200 + r)) \
      --run_label P2:het:$SPLIT:$TAG:run$r --out $OUT/run$r.jsonl > $OUT/run$r.log 2>&1 & PIDS="$PIDS $!"
done
wait $PIDS
echo "$(date +%H:%M:%S) [$GPU] jobs done $MODEL"
pkill -f "^$VPY/python.*vllm serve $MODEL --port $PORT"
while pgrep -f "^$VPY/python.*vllm serve $MODEL --port $PORT" >/dev/null; do sleep 3; done
echo "$(date +%H:%M:%S) [$GPU] server down $MODEL"
