#!/bin/bash
# Phase V orchestration (PHASE5_FREEZE.md §7).   bash run_phase5.sh <gpu> <port> smoke|tokenprobe|full
GPU=$1; PORT=$2; MODE=$3; MODEL=Qwen/Qwen2.5-14B-Instruct
ROOT=$(cd "$(dirname "$0")/../../.." && pwd); VPY=${VLLM_BIN:?set VLLM_BIN to the bin/ of a vLLM environment}; E=$ROOT/studies/5_medarc_transfer/code
export PYTHONPATH=$ROOT/common${PYTHONPATH:+:$PYTHONPATH}; export HF_HUB_OFFLINE=1
CUDA_VISIBLE_DEVICES=$GPU nohup $VPY/vllm serve $MODEL --port $PORT --dtype bfloat16 --gpu-memory-utilization ${VLLM_UTIL:-0.80} \
    --max-model-len 32768 --generation-config vllm --disable-log-requests > $ROOT/logs/vllm_p5_$MODE.log 2>&1 &
SRV=$!
until curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1; do sleep 5; kill -0 $SRV 2>/dev/null || { echo "server died"; exit 1; }; done
echo "$(date +%H:%M:%S) ready"
export TMC_BASE_URL=http://127.0.0.1:$PORT/v1 TMC_MODEL_OVERRIDE=$MODEL
cd $E; PY=${PYTHON:-python}; PIDS=""
case $MODE in
  smoke) O=$ROOT/studies/5_medarc_transfer/results/smoke
    for arm in medarc zeroshot cot; do $PY run_arms.py --arm $arm --split probe --n 5 --run 1 --out $O/${arm}_run1.jsonl & PIDS="$PIDS $!"; done
    $PY run_arms.py --arm sc --split probe --n 5 --run 1 --k 3 --out $O/sc_run1.jsonl & PIDS="$PIDS $!";;
  tokenprobe) O=$ROOT/studies/5_medarc_transfer/results/token_probe
    $PY run_arms.py --arm medarc --split probe --run 1 --out $O/medarc_run1.jsonl & PIDS="$PIDS $!"
    $PY run_arms.py --arm sc --split probe --run 1 --k 1 --out $O/sc1_run1.jsonl & PIDS="$PIDS $!";;
  full) O=$ROOT/studies/5_medarc_transfer/results/main; K=$($PY -c "import json;print(json.load(open('$ROOT/studies/5_medarc_transfer/results/K_FROZEN.json'))['k_star'])")
    for r in 1 2 3; do
      for arm in medarc zeroshot cot; do $PY run_arms.py --arm $arm --split eval --run $r --out $O/${arm}_run$r.jsonl & PIDS="$PIDS $!"; done
    done
    for r in 1 2 3 4 5 6; do $PY run_arms.py --arm sc --split eval --run $r --k $K --out $O/sc_run$r.jsonl & PIDS="$PIDS $!"; done;;
esac
wait $PIDS
echo "$(date +%H:%M:%S) jobs done"
kill $SRV; while kill -0 $SRV 2>/dev/null; do sleep 3; done
echo "$(date +%H:%M:%S) server down"
