#!/bin/bash
# Phase II empirical-oracle pilot on one local model (PHASE2_FREEZE.md §3.3).
#   bash studies/2_complementarity_router/code/run_pilot.sh <gpu> <port> <model> <probe|pilot>
#   probe = first 20 PILOT questions, timing only (no accuracy is computed); pilot = all 300.
# Per question: S x3 (T=0), SC 6 runs x 20 samples (T=0.7; pools A = runs 1-2, E = run 3, P = runs 4-6),
# M1 x3 (released). P_RUNS=5 cuts the placebo pool to 40 samples (pre-specified 48 GPU-hour branch).
GPU=$1; PORT=$2; MODEL=$3; MODE=$4
ROOT=$(cd "$(dirname "$0")/../../.." && pwd); VPY=${VLLM_BIN:?set VLLM_BIN to the bin/ of a vLLM environment}
export PYTHONPATH=$ROOT/common${PYTHONPATH:+:$PYTHONPATH}; 
export TMC_CACHE=$ROOT/cache/phase2                     # Phase II never shares Part I caches
TAG=$(basename $MODEL); DATA=$ROOT/data/splits/MedQA_p2pilot/test.jsonl
case $MODE in probe) N=20;; pilot) N=300;; *) echo "mode?"; exit 1;; esac
OUT=$ROOT/studies/2_complementarity_router/results/$MODE/$TAG; mkdir -p $OUT/S $OUT/SC; T=$OUT/timing.tsv
SC_RUNS=${P_RUNS:-6}
stamp () { echo -e "$1\t$(date +%s)\t$(date -Is)" >> $T; }
echo "$(date +%H:%M:%S) [$GPU] start server $MODEL ($MODE, N=$N)"
CUDA_VISIBLE_DEVICES=$GPU nohup $VPY/vllm serve $MODEL --port $PORT --dtype bfloat16 --gpu-memory-utilization ${VLLM_UTIL:-0.80} \
    --max-model-len 32768 --generation-config vllm --disable-log-requests > $ROOT/logs/vllm_p2_${MODE}_$TAG.log 2>&1 &
until curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1; do
  sleep 5; pgrep -f "^$VPY/python.*vllm serve $MODEL --port $PORT" >/dev/null || { echo "server died"; exit 1; }
done
echo "$(date +%H:%M:%S) [$GPU] ready $MODEL"; stamp server_ready
export TMC_BASE_URL=http://127.0.0.1:$PORT/v1 TMC_MODEL_OVERRIDE=$MODEL TMC_RUN_ID=P2-$MODE-$TAG
PIDS=""
for r in 1 2 3; do
  python $ROOT/experiments/p0_extraction.py --data $DATA --n $N --workers 16 --run_label P2:$MODE:$TAG:S:run$r \
      --out $OUT/S/run$r.jsonl > $OUT/S_run$r.log 2>&1 & PIDS="$PIDS $!"
done
for r in $(seq 1 $SC_RUNS); do
  python $ROOT/experiments/compute_matched.py --method sc-20 --data $DATA --n $N --seed $((100 + r)) --workers 16 \
      --run_label P2:$MODE:$TAG:sc-20:run$r --out $OUT/SC/run$r.jsonl > $OUT/SC_run$r.log 2>&1 & PIDS="$PIDS $!"
done
for r in 1 2 3; do
  for s in $(seq 0 $(( (N + 19) / 20 - 1 ))); do
    d=$OUT/M1/run$r/shard$s; mkdir -p $d
    ( cd ${MEDAGENTS_DIR:?set MEDAGENTS_DIR to a patched MedAgents checkout} && TMC_NONCE_NS=P2:$MODE:$TAG:M1:run$r COST_LOG=$d/cost.jsonl python run.py --method syn_verif \
        --dataset_name MedQA --dataset_dir ./datasets/MedQA_p2pilot/ --start_pos $((s*20)) --end_pos $(( s*20+20 < N ? s*20+20 : N )) \
        --output_files_folder $d > $d/log.txt 2>&1 ) & PIDS="$PIDS $!"
  done
done
wait $PIDS
stamp jobs_done
echo "$(date +%H:%M:%S) [$GPU] jobs done $MODEL"
pkill -f "^$VPY/python.*vllm serve $MODEL --port $PORT"
while pgrep -f "^$VPY/python.*vllm serve $MODEL --port $PORT" >/dev/null; do sleep 3; done
echo "$(date +%H:%M:%S) [$GPU] server down $MODEL"
