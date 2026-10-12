#!/bin/bash
# R1 amendment: SC-5 on random subsets A/B/C, using the already-running post-freeze server of a model.
PORT=$1; MODEL=$2
ROOT=$(cd "$(dirname "$0")/.." && pwd); TAG=$(basename $MODEL); DS=$ROOT/external/MedAgents/datasets
source $ROOT/env.sh
until curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1; do sleep 10; done
export TMC_BASE_URL=http://127.0.0.1:$PORT/v1 TMC_MODEL_OVERRIDE=$MODEL TMC_RUN_ID=PF-$TAG
PIDS=""
for sub in A B C; do
  python $ROOT/experiments/compute_matched.py --method sc-5 --data $DS/MedQA_rand$sub/test.jsonl --n 200 --seed 1 --workers 16 \
      --run_label PF:$TAG:R1:$sub:sc-5 --out $ROOT/runs/postfreeze/R1/$TAG/SC5_$sub.jsonl > $ROOT/runs/postfreeze/R1/$TAG/SC5_$sub.log 2>&1 & PIDS="$PIDS $!"
done
wait $PIDS; echo "$(date +%H:%M:%S) SC-5 done $TAG"
