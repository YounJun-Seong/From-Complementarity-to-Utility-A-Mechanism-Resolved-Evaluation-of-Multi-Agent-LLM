#!/bin/bash
# Layer 2 extras driver: bash experiments/run_extra.sh <gpu> <port> <model> <tasks: comma list of 2A,2D,2E>
GPU=$1; PORT=$2; MODEL=$3; TASKS=$4
ROOT=$(cd "$(dirname "$0")/.." && pwd); VPY=<VLLM_ENV>/bin
source $ROOT/env.sh; TAG=$(basename $MODEL)
if [[ $TASKS == *2A* ]]; then bash $ROOT/experiments/run_2a.sh $GPU $PORT $MODEL; fi
[[ $TASKS == *2D* || $TASKS == *2E* ]] || exit 0
echo "$(date +%H:%M:%S) [$GPU] start server $MODEL ($TASKS)"
CUDA_VISIBLE_DEVICES=$GPU nohup $VPY/vllm serve $MODEL --port $PORT --dtype bfloat16 --gpu-memory-utilization ${VLLM_UTIL:-0.85} \
    --max-model-len ${VLLM_MAXLEN:-32768} --generation-config vllm --disable-log-requests > $ROOT/logs/vllm_extra_$TAG.log 2>&1 &
until curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1; do
  sleep 5
  pgrep -f "^$VPY/python.*vllm serve $MODEL --port $PORT" >/dev/null || { echo "$(date +%H:%M:%S) [$GPU] server died: $MODEL"; exit 1; }
done
echo "$(date +%H:%M:%S) [$GPU] ready $MODEL"
export TMC_BASE_URL=http://127.0.0.1:$PORT/v1 TMC_MODEL_OVERRIDE=$MODEL TMC_RUN_ID=extra-$TAG
PIDS=""
if [[ $TASKS == *2D* ]]; then
  OUT=$ROOT/runs/layer2/2D/$TAG; mkdir -p $OUT
  for m in sc-5 sc-10 sc-20 reflect; do for r in 1 2 3; do
    python $ROOT/experiments/compute_matched.py --method $m --n 200 --seed $r --workers 24 \
        --run_label 2D:$TAG:$m:run$r --out $OUT/$m/run$r.jsonl > $OUT/$m-run$r.log 2>&1 & PIDS="$PIDS $!"
  done; done
fi
if [[ $TASKS == *2E* ]]; then
  OUT=$ROOT/runs/layer2/2E/$TAG; mkdir -p $OUT
  for p in 1 2 3; do
    python $ROOT/experiments/p0_extraction.py --n 200 --workers 16 \
        --data $ROOT/external/MedAgents/datasets/MedQA_perm$p/test.jsonl \
        --run_label 2E:$TAG:S:perm$p --out $OUT/S_perm$p.jsonl > $OUT/S_perm$p.log 2>&1 & PIDS="$PIDS $!"
    for s in 0 1 2 3 4 5 6 7 8 9; do
      d=$OUT/M1_perm$p/shard$s; mkdir -p $d
      ( cd $ROOT/external/MedAgents && TMC_NONCE_NS=2E:$TAG:M1:perm$p python run.py --method syn_verif \
          --dataset_dir ./datasets/MedQA_perm$p/ --start_pos $((s*20)) --end_pos $((s*20+20)) \
          --output_files_folder $d > $d/log.txt 2>&1 ) & PIDS="$PIDS $!"
    done
  done
fi
wait $PIDS
echo "$(date +%H:%M:%S) [$GPU] jobs done $MODEL ($TASKS)"
pkill -f "^$VPY/python.*vllm serve $MODEL --port $PORT"
while pgrep -f "^$VPY/python.*vllm serve $MODEL --port $PORT" >/dev/null; do sleep 3; done
echo "$(date +%H:%M:%S) [$GPU] server down $MODEL"
