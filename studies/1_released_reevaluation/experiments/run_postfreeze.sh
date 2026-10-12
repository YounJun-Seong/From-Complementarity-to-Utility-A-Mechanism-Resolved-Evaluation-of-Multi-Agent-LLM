#!/bin/bash
# Post-freeze R1 (random subsets), R2 (MedMCQA), R5 (Llama SC-7) on one local model.
#   bash experiments/run_postfreeze.sh <gpu> <port> <model>
GPU=$1; PORT=$2; MODEL=$3
ROOT=$(cd "$(dirname "$0")/.." && pwd); VPY=<VLLM_ENV>/bin
source $ROOT/env.sh; TAG=$(basename $MODEL); OUT=$ROOT/runs/postfreeze; DS=$ROOT/external/MedAgents/datasets
echo "$(date +%H:%M:%S) [$GPU] start server $MODEL"
CUDA_VISIBLE_DEVICES=$GPU nohup $VPY/vllm serve $MODEL --port $PORT --dtype bfloat16 --gpu-memory-utilization ${VLLM_UTIL:-0.80} \
    --max-model-len 32768 --generation-config vllm --disable-log-requests > $ROOT/logs/vllm_pf_$TAG.log 2>&1 &
until curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1; do
  sleep 5; pgrep -f "^$VPY/python.*vllm serve $MODEL --port $PORT" >/dev/null || { echo "server died"; exit 1; }
done
echo "$(date +%H:%M:%S) [$GPU] ready $MODEL"
export TMC_BASE_URL=http://127.0.0.1:$PORT/v1 TMC_MODEL_OVERRIDE=$MODEL TMC_RUN_ID=PF-$TAG
PIDS=""
m1_shards () {  # $1 dataset_name $2 dataset dir name $3 n $4 out dir $5 nonce
  for s in $(seq 0 $(( ($3 + 19) / 20 - 1 ))); do
    d=$4/shard$s; mkdir -p $d
    ( cd $ROOT/external/MedAgents && TMC_NONCE_NS=$5 COST_LOG=$d/cost.jsonl python run.py --method syn_verif \
        --dataset_name $1 --dataset_dir ./datasets/$2/ --start_pos $((s*20)) --end_pos $(( s*20+20 < $3 ? s*20+20 : $3 )) \
        --output_files_folder $d > $d/log.txt 2>&1 ) & PIDS="$PIDS $!"
  done
}
for sub in A B C; do                                                   # R1 (+ SC-5 amendment)
  mkdir -p $OUT/R1/$TAG
  if [ -z "$ONLY_SC5" ]; then
  python $ROOT/experiments/p0_extraction.py --data $DS/MedQA_rand$sub/test.jsonl --n 200 --workers 16 \
      --run_label PF:$TAG:R1:$sub:S --out $OUT/R1/$TAG/S_$sub.jsonl > $OUT/R1/$TAG/S_$sub.log 2>&1 & PIDS="$PIDS $!"
  fi
  python $ROOT/experiments/compute_matched.py --method sc-5 --data $DS/MedQA_rand$sub/test.jsonl --n 200 --seed 1 --workers 16 \
      --run_label PF:$TAG:R1:$sub:sc-5 --out $OUT/R1/$TAG/SC5_$sub.jsonl > $OUT/R1/$TAG/SC5_$sub.log 2>&1 & PIDS="$PIDS $!"
  [ -n "$ONLY_SC5" ] && continue
  m1_shards MedQA MedQA_rand$sub 200 $OUT/R1/$TAG/M1_$sub PF:$TAG:R1:$sub:M1
done
[ -n "$ONLY_SC5" ] && { wait $PIDS; echo "$(date +%H:%M:%S) [$GPU] jobs done $MODEL"; pkill -f "^$VPY/python.*vllm serve $MODEL --port $PORT"; exit 0; }
mkdir -p $OUT/R2/$TAG                                                  # R2
python $ROOT/experiments/p0_extraction.py --data $DS/MedMCQA_r500/test.jsonl --n 500 --workers 16 \
    --run_label PF:$TAG:R2:S --out $OUT/R2/$TAG/S.jsonl > $OUT/R2/$TAG/S.log 2>&1 & PIDS="$PIDS $!"
python $ROOT/experiments/compute_matched.py --method sc-5 --data $DS/MedMCQA_r500/test.jsonl --n 500 --seed 1 --workers 24 \
    --run_label PF:$TAG:R2:sc-5 --out $OUT/R2/$TAG/SC5.jsonl > $OUT/R2/$TAG/SC5.log 2>&1 & PIDS="$PIDS $!"
m1_shards MedMCQA MedMCQA_r500 500 $OUT/R2/$TAG/M1 PF:$TAG:R2:M1
if [[ $TAG == Llama* ]]; then                                          # R5
  for r in 1 2 3; do
    python $ROOT/experiments/compute_matched.py --method sc-7 --n 200 --seed $r --workers 24 \
        --run_label 2D:$TAG:sc-7:run$r --out $ROOT/runs/layer2/2D/$TAG/sc-7/run$r.jsonl > $ROOT/runs/layer2/2D/$TAG/sc-7-run$r.log 2>&1 & PIDS="$PIDS $!"
  done
fi
wait $PIDS
echo "$(date +%H:%M:%S) [$GPU] jobs done $MODEL"
pkill -f "^$VPY/python.*vllm serve $MODEL --port $PORT"
while pgrep -f "^$VPY/python.*vllm serve $MODEL --port $PORT" >/dev/null; do sleep 3; done
echo "$(date +%H:%M:%S) [$GPU] server down $MODEL"
