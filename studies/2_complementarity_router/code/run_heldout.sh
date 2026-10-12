#!/bin/bash
# Phase II HELD-OUT (Amendment 2): one execution of every arm on the 339 HELD-OUT questions, Qwen2.5-14B only.
#   bash studies/2_complementarity_router/code/run_heldout.sh <gpu> <port>
# Arms: S (1 run, T=0) | SC-22 (one request n=22, T=0.7, E3 per sample; probe = its first 4 samples)
#       | M1 (released, 1 run) | HET-22 (22 samples round-robin over M1-style recruited roles).
# Ours-Collab / Ours-Sampling / Ours-Hetero are built offline from these outputs (no separate LLM runs).
GPU=$1; PORT=$2; MODEL=Qwen/Qwen2.5-14B-Instruct
ROOT=$(cd "$(dirname "$0")/../../.." && pwd); VPY=${VLLM_BIN:?set VLLM_BIN to the bin/ of a vLLM environment}
export PYTHONPATH=$ROOT/common${PYTHONPATH:+:$PYTHONPATH}; export TMC_CACHE=$ROOT/cache/phase2 HF_HUB_OFFLINE=1
TAG=$(basename $MODEL); DATA=$ROOT/data/splits/MedQA_p2heldout/test.jsonl; N=339
OUT=$ROOT/studies/2_complementarity_router/results/heldout/$TAG; mkdir -p $OUT/S $OUT/SC $OUT/HET; T=$OUT/timing.tsv
stamp () { echo -e "$1\t$(date +%s)\t$(date -Is)" >> $T; }
CUDA_VISIBLE_DEVICES=$GPU nohup $VPY/vllm serve $MODEL --port $PORT --dtype bfloat16 --gpu-memory-utilization ${VLLM_UTIL:-0.80} \
    --max-model-len 32768 --generation-config vllm --disable-log-requests > $ROOT/logs/vllm_p2_heldout_$TAG.log 2>&1 &
until curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1; do
  sleep 5; pgrep -f "^$VPY/python.*vllm serve $MODEL --port $PORT" >/dev/null || { echo "server died"; exit 1; }
done
echo "$(date +%H:%M:%S) [$GPU] ready $MODEL"; stamp server_ready
export TMC_BASE_URL=http://127.0.0.1:$PORT/v1 TMC_MODEL_OVERRIDE=$MODEL TMC_RUN_ID=P2-heldout-$TAG
PIDS=""
python $ROOT/experiments/p0_extraction.py --data $DATA --n $N --workers 16 --run_label P2:heldout:$TAG:S:run1 \
    --out $OUT/S/run1.jsonl > $OUT/S_run1.log 2>&1 & PIDS="$PIDS $!"
python $ROOT/experiments/compute_matched.py --method sc-22 --data $DATA --n $N --seed 301 --workers 16 \
    --run_label P2:heldout:$TAG:sc-22:run1 --out $OUT/SC/run1.jsonl > $OUT/SC_run1.log 2>&1 & PIDS="$PIDS $!"
python $ROOT/studies/2_complementarity_router/code/hetero_sc.py --data $DATA --n $N --total 22 --seed 401 \
    --run_label P2:het:heldout:$TAG:run1 --out $OUT/HET/run1.jsonl > $OUT/HET_run1.log 2>&1 & PIDS="$PIDS $!"
for s in $(seq 0 $(( (N + 19) / 20 - 1 ))); do
  d=$OUT/M1/run1/shard$s; mkdir -p $d
  ( cd ${MEDAGENTS_DIR:?set MEDAGENTS_DIR to a patched MedAgents checkout} && TMC_NONCE_NS=P2:heldout:$TAG:M1:run1 COST_LOG=$d/cost.jsonl python run.py --method syn_verif \
      --dataset_name MedQA --dataset_dir ./datasets/MedQA_p2heldout/ --start_pos $((s*20)) --end_pos $(( s*20+20 < N ? s*20+20 : N )) \
      --output_files_folder $d > $d/log.txt 2>&1 ) & PIDS="$PIDS $!"
done
wait $PIDS
stamp jobs_done; echo "$(date +%H:%M:%S) [$GPU] jobs done $MODEL"
pkill -f "^$VPY/python.*vllm serve $MODEL --port $PORT"
while pgrep -f "^$VPY/python.*vllm serve $MODEL --port $PORT" >/dev/null; do sleep 3; done
echo "$(date +%H:%M:%S) [$GPU] server down $MODEL"
