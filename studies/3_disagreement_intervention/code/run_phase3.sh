#!/bin/bash
# Phase III orchestration (PHASE3_FREEZE.md §5), Qwen2.5-14B on one GPU.
#   bash studies/3_disagreement_intervention/code/run_phase3.sh <gpu> <port> smoke|tokenprobe|pilot|final|reserve
GPU=$1; PORT=$2; MODE=$3; MODEL=Qwen/Qwen2.5-14B-Instruct
ROOT=$(cd "$(dirname "$0")/../../.." && pwd); VPY=${VLLM_BIN:?set VLLM_BIN to the bin/ of a vLLM environment}; E=$ROOT/studies/3_disagreement_intervention/code
export PYTHONPATH=$ROOT/common${PYTHONPATH:+:$PYTHONPATH}; export TMC_CACHE=$ROOT/cache/phase3 HF_HUB_OFFLINE=1
case $MODE in
  smoke)      SPLIT=dev;   OUT=$ROOT/studies/3_disagreement_intervention/results/dev/smoke;       PROBE_ARGS="--start 0 --n 20";;
  tokenprobe) SPLIT=dev;   OUT=$ROOT/studies/3_disagreement_intervention/results/dev/token_probe; PROBE_ARGS="--start 0 --n 100";;   # DEV in order; the first 20 triggered are used (§5 step 3)
  pilot)      SPLIT=pilot; OUT=$ROOT/studies/3_disagreement_intervention/results/pilot/main;      PROBE_ARGS="";;
  final)      SPLIT=final; OUT=$ROOT/studies/3_disagreement_intervention/results/final/main;      PROBE_ARGS="";;
  reserve)    SPLIT=reserve; OUT=$ROOT/studies/3_disagreement_intervention/results/reserve/main;  PROBE_ARGS="";;   # Amendment (2026-10-06)
  *) echo "mode?"; exit 1;;
esac
mkdir -p $OUT; T=$OUT/timing.tsv; stamp () { echo -e "$1\t$(date +%s)\t$(date -Is)" >> $T; }
L=P3:$MODE:$SPLIT
CUDA_VISIBLE_DEVICES=$GPU nohup $VPY/vllm serve $MODEL --port $PORT --dtype bfloat16 --gpu-memory-utilization ${VLLM_UTIL:-0.80} \
    --max-model-len 32768 --generation-config vllm --disable-log-requests > $ROOT/logs/vllm_p3_$MODE.log 2>&1 &
until curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1; do
  sleep 5; pgrep -f "^$VPY/python.*vllm serve $MODEL --port $PORT" >/dev/null || { echo "server died"; exit 1; }
done
echo "$(date +%H:%M:%S) ready"; stamp server_ready
export TMC_BASE_URL=http://127.0.0.1:$PORT/v1 TMC_MODEL_OVERRIDE=$MODEL TMC_RUN_ID=P3-$MODE
cd $E
python run_probe.py --split $SPLIT $PROBE_ARGS --label $L:probe --out $OUT/probe.jsonl; stamp probe_done
if [ $MODE = tokenprobe ]; then
  python run_tdc.py --split $SPLIT --probe $OUT/probe.jsonl --max_triggered 20 --label $L:tdc --out $OUT/tdc.jsonl & P1=$!
  python run_hetero_more.py --split $SPLIT --probe $OUT/probe.jsonl --k 7 --max_triggered 20 --label $L:het --out $OUT/het.jsonl & P2=$!
  wait $P1 $P2                      # explicit PIDs: a bare `wait` would also wait for the vLLM server
else
  if [ $MODE = smoke ]; then KA=3; KH=7; else
    KA=$(python -c "import json;print(json.load(open('$ROOT/studies/3_disagreement_intervention/results/K_FROZEN.json'))['k_add'])")
    KH=$(python -c "import json;print(json.load(open('$ROOT/studies/3_disagreement_intervention/results/K_FROZEN.json'))['k_het'])"); fi
  python make_m1_input.py --split $SPLIT --probe $OUT/probe.jsonl --out_dir $OUT/m1_input
  NM=$(python -c "import json;print(len(json.load(open('$OUT/m1_input/qids.json'))))")
  PIDS=""
  python run_tdc.py --split $SPLIT --probe $OUT/probe.jsonl --label $L:tdc --out $OUT/tdc.jsonl & PIDS="$PIDS $!"
  python run_sample_more.py --split $SPLIT --probe $OUT/probe.jsonl --k $KA --label $L:sm --out $OUT/sample_more.jsonl & PIDS="$PIDS $!"
  python run_hetero_more.py --split $SPLIT --probe $OUT/probe.jsonl --k $KH --label $L:het --out $OUT/het.jsonl & PIDS="$PIDS $!"
  ln -sfn $OUT/m1_input $ROOT/data/splits/MedMCQA_p3m1_$MODE
  for s in $(seq 0 $(( (NM + 19) / 20 - 1 ))); do
    d=$OUT/M1/shard$s; mkdir -p $d
    ( cd ${MEDAGENTS_DIR:?set MEDAGENTS_DIR to a patched MedAgents checkout} && TMC_NONCE_NS=$L:M1 COST_LOG=$d/cost.jsonl python run.py --method syn_verif \
        --dataset_name MedMCQA --dataset_dir ./datasets/MedMCQA_p3m1_$MODE/ --start_pos $((s*20)) \
        --end_pos $(( s*20+20 < NM ? s*20+20 : NM )) --output_files_folder $d > $d/log.txt 2>&1 ) & PIDS="$PIDS $!"
  done
  wait $PIDS
fi
stamp jobs_done; echo "$(date +%H:%M:%S) jobs done"
pkill -f "^$VPY/python.*vllm serve $MODEL --port $PORT"
while pgrep -f "^$VPY/python.*vllm serve $MODEL --port $PORT" >/dev/null; do sleep 3; done
echo "$(date +%H:%M:%S) server down"
