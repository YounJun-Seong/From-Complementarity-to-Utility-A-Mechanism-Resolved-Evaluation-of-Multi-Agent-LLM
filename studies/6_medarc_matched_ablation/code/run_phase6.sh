#!/bin/bash
# Phase VI orchestration (PHASE6_FREEZE.md §5). Two GPUs: GPU0 = round-0 pre-pass, A, then C; GPU1 = B, B_strong, SC.
#   bash run_phase6.sh smoke|full
MODE=$1; MODEL=Qwen/Qwen2.5-14B-Instruct
ROOT=$(cd "$(dirname "$0")/../../.." && pwd); VPY=${VLLM_BIN:?set VLLM_BIN to the bin/ of a vLLM environment}; E=$ROOT/studies/6_medarc_matched_ablation/code
export PYTHONPATH=$ROOT/common${PYTHONPATH:+:$PYTHONPATH}; export HF_HUB_OFFLINE=1 P5_CACHE=$ROOT/cache/phase6 TMC_MODEL_OVERRIDE=$MODEL   # project-private weight cache (shared cache was wiped 10-08)
PY=${PYTHON:-python}
if [ $MODE = smoke ]; then O=$ROOT/studies/6_medarc_matched_ablation/results/smoke; DATA="--data $ROOT/studies/5_medarc_transfer/results/data/probe_inputs.jsonl --n 5"; else O=$ROOT/studies/6_medarc_matched_ablation/results/main; DATA=""; fi
mkdir -p $O
serve () {  # $1 gpu $2 port
  CUDA_VISIBLE_DEVICES=$1 nohup $VPY/vllm serve $MODEL --port $2 --dtype bfloat16 --gpu-memory-utilization ${VLLM_UTIL:-0.70} \
      --max-model-len 32768 --generation-config vllm --disable-log-requests > $ROOT/logs/vllm_p6_${MODE}_$1.log 2>&1 &
  echo $!
}
S0=$(serve 0 8101); S1=$(serve 1 8102)
for p in 8101 8102; do
  until curl -sf http://127.0.0.1:$p/v1/models >/dev/null 2>&1; do
    sleep 5
    if ! kill -0 $S0 2>/dev/null || ! kill -0 $S1 2>/dev/null; then   # a server died (e.g. OOM): stop, keep logs
      echo "$(date +%H:%M:%S) server died"; kill $S0 $S1 2>/dev/null; exit 3; fi
  done
done
echo "$(date +%H:%M:%S) servers ready"
cd $E
( export TMC_BASE_URL=http://127.0.0.1:8101/v1
  for r in 1 2 3; do $PY run_ablation.py --mode round0 --run $r $DATA --out $O/round0_run$r.jsonl; done
  touch $O/.round0_done
  P=""; for r in 1 2 3; do $PY run_ablation.py --mode A --run $r $DATA --out $O/A_run$r.jsonl & P="$P $!"; done; wait $P
  P=""; for r in 1 2 3; do $PY run_ablation.py --mode C --run $r $DATA --a_file $O/A_run$r.jsonl --out $O/C_run$r.jsonl & P="$P $!"; done; wait $P
) > $O/gpu0.log 2>&1 &
G0=$!
( export TMC_BASE_URL=http://127.0.0.1:8102/v1
  until [ -f $O/.round0_done ]; do sleep 10; done
  P=""
  for r in 1 2 3; do $PY run_ablation.py --mode B --run $r $DATA --out $O/B_run$r.jsonl & P="$P $!"; done
  for r in 1 2 3; do $PY run_ablation.py --mode cot --run $r $DATA --out $O/cot_run$r.jsonl & P="$P $!"; done
  for r in 1 2 3 4 5 6; do $PY run_ablation.py --mode sc --run $r --k 25 $DATA --out $O/sc_run$r.jsonl & P="$P $!"; done
  wait $P
) > $O/gpu1.log 2>&1 &
G1=$!
wait $G0 $G1
echo "$(date +%H:%M:%S) jobs done"
kill $S0 $S1; while kill -0 $S0 2>/dev/null || kill -0 $S1 2>/dev/null; do sleep 3; done
echo "$(date +%H:%M:%S) servers down"
