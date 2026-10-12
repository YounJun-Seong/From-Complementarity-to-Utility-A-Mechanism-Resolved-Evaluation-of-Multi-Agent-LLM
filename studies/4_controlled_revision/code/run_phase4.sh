#!/bin/bash
# Phase IV orchestration (PHASE4_FREEZE.md §5), Qwen2.5-14B, one GPU.   bash run_phase4.sh <gpu> <port> smoke|full
GPU=$1; PORT=$2; MODE=$3; MODEL=Qwen/Qwen2.5-14B-Instruct
ROOT=$(cd "$(dirname "$0")/../../.." && pwd); VPY=${VLLM_BIN:?set VLLM_BIN to the bin/ of a vLLM environment}
export PYTHONPATH=$ROOT/common${PYTHONPATH:+:$PYTHONPATH}; export TMC_CACHE=$ROOT/cache/phase4 HF_HUB_OFFLINE=1
case $MODE in
  smoke) SPLIT=dev;   OUT=$ROOT/studies/4_controlled_revision/results/dev_smoke; PROBE_ARGS="--start 100 --n 10";;
  full)  SPLIT=final; OUT=$ROOT/studies/4_controlled_revision/results/main;      PROBE_ARGS="";;
  *) echo "mode?"; exit 1;;
esac
mkdir -p $OUT; T=$OUT/timing.tsv; stamp () { echo -e "$1\t$(date +%s)\t$(date -Is)" >> $T; }
L=P4:$MODE:$SPLIT
CUDA_VISIBLE_DEVICES=$GPU nohup $VPY/vllm serve $MODEL --port $PORT --dtype bfloat16 --gpu-memory-utilization ${VLLM_UTIL:-0.80} \
    --max-model-len 32768 --generation-config vllm --disable-log-requests > $ROOT/logs/vllm_p4_$MODE.log 2>&1 &
SRV=$!
until curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1; do
  sleep 5; kill -0 $SRV 2>/dev/null || { echo "server died"; exit 1; }
done
echo "$(date +%H:%M:%S) ready"; stamp server_ready
export TMC_BASE_URL=http://127.0.0.1:$PORT/v1 TMC_MODEL_OVERRIDE=$MODEL TMC_RUN_ID=P4-$MODE
cd $ROOT/studies/3_disagreement_intervention/code
python run_probe.py --split $SPLIT $PROBE_ARGS --label $L:probe --out $OUT/probe.jsonl; stamp probe_done
cd $ROOT/studies/4_controlled_revision/code
python run_revisions.py --split $SPLIT --probe $OUT/probe.jsonl --label $L:rev --out $OUT/revisions.jsonl; stamp jobs_done
echo "$(date +%H:%M:%S) jobs done"
kill $SRV; while kill -0 $SRV 2>/dev/null; do sleep 3; done
echo "$(date +%H:%M:%S) server down"
