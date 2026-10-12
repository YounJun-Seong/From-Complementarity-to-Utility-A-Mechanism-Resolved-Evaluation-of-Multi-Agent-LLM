#!/bin/bash
# Fill pass for 2A M1 shards that stopped early (pre-fix harness): resume each shard from its
# first missing question with the fixed run.py.   bash experiments/fill_2a.sh <gpu> <port> <model>
GPU=$1; PORT=$2; MODEL=$3
ROOT=$(cd "$(dirname "$0")/.." && pwd); VPY=<VLLM_ENV>/bin
source $ROOT/env.sh; TAG=$(basename $MODEL); OUT=$ROOT/runs/layer2/2A/$TAG
todo=""; for d in $OUT/M1/run*/shard*; do n=$(wc -l < $d/chatgpt-syn_verif 2>/dev/null || echo 0); [ "$n" -lt 20 ] && todo="$todo $d:$n"; done
[ -z "$todo" ] && { echo "$TAG: nothing to fill"; exit 0; }
echo "$(date +%H:%M:%S) fill $TAG:$todo"
CUDA_VISIBLE_DEVICES=$GPU nohup $VPY/vllm serve $MODEL --port $PORT --dtype bfloat16 --gpu-memory-utilization 0.85 \
    --max-model-len 32768 --generation-config vllm --disable-log-requests > $ROOT/logs/vllm_2a_fill_$TAG.log 2>&1 &
until curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1; do sleep 5; done
export TMC_BASE_URL=http://127.0.0.1:$PORT/v1 TMC_MODEL_OVERRIDE=$MODEL TMC_RUN_ID=2A-$TAG
PIDS=""
for item in $todo; do
  d=${item%:*}; n=${item##*:}; r=$(basename $(dirname $d)); s=$(basename $d); s=${s#shard}
  ( cd $ROOT/external/MedAgents && TMC_NONCE_NS=2A:$TAG:M1:$r python run.py --method syn_verif \
      --start_pos $((s*20+n)) --end_pos $((s*20+20)) --output_files_folder $d >> $d/log.txt 2>&1 ) & PIDS="$PIDS $!"
done
wait $PIDS
pkill -f "^$VPY/python.*vllm serve $MODEL --port $PORT"
while pgrep -f "^$VPY/python.*vllm serve $MODEL --port $PORT" >/dev/null; do sleep 3; done
echo "$(date +%H:%M:%S) fill done $TAG; M1=$(cat $OUT/M1/run*/shard*/chatgpt-syn_verif | wc -l)/600"
