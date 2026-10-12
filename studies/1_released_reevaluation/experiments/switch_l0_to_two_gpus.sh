#!/bin/bash
# After Qwen14B 2E frees GPU0: stop the single-GPU L0 re-run and restart it split across both GPUs.
ROOT=$(cd "$(dirname "$0")/.." && pwd); VPY=<VLLM_ENV>/bin
cd $ROOT
until grep -q "server down Qwen/Qwen2.5-14B-Instruct" logs/extra_gpu0_14b.log; do sleep 30; done
echo "$(date +%H:%M:%S) GPU0 free; stopping single-GPU L0 re-run"
for p in $(pgrep -f "^/bin/bash -c while kill -0 3421838"); do kill $p; done            # queue3 waiter
for p in $(pgrep -f "^bash experiments/run_layer3_l0_resume.sh"); do kill $p; done       # resume driver
for p in $(pgrep -f "^python main.py --dataset medqa_us"); do kill $p; done               # MedLA L0 process
sleep 5
pkill -f "^$VPY/python.*vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8051"
while pgrep -f "^$VPY/python.*vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8051" >/dev/null; do sleep 3; done
echo "$(date +%H:%M:%S) stopped; valid rows so far: $(wc -l < runs/layer3/L0/run1.jsonl)"
nohup bash experiments/run_layer3_l0_shard.sh 1 8051 0 2 0.85 > logs/l0_shard0.log 2>&1 &
nohup bash experiments/run_layer3_l0_shard.sh 0 8052 1 2 0.75 > logs/l0_shard1.log 2>&1 &
echo "$(date +%H:%M:%S) launched shard 0 (GPU1, even idx) and shard 1 (GPU0, odd idx)"
