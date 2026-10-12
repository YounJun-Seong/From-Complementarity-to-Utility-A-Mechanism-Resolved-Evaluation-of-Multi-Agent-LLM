"""R3 inference-consumption table (no LLM calls): input / output / total tokens and calls per question for
S, SC-5/10/20, SC-7 (Llama), Reflection and M1 on MedQA first-200 (2A / 2D runs, 3 runs each).
M1 token split: per-question COST_LOG when present (Llama), else the cache records listed in
runs/layer2/2A/<model>.m1_cachefiles.txt (unique cached generations; Qwen).

  python experiments/compute_table.py
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import layer2_2a_report as A   # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
MODELS = ["Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct", "Llama-3.1-8B-Instruct"]


def jl(p):
    return [json.loads(l) for l in open(p)]


out = open(f"{ROOT}/paper_assets/table_compute_consumption.md", "w")
for tag in MODELS:
    rows = []
    S = A.load_s(tag)
    rows.append(("S", np.mean([[r["E3"] == r["gold"] for r in run] for run in S]),
                 np.mean([r["S_prompt_tokens"] for run in S for r in run]), np.mean([r["S_completion_tokens"] for run in S for r in run]), 2.0))
    for m in ("sc-5", "sc-7", "sc-10", "sc-20", "reflect"):
        fs = sorted(glob.glob(f"{ROOT}/runs/layer2/2D/{tag}/{m}/run*.jsonl"))
        if len(fs) < 3:
            continue
        R = [jl(f) for f in fs]
        rows.append(({"reflect": "Reflection"}.get(m, m.upper()), np.mean([[r["pred"] == r["gold"] for r in run] for run in R]),
                     np.mean([r["prompt_tokens"] for run in R for r in run]), np.mean([r["completion_tokens"] for run in R for r in run]),
                     np.mean([r["calls"] for run in R for r in run])))
    M = A.load_m1(tag)
    acc = np.mean([[r["pred_answer"] == r["gold_answer"] for r in run] for run in M])
    logs = glob.glob(f"{ROOT}/runs/layer2/2A/{tag}/M1/run*/shard*/cost.jsonl")
    lst = f"{ROOT}/runs/layer2/2A/{tag}.m1_cachefiles.txt"
    if logs:
        C = [json.loads(l) for f in logs for l in open(f)]
        rows.append(("MedAgents M1", acc, np.mean([c["prompt_tokens"] for c in C]), np.mean([c["completion_tokens"] for c in C]),
                     np.mean([c["calls"] for c in C])))
    elif os.path.exists(lst) and os.path.getsize(lst) > 0:
        pi = po = n = 0
        for f in open(lst).read().split():
            u = json.load(open(f)).get("usage") or {}
            pi += u.get("prompt_tokens", 0); po += u.get("completion_tokens", 0); n += 1
        rows.append(("MedAgents M1", acc, pi / 600, po / 600, n / 600))
    else:
        rows.append(("MedAgents M1", acc, float("nan"), float("nan"), float("nan")))
    out.write(f"\n**{tag}** (MedQA first 200, 3 runs; per question)\n\n| Method | Accuracy | Input tokens | Output tokens | Total tokens | Calls |\n|---|---|---|---|---|---|\n")
    print(f"=== {tag} ===")
    for name, a, ti, to, c in rows:
        out.write(f"| {name} | {a:.3f} | {ti:,.0f} | {to:,.0f} | {ti + to:,.0f} | {c:.1f} |\n")
        print(f"  {name:13s} acc {a:.3f}  input {ti:8.0f}  output {to:7.0f}  total {ti + to:8.0f}  calls {c:5.1f}")
out.write("\nM1 calls for Qwen count unique cached generations per run namespace (identical repeated calls within a run are served from cache).\n")
