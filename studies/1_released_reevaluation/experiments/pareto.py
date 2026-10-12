"""Accuracy–compute frontier per open model (no LLM calls): S, SC-5/10/20, Reflection (2D) and M1 (2A),
MedQA 0–199. Accuracy = mean over 3 runs; tokens = mean total tokens per question. A method is on the
Pareto frontier if no other method has >= accuracy with <= tokens (one strictly better).
Writes runs/layer2/2D/pareto_<model>.csv.

  python experiments/pareto.py
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


def m1_tokens(tag):
    """M1 tokens/q: per-question COST_LOG if the 2A run logged it, else the 2A summary.csv value."""
    logs = glob.glob(f"{ROOT}/runs/layer2/2A/{tag}/M1/run*/shard*/cost.jsonl")
    if logs:
        C = [json.loads(l) for f in logs for l in open(f)]
        return np.mean([c["prompt_tokens"] + c["completion_tokens"] for c in C])
    import csv
    for r in csv.DictReader(open(f"{ROOT}/runs/layer2/2A/summary.csv")):
        if r["model"] == tag:
            return float(r["M1_tok_q"])
    return float("nan")


for tag in MODELS:
    rows = []
    S = A.load_s(tag)
    if S is None:
        print(f"{tag}: 2A S missing"); continue
    rows.append(("S (zero-shot CoT)", np.mean([[r["E3"] == r["gold"] for r in run] for run in S]),
                 np.mean([r["S_prompt_tokens"] + r["S_completion_tokens"] for run in S for r in run]), 2))
    for m in ("sc-5", "sc-10", "sc-20", "reflect"):
        fs = sorted(glob.glob(f"{ROOT}/runs/layer2/2D/{tag}/{m}/run*.jsonl"))
        if len(fs) < 3:
            continue
        R = [jl(f) for f in fs]
        rows.append((m.upper() if m != "reflect" else "Reflection", np.mean([[r["pred"] == r["gold"] for r in run] for run in R]),
                     np.mean([r["prompt_tokens"] + r["completion_tokens"] for run in R for r in run]),
                     np.mean([r["calls"] for run in R for r in run])))
    M = A.load_m1(tag)
    if M is not None:
        rows.append(("M1 MedAgents", np.mean([[r["pred_answer"] == r["gold_answer"] for r in run] for run in M]),
                     m1_tokens(tag), float("nan")))
    front = [not any((b[1] >= a[1] and b[2] <= a[2]) and (b[1] > a[1] or b[2] < a[2]) for b in rows) for a in rows]
    print(f"=== {tag} ===")
    for (name, acc, tok, calls), f in zip(rows, front):
        print(f"  {name:20s} acc {acc:.3f}  tokens/q {tok:8.0f}  {'<- frontier' if f else ''}")
    with open(f"{ROOT}/runs/layer2/2D/pareto_{tag}.csv", "w") as fh:
        fh.write("method,acc,tokens_per_q,calls_per_q,frontier\n")
        for (name, acc, tok, calls), f in zip(rows, front):
            fh.write(f"{name},{acc:.4f},{tok:.0f},{calls},{int(f)}\n")
