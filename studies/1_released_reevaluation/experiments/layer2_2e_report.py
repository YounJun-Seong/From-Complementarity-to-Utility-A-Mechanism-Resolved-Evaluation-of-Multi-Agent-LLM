"""Layer 2E report (no LLM calls): option-order robustness, Qwen2.5-14B, MedQA 0–199.
Original order = 2A run 1; permutations 1–3 = 2E (1 run each). Answers are compared by option
CONTENT (letter -> text) so a consistent answer survives relabelling.

  python experiments/layer2_2e_report.py
"""
import glob
import json
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
TAG = "Qwen2.5-14B-Instruct"
Q = 200
RNG = np.random.default_rng(0)


def jl(p):
    return [json.loads(l) for l in open(p)]


def m1(d):
    by = {}
    for sd in glob.glob(f"{d}/shard*"):
        s = int(os.path.basename(sd)[5:])
        for k, l in enumerate(open(f"{sd}/chatgpt-syn_verif")):
            by[s * 20 + k] = json.loads(l)
    return [by[i] for i in range(Q)]


data = {"orig": jl(f"{ROOT}/external/MedAgents/datasets/MedQA/test.jsonl")[:Q]}
for p in (1, 2, 3):
    data[f"perm{p}"] = jl(f"{ROOT}/external/MedAgents/datasets/MedQA_perm{p}/test.jsonl")[:Q]
S = {"orig": jl(f"{ROOT}/runs/layer2/2A/{TAG}/S/run1.jsonl")[:Q]}
M = {"orig": m1(f"{ROOT}/runs/layer2/2A/{TAG}/M1/run1")}
for p in (1, 2, 3):
    S[f"perm{p}"] = jl(f"{ROOT}/runs/layer2/2E/{TAG}/S_perm{p}.jsonl")
    M[f"perm{p}"] = m1(f"{ROOT}/runs/layer2/2E/{TAG}/M1_perm{p}")

content = lambda cond, i, letter: data[cond][i]["options"].get(letter, "") if letter else ""
print(f"{'order':6s} {'S acc':>6s} {'M1 acc':>7s} {'M1 − S [95% CI]':>24s}")
diffs = []
for c in data:
    s = np.array([S[c][i]["E3"] == data[c][i]["answer_idx"] for i in range(Q)], float)
    m = np.array([M[c][i]["pred_answer"] == data[c][i]["answer_idx"] for i in range(Q)], float)
    d = m - s; diffs.append(d)
    bs = [d[RNG.integers(0, Q, Q)].mean() for _ in range(10000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"{c:6s} {s.mean():6.3f} {m.mean():7.3f} {d.mean():+7.3f} [{lo:+.3f},{hi:+.3f}]")
d = np.mean(diffs, 0)
bs = [d[RNG.integers(0, Q, Q)].mean() for _ in range(10000)]
lo, hi = np.percentile(bs, [2.5, 97.5])
print(f"{'pooled':6s} (mean over 4 orders per question) M1 − S = {d.mean():+.3f} [{lo:+.3f},{hi:+.3f}]")
for name, R, key in (("S", S, "E3"), ("M1", M, "pred_answer")):
    same = np.mean([len({content(c, i, R[c][i][key]) for c in data}) == 1 for i in range(Q)])
    print(f"{name}: same chosen option content in all 4 orders {same:.3f}")
