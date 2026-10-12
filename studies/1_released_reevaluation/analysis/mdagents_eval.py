"""Extract MDAgents' final answer letter from output/*.json and score vs gold label,
on the SAME first-N MedQA questions as MedAgents/single (aligned test.jsonl). Prints
accuracy + difficulty routing breakdown, and (if the other methods' per-question
correctness are available) a same-N bootstrap-CI comparison."""
import json
import os
import re
import sys

import numpy as np
import pandas as pd

RNG = np.random.default_rng(0)
MDA = "external/MDAgents/output/gpt-4o-mini_medqa_adaptive.json"
MED_OUT = "external/MedAgents/outputs/MedQA"


def flat(resp):
    if isinstance(resp, dict):
        return "\n".join(str(v) for v in resp.values())
    return str(resp)


def extract(resp):
    t = flat(resp)
    # prefer an explicit "Answer: (X)" / "Answer: X)" near the end
    pats = [r"Answer:\s*\(?([ABCDE])\)?", r"answer\s*is\s*\(?([ABCDE])\)?",
            r"\(([ABCDE])\)", r"\b([ABCDE])\b"]
    for pat in pats:
        m = list(re.finditer(pat, t, re.IGNORECASE))
        if m:
            return m[-1].group(1).upper()
    return ""


def official(method, n):
    L = [json.loads(l) for l in open(f"{MED_OUT}/chatgpt-{method}")]
    return np.array([int(x["pred_answer"] == x["gold_answer"]) for x in L])[:n]


def ci(x):
    b = [RNG.choice(x, len(x), replace=True).mean() for _ in range(5000)]
    return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def gain(a, b):
    d = b - a
    bs = [RNG.choice(d, len(d), replace=True).mean() for _ in range(5000)]
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def main():
    D = json.load(open(MDA))
    n = len(D)
    corr = np.array([int(extract(d["response"]) == d["label"]) for d in D])
    diffs = pd.Series([d["difficulty"] for d in D])
    print(f"MDAgents (official, adaptive) on first {n} MedQA:")
    m, lo, hi = ci(corr)
    print(f"  accuracy = {m:.3f} [{lo:.3f},{hi:.3f}]")
    print(f"  difficulty routing: {diffs.value_counts().to_dict()}")

    # same-N comparison vs base_cot / multi (MedAgents) / best single
    base = official("base_cot", n); syn = official("syn_verif", n)
    print(f"\n  same {n} questions:")
    print(f"    base_cot (paper baseline) = {base.mean():.3f}")
    print(f"    MedAgents (syn_verif)     = {syn.mean():.3f}")
    print(f"    MDAgents (adaptive)       = {corr.mean():.3f}")
    sv = "data/single_variants_medqa.csv"; e2 = "data/e2_fair_single.csv"
    if os.path.exists(e2):
        z = pd.read_csv(e2)["zeroshot_cot"].to_numpy()[:n]
        print(f"    single: zeroshot_cot      = {z.mean():.3f}")
        g, lo, hi = gain(z, corr)
        print(f"    MDAgents - zeroshot_cot   = {g:+.3f} [{lo:+.3f},{hi:+.3f}]  "
              f"{'TIE' if lo <= 0 <= hi else 'differ'}")
    g, lo, hi = gain(base, corr)
    print(f"    MDAgents - base_cot       = {g:+.3f} [{lo:+.3f},{hi:+.3f}]")
    print("MDAGENTS_EVAL_DONE")


if __name__ == "__main__":
    main()
