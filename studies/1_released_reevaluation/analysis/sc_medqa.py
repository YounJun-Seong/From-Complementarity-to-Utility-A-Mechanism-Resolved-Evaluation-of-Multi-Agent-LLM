"""E-⑥ same-compute self-consistency vs multi-agent, on MedQA (MCQA).
Multi-agent syn_verif spends ~24 LLM calls/question. Here a single model is given the SAME
budget as k=20 CoT samples (majority vote). Compares SC-single to (a) 1-shot single zeroshot_cot
and (b) the multi-agent system on the SAME first-N questions. If SC-single >= multi-agent at a
fraction of the coordination machinery, the multi-agent design adds nothing even at matched compute.
Also serves as a POSITIVE CONTROL: SC>single (if significant) shows our harness detects real gains.
"""
import argparse
import json
import os
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from llm import sample_n

RNG = np.random.default_rng(0)
K = 20
QA = [json.loads(l) for l in open("external/MedAgents/datasets/MedQA/test.jsonl")]
COT = ("You are an expert physician. Reason step by step about the case, then end with a "
       "line 'The answer is X' where X is one of A, B, C, D, E.")


def opts(q):
    return "\n".join(f"{k}. {v}" for k, v in q["options"].items())


def letter(t):
    m = re.search(r"answer is\s*\(?([ABCDE])\)?", t, re.I)
    if m:
        return m.group(1).upper()
    m = list(re.finditer(r"\b([ABCDE])\b", t))
    return m[-1].group(1).upper() if m else ""


def sc(q, k=K):
    outs = sample_n([{"role": "system", "content": COT},
                     {"role": "user", "content": f"{q['question']}\n{opts(q)}"}],
                    k=k, temperature=0.7, max_tokens=500, seed=0)
    votes = [letter(t) for t in outs]
    votes = [v for v in votes if v]
    return Counter(votes).most_common(1)[0][0] if votes else ""


def official(method, n):
    L = [json.loads(l) for l in open(f"external/MedAgents/outputs/MedQA/chatgpt-{method}")]
    return np.array([int(x["pred_answer"] == x["gold_answer"]) for x in L])[:n]


def ci(x):
    b = [RNG.choice(x, len(x), replace=True).mean() for _ in range(5000)]
    return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def gain(a, b):
    d = b - a
    bs = [RNG.choice(d, len(d), replace=True).mean() for _ in range(5000)]
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--k", type=int, default=K)
    args = ap.parse_args()
    n = args.n
    Q = QA[:n]
    kk = args.k
    with ThreadPoolExecutor(max_workers=8) as ex:
        preds = list(ex.map(lambda q: sc(q, k=kk), Q))
    scv = np.array([int(p == q["answer_idx"]) for p, q in zip(preds, Q)])
    pd.DataFrame({"sc": scv}).to_csv("data/sc_medqa.csv", index=False)

    single = pd.read_csv("data/e2_fair_single.csv")["zeroshot_cot"].to_numpy()[:n]
    multi = official("syn_verif", n)
    print(f"MedQA self-consistency (k={K}) vs single vs multi-agent (n={n}):")
    for nm, x in [("single: zeroshot_cot(1)", single), (f"self_consistency(k={K})", scv),
                  ("multi-agent (syn_verif)", multi)]:
        m, lo, hi = ci(x); print(f"  {nm:26s} acc={m:.3f} [{lo:.3f},{hi:.3f}]")
    g, lo, hi = gain(single, scv)
    print(f"\n  SC - single    = {g:+.3f} [{lo:+.3f},{hi:+.3f}]  "
          f"{'POSITIVE CONTROL: SC detects a real gain' if lo > 0 else 'n.s.'}")
    g, lo, hi = gain(multi, scv)
    print(f"  SC - multi     = {g:+.3f} [{lo:+.3f},{hi:+.3f}]  "
          f"{'SC beats multi at matched compute' if lo > 0 else 'TIE/na'}")
    print("SC_MEDQA_DONE")


if __name__ == "__main__":
    main()
