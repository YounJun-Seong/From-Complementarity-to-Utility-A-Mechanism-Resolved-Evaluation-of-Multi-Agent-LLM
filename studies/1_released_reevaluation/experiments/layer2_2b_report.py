"""Layer 2B report (no LLM calls). Per Qwen model: MedAgents' own single-agent baseline at 300 (B1)
vs 700 (B3) tokens, 3 runs each, MedQA 0–199. Question-level means over runs; paired bootstrap over
questions (10k, seed 0). Mechanism split as P0 Table 2, at the run×question level (a unit is
"truncated" if that B1 run hit finish_reason=length), with question-cluster bootstrap CIs.
2A S and M1 for the same model are shown for reference.

  python experiments/layer2_2b_report.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import layer2_2a_report as A   # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
RNG = np.random.default_rng(0)
Q = 200


def load(tag, budget):
    runs = []
    for r in (1, 2, 3):
        f = f"{ROOT}/runs/layer2/2B/{tag}/run{r}/empty_{budget}.jsonl"
        if not os.path.exists(f):
            return None
        runs.append([json.loads(l) for l in open(f)][:Q])
    return runs


def cluster_ci(fn, B=5000):
    vals = [fn(RNG.integers(0, Q, Q)) for _ in range(B)]
    return np.nanpercentile(vals, [2.5, 97.5])


def main():
    print(f"{'model':22s} {'B1':>6s} {'B3':>6s} {'B3−B1 [95% CI]':>24s} {'len@300':>7s} {'len@700':>7s} "
          f"{'2A S':>6s} {'2A M1':>6s}")
    split = []
    for tag in A.MODELS:
        b1, b3 = load(tag, 300), load(tag, 700)
        if b1 is None or b3 is None:
            print(f"{tag:22s} pending"); continue
        c1 = np.array([[r["pred"] == r["gold"] for r in run] for run in b1], float)
        c3 = np.array([[r["pred"] == r["gold"] for r in run] for run in b3], float)
        t1 = np.array([[r["finish_reason"] == "length" for r in run] for run in b1])
        t3 = np.array([[r["finish_reason"] == "length" for r in run] for run in b3])
        d = c3.mean(0) - c1.mean(0)
        lo, hi = cluster_ci(lambda i: d[i].mean())
        S, M = A.load_s(tag), A.load_m1(tag)
        s = np.mean([[r["E3"] == r["gold"] for r in run] for run in S]) if S else np.nan
        m = np.mean([[r["pred_answer"] == r["gold_answer"] for r in run] for run in M]) if M else np.nan
        print(f"{tag:22s} {c1.mean():6.3f} {c3.mean():6.3f} {d.mean():+7.3f} [{lo:+.3f},{hi:+.3f}] "
              f"{t1.mean():7.3f} {t3.mean():7.3f} {s:6.3f} {m:6.3f}")

        def rec(i, mask):
            mm = mask[:, i]
            return (c3[:, i][mm].mean() - c1[:, i][mm].mean()) if mm.any() else np.nan
        split.append((tag, t1, c1, c3, rec))
    print("\nmechanism split (units = run x question; B1 finish state):")
    print(f"{'model':22s} {'subset':12s} {'n':>5s} {'B1':>6s} {'B3':>6s} {'B3−B1 [95% CI]':>24s} {'share of net gain':>17s}")
    for tag, t1, c1, c3, rec in split:
        net = (c3 - c1).sum()
        for lab, mask in (("truncated", t1), ("natural", ~t1)):
            lo, hi = cluster_ci(lambda i: rec(i, mask))
            g = (c3 - c1)[mask].sum()
            print(f"{tag:22s} {lab:12s} {mask.sum():5d} {c1[mask].mean():6.3f} {c3[mask].mean():6.3f} "
                  f"{c3[mask].mean() - c1[mask].mean():+7.3f} [{lo:+.3f},{hi:+.3f}] "
                  f"{(g / net if net else float('nan')):17.0%}")


if __name__ == "__main__":
    main()
