"""E-② TOST equivalence testing. 'CI includes 0' is absence-of-evidence, not
evidence-of-absence. Here we pre-register an equivalence margin delta and test
H0: |acc_multi - acc_single| >= delta  vs  H1: |diff| < delta  (two one-sided tests
on the paired per-question difference, bootstrap). If the 90% CI of the difference
lies fully within (-delta, +delta) we DECLARE statistical equivalence.

Margin delta = 0.05 (5 accuracy points) — pre-registered as the smallest difference
we would consider practically meaningful for a multi-agent system that costs ~N x more.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

RNG = np.random.default_rng(0)
DELTA = 0.05
MED_OUT = "external/MedAgents/outputs"


def official(path):
    L = [json.loads(l) for l in open(path)]
    return np.array([int(x["pred_answer"] == x["gold_answer"]) for x in L])


def tost(single, multi, delta=DELTA, B=10000):
    """Paired bootstrap TOST. Returns (diff, 90%CI lo/hi, equivalent?, both one-sided p)."""
    n = min(len(single), len(multi))
    d = multi[:n] - single[:n]
    boot = np.array([RNG.choice(d, n, replace=True).mean() for _ in range(B)])
    lo, hi = np.percentile(boot, [5, 95])          # 90% CI == two one-sided 5% tests
    # one-sided p-values
    p_upper = np.mean(boot >= delta)               # H0a: diff >= +delta
    p_lower = np.mean(boot <= -delta)              # H0b: diff <= -delta
    p_tost = max(p_upper, p_lower)
    equiv = (lo > -delta) and (hi < delta)
    return d.mean(), lo, hi, equiv, p_tost, n


def report(name, single, multi):
    diff, lo, hi, equiv, p, n = tost(single, multi)
    verdict = "EQUIVALENT" if equiv else "not shown (widen n or effect real)"
    print(f"  {name:34s} n={n:3d}  multi-single={diff:+.3f}  90%CI[{lo:+.3f},{hi:+.3f}]  "
          f"p_TOST={p:.3f}  -> {verdict}")


def main():
    print(f"TOST equivalence, pre-registered margin delta = +/-{DELTA}\n")

    # MedQA n=200: standard single (zeroshot_cot) vs MedAgents syn_verif
    e2 = pd.read_csv("data/e2_fair_single.csv")
    syn = official(f"{MED_OUT}/MedQA/chatgpt-syn_verif")
    report("MedQA: zeroshot_cot vs MedAgents", e2["zeroshot_cot"].to_numpy(), syn)

    # MedMCQA n=100
    if os.path.exists("data/e2_fair_mcqa.csv"):
        e2m = pd.read_csv("data/e2_fair_mcqa.csv")
        synm = official(f"{MED_OUT}/MedMCQA/chatgpt-syn_verif")
        report("MedMCQA: zeroshot_cot vs MedAgents", e2m["zeroshot_cot"].to_numpy(), synm)

    # MedQA n=50: single vs MDAgents (2nd system)
    mda = "external/MDAgents/output/gpt-4o-mini_medqa_adaptive.json"
    if os.path.exists(mda):
        import re
        D = json.load(open(mda))
        def ext(r):
            t = "\n".join(str(v) for v in r.values()) if isinstance(r, dict) else str(r)
            for pat in [r"Answer:\s*\(?([ABCDE])\)?", r"\(([ABCDE])\)", r"\b([ABCDE])\b"]:
                m = list(re.finditer(pat, t, re.I))
                if m:
                    return m[-1].group(1).upper()
            return ""
        mdc = np.array([int(ext(d["response"]) == d["label"]) for d in D])
        n = len(mdc)
        report("MedQA: zeroshot_cot vs MDAgents", e2["zeroshot_cot"].to_numpy()[:n], mdc)

    print(f"\nNote: equivalence at delta={DELTA} means multi-agent is not more than "
          f"{DELTA*100:.0f} accuracy points better than a standard single prompt.")
    print("TOST_DONE")


if __name__ == "__main__":
    main()
