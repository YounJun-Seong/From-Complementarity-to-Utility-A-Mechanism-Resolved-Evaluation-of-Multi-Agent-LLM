"""MedLA full test (MEDLA_PREREGISTRATION.md Amendment 2; no LLM calls). 4-option MedQA, 1,273 q.
Primary: run 1 of each system (L0, S4, L1), question-level paired: bootstrap 95% CI (10k, seed 0),
exact McNemar, paired-t TOST (δ=0.03 primary). Sensitivity: S4 / L1 runs 1–3.
Then the pre-registered sequential replication rule for L0.

  python experiments/medla_full_report.py
"""
import json
import os

import numpy as np
from scipy import stats

ROOT = os.path.join(os.path.dirname(__file__), "..")
RNG = np.random.default_rng(0)
Q = 1273
PRICE = (0.15, 0.60)
D = f"{ROOT}/runs/medla_full"


def medla(sysname, r):
    rows = {json.loads(l)["idx"]: json.loads(l) for l in open(f"{D}/{sysname}/run{r}.jsonl")}
    R = [rows[i] for i in range(Q)]
    return R, np.array([x["final_decision"] == x["label"] for x in R], float), [x["label"] for x in R]


def s4(r):
    R = [json.loads(l) for l in open(f"{D}/S4/run{r}.jsonl")]
    return R, np.array([x["E3"] == x["gold"] for x in R], float), [x["gold"] for x in R]


def compare(a, b, name_a, name_b):
    d = b - a
    bs = np.array([d[RNG.integers(0, Q, Q)].mean() for _ in range(10000)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    n01, n10 = int(((a == 0) & (b == 1)).sum()), int(((a == 1) & (b == 0)).sum())
    p = stats.binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    se = d.std(ddof=1) / np.sqrt(Q)
    tost = {dl: max(stats.t.sf((d.mean() + dl) / se, Q - 1), stats.t.cdf((d.mean() - dl) / se, Q - 1))
            for dl in (0.01, 0.02, 0.03, 0.04, 0.05)}
    print(f"  {name_b} − {name_a} = {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]  McNemar p={p:.2g} "
          f"({name_b} only {n01}, {name_a} only {n10})")
    print("    TOST p: " + " ".join(("*" if dl == 0.03 else "") + f"δ{dl}:{v:.4f}" for dl, v in tost.items()))
    return d.mean(), lo, hi


def main():
    L0, c0, g0 = medla("L0", 1)
    S = [s4(r) for r in (1, 2, 3)]
    L1 = [medla("L1", r) for r in (1, 2, 3)]
    assert g0 == S[0][2] == L1[0][2], "gold mismatch"
    print("MedLA full test — MedQA-US 4-option, 1,273 questions, gpt-4o-mini")
    tokS = np.mean([x["S_prompt_tokens"] + x["S_completion_tokens"] for x in S[0][0]])
    usdS = np.mean([(x["S_prompt_tokens"] * PRICE[0] + x["S_completion_tokens"] * PRICE[1]) / 1e6 for x in S[0][0]])
    tok0 = np.mean([x["prompt_tokens"] + x["completion_tokens"] for x in L0])
    usd0 = np.mean([(x["prompt_tokens"] * PRICE[0] + x["completion_tokens"] * PRICE[1]) / 1e6 for x in L0])
    tok1 = np.mean([x["prompt_tokens"] + x["completion_tokens"] for x in L1[0][0]])
    print(f"  L0 run1 {c0.mean():.4f}  (calls/q {np.mean([x['n_calls'] for x in L0]):.1f}, tokens/q {tok0:.0f}, $/q {usd0:.5f}, "
          f"wall {np.mean([x['wall_time'] for x in L0]):.0f}s/q, crash {np.mean([x['error'] is not None for x in L0]):.3f})")
    print(f"  S4 run1 {S[0][1].mean():.4f}  (tokens/q {tokS:.0f}, $/q {usdS:.5f});  runs 1-3: "
          f"{', '.join(f'{s[1].mean():.4f}' for s in S)}  SD {np.std([s[1].mean() for s in S], ddof=1):.4f}")
    print(f"  L1 run1 {L1[0][1].mean():.4f}  (tokens/q {tok1:.0f});  runs 1-3: "
          f"{', '.join(f'{l[1].mean():.4f}' for l in L1)}  SD {np.std([l[1].mean() for l in L1], ddof=1):.4f}")
    print(f"  cost L0/S4: tokens {tok0 / tokS:.1f}x, $ {usd0 / usdS:.1f}x\n")

    print("PRIMARY (run 1 vs run 1)")
    d, lo, hi = compare(S[0][1], c0, "S4", "L0")
    print("SECONDARY")
    compare(L1[0][1], c0, "L1", "L0")
    print("DESCRIPTIVE")
    compare(L1[0][1], S[0][1], "L1", "S4")
    print("SENSITIVITY: L0 run1 vs S4 / L1 averaged over runs 1–3")
    compare(np.mean([s[1] for s in S], 0), c0, "S4(mean of 3)", "L0")
    compare(np.mean([l[1] for l in L1], 0), c0, "L1(mean of 3)", "L0")

    trigger = abs(d) <= 0.02 or (lo <= -0.03 <= hi) or (lo <= 0.03 <= hi)
    print(f"\nSequential replication rule: |Δ|={abs(d):.4f} (≤0.02: {abs(d) <= 0.02}); "
          f"CI [{lo:+.4f},{hi:+.4f}] contains −0.03: {lo <= -0.03 <= hi}, +0.03: {lo <= 0.03 <= hi}")
    print("=> L0 runs 2–3 REQUIRED by the rule" if trigger else "=> no further L0 runs")


if __name__ == "__main__":
    main()
