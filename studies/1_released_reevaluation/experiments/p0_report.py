"""P0 closing tables (epoch data only; no API calls).

Table 1  method summary: acc, Δ vs B1 / B3 with 95% CI + McNemar, TOST(δ=.03) where
         pre-registered, tokens/q, cost ratio vs S.
Table 2  mechanism: B1 subsets (length-truncated vs natural stop) x {B1, B3, M1}.
Cost uses gpt-4o-mini list price (USD per 1M tokens) below; change PRICE if it differs.

  python experiments/p0_report.py
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from p0_stats import compare, medagents_rerun, systems, B1, B3, B4, M1, S_   # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
PRICE = {"in": 0.15, "out": 0.60}          # gpt-4o-mini-2024-07-18, $/1M tokens


def load(p):
    return [json.loads(l) for l in open(p)]


def tokens():
    T = {}
    for k, f in ((B1, "empty_300"), (B3, "empty_700"), (B4, "expert_700")):
        R = load(f"{ROOT}/runs/p0_sweep/{f}.jsonl")
        T[k] = (np.mean([r["prompt_tokens"] for r in R]), np.mean([r["completion_tokens"] for r in R]), 1.0)
    E = load(f"{ROOT}/runs/p0_extraction.jsonl")
    if "S_prompt_tokens" in E[0]:
        T[S_] = (np.mean([r["S_prompt_tokens"] for r in E]), np.mean([r["S_completion_tokens"] for r in E]), 2.0)
    C = [r for f in glob.glob(f"{ROOT}/runs/m1_medagents/shards/*/cost.jsonl") for r in load(f)]
    if len({r["idx"] for r in C}) == 1273:
        C = {r["idx"]: r for r in C}.values()
        T[M1] = (np.mean([r["prompt_tokens"] for r in C]), np.mean([r["completion_tokens"] for r in C]),
                 np.mean([r["calls"] for r in C]))
    return T


def cost(t):
    return (t[0] * PRICE["in"] + t[1] * PRICE["out"]) / 1e6


def main():
    S = systems()
    T = tokens()
    names = [(B1, "B1 Empty-300"), (B3, "B3 Empty-700"), (B4, "B4 Expert-700"), (S_, "S strong zs-CoT"), (M1, "M1 MedAgents")]
    print("Table 1 — MedQA test, n=1273, gpt-4o-mini, P0 epoch")
    print(f"{'method':17s} {'acc':>6s} {'Δ vs B1 [95% CI]':>26s} {'p':>8s} {'Δ vs B3 [95% CI]':>26s} {'p':>8s} "
          f"{'TOST.03':>8s} {'tok/q':>7s} {'calls/q':>7s} {'$/q':>9s} {'tok×S':>6s} {'$×S':>6s}")
    for k, lab in names:
        if k not in S:
            print(f"{lab:17s} pending"); continue
        cells = []
        for ref in (B1, B3):
            if k == ref or ref not in S or (ref == B3 and k == B1):
                cells.append((f"{'–':>26s}", f"{'':>8s}")); continue
            r = compare(S, ref, k)
            cells.append((f"{r['delta']:+.3f} [{r['ci95_lo']:+.3f},{r['ci95_hi']:+.3f}]".rjust(26), f"{r['mcnemar_p']:8.1e}"))
        tost = ""
        if k == M1:
            tost = " ".join(f"{compare(S, a, M1)['tost_0.03']:.3f}" for a in (B3, S_) if a in S)
        t = T.get(k)
        tok = f"{t[0] + t[1]:7.0f} {t[2]:7.1f} {cost(t):9.6f}" if t else f"{'':>7s} {'':>7s} {'':>9s}"
        rel = f"{(t[0] + t[1]) / sum(T[S_][:2]):6.1f} {cost(t) / cost(T[S_]):6.1f}" if t and S_ in T else ""
        print(f"{lab:17s} {S[k].mean():6.3f} {cells[0][0]} {cells[0][1]} {cells[1][0]} {cells[1][1]} {tost:>8s} {tok} {rel}")
    print("TOST.03 for M1 = p vs B3, p vs S (pre-registered primary margin).")

    sys.path.insert(0, os.path.dirname(__file__))
    from p0_truncation import category
    R1 = load(f"{ROOT}/runs/p0_sweep/empty_300.jsonl")
    trunc = np.array([category(r) == "length" for r in R1])
    b1, b3 = S[B1], S[B3]
    m1 = S.get(M1)
    print("\nTable 2 — mechanism, split by B1 (Empty-300) finish state")
    print(f"{'B1 subset':18s} {'n':>5s} {'B1':>6s} {'B3':>6s} {'M1':>6s} {'B3−B1':>7s} {'M1−B1':>7s} {'M1−B3':>7s}")
    for lab, k in (("length-truncated", trunc), ("natural stop", ~trunc), ("all", np.ones_like(trunc))):
        m = f"{m1[k].mean():6.3f}" if m1 is not None else f"{'?':>6s}"
        d1 = f"{m1[k].mean() - b1[k].mean():+7.3f}" if m1 is not None else f"{'?':>7s}"
        d3 = f"{m1[k].mean() - b3[k].mean():+7.3f}" if m1 is not None else f"{'?':>7s}"
        print(f"{lab:18s} {k.sum():5d} {b1[k].mean():6.3f} {b3[k].mean():6.3f} {m} {b3[k].mean() - b1[k].mean():+7.3f} {d1} {d3}")
    if m1 is not None:
        g = m1.astype(int) - b1.astype(int)
        print(f"\nM1 net gain over B1: {g.sum():+d} questions; on the {trunc.sum()} truncated {g[trunc].sum():+d} "
              f"({g[trunc].sum() / g.sum():.0%}), on the {(~trunc).sum()} natural-stop {g[~trunc].sum():+d}")


if __name__ == "__main__":
    main()
