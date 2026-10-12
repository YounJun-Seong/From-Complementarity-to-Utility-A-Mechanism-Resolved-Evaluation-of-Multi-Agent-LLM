"""MedLA pilot report (MEDLA_PREREGISTRATION.md; no LLM calls). 4-option MedQA, questions 0–199.
Per-question score = mean correctness over 3 runs; paired bootstrap over questions (10k, seed 0);
exact McNemar on majority-vote answers; paired-t TOST (primary δ = 0.03).

  python experiments/medla_report.py
"""
import collections
import json
import os

import numpy as np
from scipy import stats

ROOT = os.path.join(os.path.dirname(__file__), "..")
RNG = np.random.default_rng(0)
Q = 200
PRICE = (0.15, 0.60)


def load(sysname):
    runs = []
    for r in (1, 2, 3):
        rows = {json.loads(l)["idx"]: json.loads(l) for l in open(f"{ROOT}/runs/medla/{sysname}/run{r}.jsonl")} \
            if sysname != "S4" else dict(enumerate(json.loads(l) for l in open(f"{ROOT}/runs/medla/S4/run{r}.jsonl")))
        runs.append([rows[i] for i in range(Q)])
    return runs


def answers(sysname, runs):
    if sysname == "S4":
        return [[r["E3"] for r in run] for run in runs], [r["gold"] for r in runs[0]]
    return [[r["final_decision"] if r["error"] is None else "" for r in run] for run in runs], [r["label"] for r in runs[0]]


def main():
    A, G, meta = {}, None, {}
    for s in ("L0", "L1", "S4"):
        runs = load(s)
        ans, gold = answers(s, runs)
        G = G or gold
        assert gold == G, "gold mismatch across systems"
        A[s] = ans
        corr = np.array([[a == g for a, g in zip(run, G)] for run in ans], float)
        if s == "S4":
            tok = np.mean([r["S_prompt_tokens"] + r["S_completion_tokens"] for run in runs for r in run])
            usd = np.mean([(r["S_prompt_tokens"] * PRICE[0] + r["S_completion_tokens"] * PRICE[1]) / 1e6 for run in runs for r in run])
            calls, crash, wall = 2.0, 0.0, float("nan")
        else:
            tok = np.mean([r["prompt_tokens"] + r["completion_tokens"] for run in runs for r in run])
            usd = np.mean([(r["prompt_tokens"] * PRICE[0] + r["completion_tokens"] * PRICE[1]) / 1e6 for run in runs for r in run])
            calls = np.mean([r["n_calls"] for run in runs for r in run])
            crash = np.mean([r["error"] is not None for run in runs for r in run])
            wall = np.mean([r["wall_time"] for run in runs for r in run])
        meta[s] = dict(q=corr.mean(0), run_acc=corr.mean(1), tok=tok, usd=usd, calls=calls, crash=crash, wall=wall,
                       stab=np.mean([len({run[i] for run in ans}) == 1 for i in range(Q)]),
                       unparsed=np.mean([a not in "ABCD" or a == "" for run in ans for a in run]))
    print("MedLA pilot — MedQA-US 4-option, questions 0–199, gpt-4o-mini, 3 runs per system")
    print(f"{'system':4s} {'acc':>6s} {'± SD':>6s} {'runs':>20s} {'stable3':>7s} {'unparsed':>8s} {'crash':>6s} "
          f"{'calls/q':>7s} {'tok/q':>8s} {'$/q':>8s} {'wall s/q':>8s}")
    for s, m in meta.items():
        print(f"{s:4s} {m['q'].mean():6.3f} {m['run_acc'].std(ddof=1):6.3f} {' '.join(f'{a:.3f}' for a in m['run_acc']):>20s} "
              f"{m['stab']:7.3f} {m['unparsed']:8.3f} {m['crash']:6.3f} {m['calls']:7.1f} {m['tok']:8.0f} {m['usd']:8.5f} {m['wall']:8.1f}")

    def maj(s):
        out = []
        for i in range(Q):
            c = collections.Counter(run[i] for run in A[s]).most_common()
            out.append(c[0][0] if len(c) == 1 or c[0][1] > c[1][1] else A[s][0][i])   # tie -> run 1
        return np.array([a == g for a, g in zip(out, G)])

    print()
    for a, b, tag in (("S4", "L0", "PRIMARY"), ("L1", "L0", "secondary"), ("L1", "S4", "descriptive")):
        d = meta[b]["q"] - meta[a]["q"]
        bs = np.array([d[RNG.integers(0, Q, Q)].mean() for _ in range(10000)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        ma, mb = maj(a), maj(b)
        n01, n10 = int((~ma & mb).sum()), int((ma & ~mb).sum())
        p = stats.binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
        se = d.std(ddof=1) / np.sqrt(Q)
        tost = {dl: max(stats.t.sf((d.mean() + dl) / se, Q - 1), stats.t.cdf((d.mean() - dl) / se, Q - 1))
                for dl in (0.01, 0.02, 0.03, 0.04, 0.05)}
        print(f"[{tag}] {b} − {a} = {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]  McNemar(majority) p={p:.2g} "
              f"({b} only {n01}, {a} only {n10})")
        print("    TOST p: " + " ".join(("*" if dl == 0.03 else "") + f"δ{dl}:{v:.3f}" for dl, v in tost.items()))
    print(f"\ncost ratio L0/S4: tokens {meta['L0']['tok'] / meta['S4']['tok']:.1f}x, $ {meta['L0']['usd'] / meta['S4']['usd']:.1f}x")


if __name__ == "__main__":
    main()
