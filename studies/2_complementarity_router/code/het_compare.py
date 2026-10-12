"""Descriptive SC-k* vs HET-k* vs M1 comparison on PILOT (requested alongside the Amendment 1 diagnostic).
Per question: SC = p_SC(k*) from the 60-sample SC pool (runs 1–3), HET = same estimator on the 63-sample HET
pool, M1 = mean over 3 runs, S = mean over 3 runs. Paired bootstrap over questions (10,000, seed 0).
Also the share of question-runs with well-formed 5 + 2 MedAgents recruitment (in M1 and in HET).

  python studies/2_complementarity_router/code/het_compare.py   ->  studies/2_complementarity_router/results/pilot/HET_COMPARE.txt
"""
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import oracle_analysis as OA    # noqa: E402
import pilot_analysis as P      # noqa: E402
from het_analysis import K_STAR  # noqa: E402

WELL = lambda d: 0 < len(d) <= 60 and "\n" not in d and not re.search(r"\b(the|question|patient|answer|let's|I)\b", d, re.I)


def main():
    lines = []
    for tag, _ in P.MODELS:
        rng = np.random.default_rng(0)
        S, SC, M, cost = P.load(tag)
        H = [P.jl(f"{P.BASE}/{tag}/HET/run{r}.jsonl") for r in (1, 2, 3)]
        gold = np.array([r["gold"] for r in S[0]])
        k = K_STAR[tag]
        s = np.array([[r["E3"] == r["gold"] for r in run] for run in S], float).mean(0)
        m = np.array([[r["pred_answer"] == r["gold_answer"] for r in run] for run in M], float).mean(0)
        sc = OA.p_sc(np.concatenate([np.array([r["votes"] for r in run]) for run in SC[:3]], 1), gold, k, rng)
        het = OA.p_sc(np.concatenate([np.array([r["votes"] for r in run]) for run in H], 1), gold, k, rng)
        tS = np.mean([r["S_prompt_tokens"] + r["S_completion_tokens"] for run in S for r in run])
        tSC = k * np.mean([(r["prompt_tokens"] + r["completion_tokens"]) / 20 for run in SC[:3] for r in run])
        tH = k * np.mean([(r["prompt_tokens"] + r["completion_tokens"]) / 21 for run in H for r in run])
        tM = np.mean([c["prompt_tokens"] + c["completion_tokens"] for c in cost])
        mrec = [r for run in M for r in run if isinstance(r.get("question_domains"), list)]
        wm = np.mean([len(r["question_domains"]) == 5 and len(r["option_domains"]) == 2
                      and all(map(WELL, r["question_domains"] + r["option_domains"])) for r in mrec])
        wh = np.mean([len(r["domains"]) == 7 and all(map(WELL, r["domains"])) for run in H for r in run])
        f = lambda d: "{:+.1f} pp [{:+.1f}, {:+.1f}]".format(*(100 * np.array(OA.boot_ci(d, rng))))
        lines += [f"=== {tag}  (k* = {k}) ===",
                  f"accuracy (expected, per question):  S {s.mean():.3f}  SC-{k} {sc.mean():.3f}  HET-{k} {het.mean():.3f}  M1 {m.mean():.3f}",
                  f"tokens / question:                  S {tS:.0f}  SC-{k} {tSC:.0f}  HET-{k} {tH:.0f}  M1 {tM:.0f}",
                  f"HET − SC   {f(het - sc)}    (role heterogeneity)",
                  f"M1 − HET   {f(m - het)}    (interaction / orchestration beyond heterogeneity)",
                  f"M1 − SC    {f(m - sc)}",
                  f"well-formed 5+2 recruitment: M1 {wm:.1%} of question-runs, HET {wh:.1%}", ""]
    open(f"{P.BASE}/HET_COMPARE.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
