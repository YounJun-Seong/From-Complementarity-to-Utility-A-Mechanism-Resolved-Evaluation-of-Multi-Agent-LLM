"""Phase V §5 G2 check: calling the unchanged Phase II `oracle_analysis.analyze` with a constant feature matrix
reproduces the frozen Phase II PILOT H, H_plc and H − H_plc (Qwen2.5-14B) exactly. No LLM calls.

  python studies/5_medarc_transfer/code/test_c_star_reproduction.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
sys.path.insert(0, f"{ROOT}/studies/2_complementarity_router/code")
import oracle_analysis as OA    # noqa: E402
import pilot_analysis as P      # noqa: E402


def main():
    tag = "Qwen2.5-14B-Instruct"
    frozen = json.load(open(f"{ROOT}/studies/2_complementarity_router/results/pilot/RESULTS.json"))[tag]
    S, SC, M, cost = P.load(tag)
    gold = np.array([r["gold"] for r in S[0]])
    Sm = np.array([[r["E3"] == r["gold"] for r in run] for run in S], float)
    Mm = np.array([[r["pred_answer"] == r["gold_answer"] for r in run] for run in M], float)
    votes = [np.array([r["votes"] for r in run]) for run in SC]
    tok = frozen["tok"]
    res = OA.analyze(Sm, np.concatenate(votes[:3], axis=1), gold, Mm, tok, np.zeros((len(gold), 2)),
                     placebo_votes=np.concatenate(votes[3:], axis=1), pool_split=40, k=frozen["k_star"])
    ok = all(np.allclose(res[k], frozen[k]) for k in ("H", "H_plc", "H_minus_H_plc"))
    for k in ("H", "H_plc", "H_minus_H_plc"):
        print(f"{k:14s} frozen {np.round(frozen[k], 6)}  recomputed {np.round(res[k], 6)}")
    print("C* REPRODUCTION", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
