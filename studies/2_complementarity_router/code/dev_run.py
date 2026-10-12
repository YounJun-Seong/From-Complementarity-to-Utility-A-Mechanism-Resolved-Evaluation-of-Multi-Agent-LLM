"""DEV code validation (PHASE2_FREEZE.md §3.2, §3.6). Runs the full analysis on the Part I first-200
outputs (S x3, SC-20 x3, M1 x3) for each open-weight model. Placebo is unavailable on DEV (no extra SC
runs). Numbers are exploratory and play no role in any Phase II decision; this script prints only
pipeline checks and writes the numbers to studies/2_complementarity_router/results/dev/.

  python studies/2_complementarity_router/code/dev_run.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "1_released_reevaluation", "experiments"))
import features as F                      # noqa: E402
import oracle_analysis as OA              # noqa: E402
import layer2_2a_report as A              # noqa: E402
import contextlib, io                     # noqa: E402,E401
with contextlib.redirect_stdout(io.StringIO()):
    import pareto                         # noqa: E402  (prints its table at import)

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
OUT = f"{ROOT}/studies/2_complementarity_router/results/dev"
MODELS = ["Qwen2.5-14B-Instruct", "Llama-3.1-8B-Instruct", "Qwen2.5-7B-Instruct"]
Q = 200


def jl(p):
    return [json.loads(l) for l in open(p)]


def main():
    os.makedirs(OUT, exist_ok=True)
    qs = [r["question"] for r in jl(f"{ROOT}/data/splits/MedQA/test.jsonl")][:Q]
    ok = True
    for tag in MODELS:
        S, M = A.load_s(tag), A.load_m1(tag)
        sc = [jl(f"{ROOT}/studies/1_released_reevaluation/runs/layer2/2D/{tag}/sc-20/run{r}.jsonl") for r in (1, 2, 3)]
        gold = np.array([r["gold"] for r in S[0]])
        checks = {
            "S/M1/SC aligned on gold": all(r["gold"] == g for run in S for r, g in zip(run, gold))
            and all(r["gold_answer"] == g for run in M for r, g in zip(run, gold))
            and all(r["gold"] == g for run in sc for r, g in zip(run, gold)),
            "SC runs have 20 votes": all(len(r["votes"]) == 20 for run in sc for r in run),
        }
        Sm = np.array([[r["E3"] == r["gold"] for r in run] for run in S], float)
        Mm = np.array([[r["pred_answer"] == r["gold_answer"] for r in run] for run in M], float)
        votes = np.concatenate([np.array([r["votes"] for r in run]) for run in sc], axis=1)   # Q x 60
        tok = dict(S=np.mean([r["S_prompt_tokens"] + r["S_completion_tokens"] for run in S for r in run]),
                   SC=np.mean([(r["prompt_tokens"] + r["completion_tokens"]) / 20 for run in sc for r in run]),
                   M=pareto.m1_tokens(tag))
        X = F.build(S[0], votes[:, :4], list(range(Q)), qs)
        checks["features finite"] = bool(np.isfinite(X).all())
        res = OA.analyze(Sm, votes, gold, Mm, tok, X, placebo_votes=None)
        checks["types sum to 1"] = abs(sum(res["types"].values()) - 1) < 1e-9
        checks["H finite"] = bool(np.isfinite(res["H"][0]))
        checks["placebo skipped -> G2 fails"] = res["H_minus_H_plc"] is None and not res["pass_G2"]
        json.dump(dict(model=tag, tok=tok, note="DEV, exploratory, no decision role", **res),
                  open(f"{OUT}/{tag}.json", "w"), indent=1, default=float)
        print(f"{tag:22s} k*={res['k_star']:2d}  " + "  ".join(f"{k}: {'ok' if v else 'FAIL'}" for k, v in checks.items()))
        ok &= all(checks.values())
    print("DEV PIPELINE", "PASS" if ok else "FAIL", f"(numbers in {os.path.relpath(OUT, ROOT)}/, not inspected)")


if __name__ == "__main__":
    main()
