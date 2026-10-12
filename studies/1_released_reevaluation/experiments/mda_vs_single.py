"""MDAgents F0–F3 vs strong single S (P0 epoch, zero-shot CoT + E3) on MedQA questions 0–199.
Per question, a condition's score = mean correctness over its 5 runs; paired bootstrap over
questions (10k, seed 0). Two scorers: pre-registered (letter only) and sensitivity (+ option text).

  python experiments/mda_vs_single.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from mda_analyze import extract, load_runs   # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
RNG = np.random.default_rng(0)
S1 = np.array([int(json.loads(l)["E3"] == json.loads(l)["gold"]) for l in open(f"{ROOT}/runs/p0_extraction.jsonl")][:200])
S_RUNS = np.array([[int(json.loads(l)["E3"] == json.loads(l)["gold"]) for l in open(f"{ROOT}/runs/mda/S5/run{r}.jsonl")]
                   for r in range(1, 6)])
S = S_RUNS.mean(0)          # per-question mean over 5 runs (same design as F0–F3)
E3 = [[json.loads(l)["E3"] for l in open(f"{ROOT}/runs/mda/S5/run{r}.jsonl")] for r in range(1, 6)]


def per_question(cond, option_text):
    runs = load_runs(cond)
    M = []
    for rows in runs.values():
        M.append([int(rows[i]["error"] is None and
                      extract(rows[i]["response"], rows[i]["options"] if option_text else None)[0] == rows[i]["label"])
                  for i in range(200)])
    return np.array(M).mean(0)


accs = S_RUNS.mean(1)
same = np.mean([len({E3[r][i] for r in range(5)}) == 1 for i in range(200)])
print(f"S strong single x5 (questions 0–199): acc {accs.mean():.3f} ± SD {accs.std(ddof=1):.3f} "
      f"(runs {', '.join(f'{a:.3f}' for a in accs)}); identical answer in all 5 runs {same:.3f}; "
      f"P0 single realization {S1.mean():.3f}\n")
for option_text in (False, True):
    print("scorer:", "sensitivity (+option text)" if option_text else "pre-registered (letter only)")
    for c in ("F0", "F1", "F2_final_v2", "F3_final_v2"):
        m = per_question(c, option_text)
        d = m - S
        bs = np.array([d[RNG.integers(0, 200, 200)].mean() for _ in range(10000)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"  {c:12s} acc {m.mean():.3f}   MAS − S = {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
    print()
