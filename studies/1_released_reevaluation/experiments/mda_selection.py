"""Zero-cost selection-effect check: strong single S (per-question mean over 5 runs) on the
questions that MDAgents routed / discussed differently. Unit = MDAgents run x question.

  python experiments/mda_selection.py F2_final_v2 F3_final_v2
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from mda_analyze import extract, load_runs   # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
S = np.array([[int(json.loads(l)["E3"] == json.loads(l)["gold"]) for l in open(f"{ROOT}/runs/mda/S5/run{r}.jsonl")]
              for r in range(1, 6)]).mean(0)


def show(label, units):
    if not units:
        return
    s = np.mean([S[i] for i, _ in units]); m = np.mean([c for _, c in units])
    print(f"  {label:34s} n={len(units):4d}  S {s:.3f}  MDAgents {m:.3f}  gap {m - s:+.3f}")


for cond in sys.argv[1:] or ["F2_final_v2", "F3_final_v2"]:
    rows = [r for rr in load_runs(cond).values() for r in rr.values()]
    unit = lambda r: (r["idx"], int(r["error"] is None and extract(r["response"])[0] == r["label"]))
    print(f"=== {cond} (pre-registered scorer; crash = wrong) ===")
    for route in ("basic", "intermediate", "advanced"):
        show(f"route={route}", [unit(r) for r in rows if r["difficulty"] == route])
    inter = [r for r in rows if r["difficulty"] == "intermediate" and r["error"] is None]
    show("intermediate, noAns (no discussion)", [unit(r) for r in inter if r["noAns"]])
    show("intermediate, discussion triggered", [unit(r) for r in inter if not r["noAns"]])
    show("all", [unit(r) for r in rows])
    print()
