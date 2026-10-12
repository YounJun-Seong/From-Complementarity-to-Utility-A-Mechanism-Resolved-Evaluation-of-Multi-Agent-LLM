"""Zero-cost noAns analysis on existing MDAgents logs (no API calls): is the released
collaboration-control-flow defect (moderator receives final_answer=None) associated with errors?

  python experiments/mda_noans.py F2_final_v2 [F3_final_v2 ...]
"""
import collections
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from mda_analyze import extract, load_runs   # noqa: E402


def analyze(cond):
    runs = load_runs(cond)
    rows = [r for rr in runs.values() for r in rr.values() if r["error"] is None]
    inter = [r for r in rows if r["difficulty"] == "intermediate"]
    for scorer, opt in (("pre-registered", False), ("+option text", True)):
        c = lambda r: extract(r["response"], r["options"] if opt else None)[0] == r["label"]
        na = np.array([c(r) for r in inter if r["noAns"]])
        ok = np.array([c(r) for r in inter if not r["noAns"]])
        a, b = (~na).sum() + .5, na.sum() + .5          # errors / correct with noAns (Haldane)
        cc, d = (~ok).sum() + .5, ok.sum() + .5
        print(f"{cond} [{scorer}] intermediate completions {len(inter)}: noAns {len(na)} ({len(na)/len(inter):.0%})")
        print(f"   acc with noAns {na.mean():.3f} (n={len(na)}) | without {ok.mean():.3f} (n={len(ok)}) | "
              f"odds ratio of error {(a / b) / (cc / d):.2f}")
    # per question: number of runs (of 5) with noAns vs accuracy over intermediate-routed runs
    by_q = collections.defaultdict(list)
    for r in inter:
        by_q[r["idx"]].append((r["noAns"], extract(r["response"])[0] == r["label"]))
    strata = collections.defaultdict(list)
    for v in by_q.values():
        strata[sum(x for x, _ in v)].append(np.mean([y for _, y in v]))
    print("   questions by #runs with noAns -> mean accuracy of their intermediate runs:",
          {k: (len(v), round(float(np.mean(v)), 3)) for k, v in sorted(strata.items())})
    other = [r for r in rows if r["difficulty"] != "intermediate"]
    print(f"   for reference: basic+advanced completions {len(other)}, "
          f"acc {np.mean([extract(r['response'])[0] == r['label'] for r in other]):.3f}\n")


for c in sys.argv[1:] or ["F2_final_v2"]:
    analyze(c)
