"""P0-3: split the base_cot "answer-completion failure" into mutually exclusive causes
(defined on the short-budget run), then show accuracy at short vs long budget per cause.

  length        API finish_reason == "length"
  no_answer     natural stop, no "Answer:" line at all
  malformed     natural stop, "Answer" line present but no single clean A-E letter
  extract_fail  natural stop, clean answer letter L, but MedAgents' extractor != L
  clean         natural stop, extractor == L

  python experiments/p0_truncation.py --role empty --short 300 --long 700
"""
import argparse
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
ANS = re.compile(r"Answer\s*[:：]\s*\**\s*\(?\[?([A-E])\]?\)?(?![A-Za-z])")
CATS = ["length", "no_answer", "malformed", "extract_fail", "clean"]


def load(path):
    return [json.loads(l) for l in open(path)]


def category(r):
    if r["finish_reason"] == "length":
        return "length"
    out = r["output"] or ""
    m = ANS.findall(out)
    if not m:
        return "malformed" if re.search(r"Answer\s*[:：]", out.split("Thought", 1)[-1]) else "no_answer"
    return "clean" if r["pred"] == m[-1] else "extract_fail"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", default="empty")
    ap.add_argument("--short", type=int, default=300)
    ap.add_argument("--long", type=int, default=700)
    ap.add_argument("--dir", default=os.path.join(ROOT, "runs/p0_sweep"))
    ap.add_argument("--with-m1", action="store_true", help="add the epoch MedAgents rerun column")
    a = ap.parse_args()
    S = load(f"{a.dir}/{a.role}_{a.short}.jsonl")
    L = load(f"{a.dir}/{a.role}_{a.long}.jsonl")
    M = None
    if a.with_m1:
        sys.path.insert(0, os.path.dirname(__file__))
        from p0_stats import medagents_rerun
        m = medagents_rerun()
        M = [{"pred": int(x), "gold": 1} for x in m] if m is not None else None
    cat = np.array([category(r) for r in S])
    cs = np.array([r["pred"] == r["gold"] for r in S])
    cl = np.array([r["pred"] == r["gold"] for r in L])
    cm = np.array([r["pred"] == r["gold"] for r in M]) if M else None
    print(f"role={a.role}  budget {a.short} -> {a.long}  (n={len(S)})")
    hdr = f"{'cause':13s} {'n':>5s} {'acc@'+str(a.short):>8s} {'acc@'+str(a.long):>8s} {'recovery':>9s}"
    print(hdr + (f" {'MedAgents':>9s}" if M else ""))
    for c in CATS + ["ALL"]:
        k = np.ones(len(S), bool) if c == "ALL" else cat == c
        if not k.any():
            continue
        line = f"{c:13s} {k.sum():5d} {cs[k].mean():8.3f} {cl[k].mean():8.3f} {cl[k].mean() - cs[k].mean():+9.3f}"
        print(line + (f" {cm[k].mean():9.3f}" if M else ""))
    gain = cl.sum() - cs.sum()
    aff = cat != "clean"
    print(f"\nnet gain {a.short}->{a.long}: {gain:+d} questions; "
          f"on affected (non-clean) {int(cl[aff].sum() - cs[aff].sum()):+d}, on clean {int(cl[~aff].sum() - cs[~aff].sum()):+d}")
    if M:
        g = cm.astype(int) - cs.astype(int)
        print(f"MedAgents net gain over {a.role}@{a.short}: {g.sum():+d}; "
              f"on affected {g[aff].sum():+d}, on clean {g[~aff].sum():+d}")


if __name__ == "__main__":
    main()
