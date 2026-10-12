"""Layer 2C report (no LLM calls): M1-original (2A) vs M1-expanded (2C) vs S (2A), Qwen2.5 7B/14B,
MedQA 0–199, 3 runs each. Question-level paired bootstrap (10k, seed 0). Crash / unparsed = wrong.
Per call type: truncation rate (finish_reason=length), from the cache namespaces 2A:<m>:M1 / 2C:<m>:M1x.

  python experiments/layer2_2c_report.py
"""
import collections
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import layer2_2a_report as A   # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
RNG = np.random.default_rng(0)
Q = 200
MODELS = ["Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct"]


def load_m1(base, tag):
    old = A.BASE
    A.BASE = base
    try:
        return A.load_m1(tag)
    finally:
        A.BASE = old


def ctype(r):
    mt = r["request"]["max_tokens"]
    if mt in (50, 150): return "domain classification"
    if mt in (30, 75): return "vote"
    if mt in (300, 700): return "expert analysis"
    if mt in (500, 1000): return "advice"
    return "synthesis / answer / revision"


def call_stats(prefix):
    c = collections.defaultdict(collections.Counter); tot = collections.Counter()
    for f in glob.glob(f"{ROOT}/data/llm_cache/*.json"):
        s = open(f).read()
        if f'"nonce": "{prefix}' not in s:
            continue
        r = json.loads(s); t = ctype(r)
        c[t]["n"] += 1; c[t]["len"] += r.get("finish_reason") == "length"
        u = r.get("usage") or {}; tot["calls"] += 1
        tot["tok"] += u.get("prompt_tokens", 0) + u.get("completion_tokens", 0)
    return c, tot


def boot(d):
    bs = np.array([d[RNG.integers(0, Q, Q)].mean() for _ in range(10000)])
    return np.percentile(bs, [2.5, 97.5])


def main():
    for tag in MODELS:
        S = A.load_s(tag); Mo = load_m1(A.BASE, tag); Mx = load_m1(f"{ROOT}/runs/layer2/2C", tag)
        if Mx is None:
            print(f"{tag}: M1-expanded incomplete"); continue
        s = np.array([[r["E3"] == r["gold"] for r in run] for run in S], float).mean(0)
        acc = lambda M: np.array([[r["pred_answer"] == r["gold_answer"] for r in run] for run in M], float).mean(0)
        mo, mx = acc(Mo), acc(Mx)
        crash = lambda M: np.mean([r.get("error") is not None for run in M for r in run])
        print(f"=== {tag} ===")
        print(f"  S {s.mean():.3f} | M1-original {mo.mean():.3f} (crash {crash(Mo):.3f}) | "
              f"M1-expanded {mx.mean():.3f} (crash {crash(Mx):.3f})")
        for lab, d in (("M1-expanded − S", mx - s), ("M1-expanded − M1-original", mx - mo), ("M1-original − S", mo - s)):
            lo, hi = boot(d)
            print(f"  {lab:26s} {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
        for name, prefix in (("M1-original", f"2A:{tag}:M1:"), ("M1-expanded", f"2C:{tag}:M1x:")):
            c, tot = call_stats(prefix)
            per = ", ".join(f"{t} {v['len'] / v['n']:.0%}" for t, v in sorted(c.items()))
            print(f"  {name:12s} calls/q {tot['calls'] / (3 * Q):.1f}, tokens/q {tot['tok'] / (3 * Q):.0f}; truncated: {per}")
        print()


if __name__ == "__main__":
    main()
