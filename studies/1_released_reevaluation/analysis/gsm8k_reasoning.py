"""E-① reasoning (non-MCQA) + positive control + self-consistency, on GSM8K math —
multi-agent debate's home turf (Du et al. 2023 used arithmetic).
Methods on the SAME questions:
  single_cot        : 1 CoT sample (temp 0)                     -> the standard single baseline
  self_consistency  : k CoT samples (temp 0.7), majority vote    -> cheap ensemble; POSITIVE CONTROL
  debate            : 3 agents x 2 rounds (temp 0.7), majority   -> multi-agent on math
If self_consistency >> single_cot (a known-true effect), our harness demonstrably DETECTS a
real aggregation gain -> the multi-agent null is not measurement insensitivity. If debate only
ties single/SC, the reality-check extends to open-ended reasoning.
"""
import argparse
import json
import os
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import llm
from llm import chat, sample_n

RNG = np.random.default_rng(0)
K_SC = 10          # self-consistency samples
N_AG = 3           # debate agents
ROUNDS = 2         # debate rounds


def gold(ans):
    return num(ans.split("####")[-1])


def num(t):
    t = t.replace(",", "")
    m = re.findall(r"-?\d+\.?\d*", t)
    return m[-1].rstrip(".").rstrip("0").rstrip(".") if m else ""


def norm(x):
    try:
        return str(int(float(x)))
    except Exception:
        return str(x)


COT = ("Solve the problem step by step. End your answer with a line "
       "'The answer is <number>'.")


def extract_final(t):
    m = re.search(r"answer is\s*\$?(-?[\d,]+\.?\d*)", t, re.I)
    return norm(num(m.group(1))) if m else norm(num(t))


def single_cot(q):
    t = chat([{"role": "system", "content": COT}, {"role": "user", "content": q}],
             json_mode=False, temperature=0.0, max_tokens=600)
    return extract_final(t)


def self_consistency(q, k=K_SC):
    outs = sample_n([{"role": "system", "content": COT}, {"role": "user", "content": q}],
                    k=k, temperature=0.7, max_tokens=600, seed=0)
    votes = [extract_final(t) for t in outs]
    votes = [v for v in votes if v]
    return Counter(votes).most_common(1)[0][0] if votes else ""


def debate(q, n_agents=N_AG, rounds=ROUNDS):
    # round 0: diverse initial solutions
    sols = sample_n([{"role": "system", "content": COT}, {"role": "user", "content": q}],
                    k=n_agents, temperature=0.7, max_tokens=600, seed=0)
    for r in range(rounds):
        new = []
        for i in range(n_agents):
            others = "\n\n".join(f"Agent {j+1} solution:\n{sols[j][:900]}"
                                 for j in range(n_agents) if j != i)
            msg = (f"Problem: {q}\n\nYour previous solution:\n{sols[i][:900]}\n\n"
                   f"Other agents' solutions:\n{others}\n\nUsing these, reconsider carefully "
                   f"and give an updated step-by-step solution. End with 'The answer is <number>'.")
            out = sample_n([{"role": "system", "content": COT}, {"role": "user", "content": msg}],
                           k=1, temperature=0.7, max_tokens=600, seed=i + 1)[0]
            new.append(out)
        sols = new
    votes = [extract_final(s) for s in sols]
    votes = [v for v in votes if v]
    return Counter(votes).most_common(1)[0][0] if votes else ""


METHODS = {"single_cot": single_cot, "self_consistency": self_consistency, "debate": debate}


def ci(x):
    b = [RNG.choice(x, len(x), replace=True).mean() for _ in range(5000)]
    return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def gain(a, b):
    d = b - a
    bs = [RNG.choice(d, len(d), replace=True).mean() for _ in range(5000)]
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test").select(range(args.n))
    Q = [r["question"] for r in ds]
    G = [norm(gold(r["answer"])) for r in ds]

    res = {}
    for name, fn in METHODS.items():
        with ThreadPoolExecutor(max_workers=8) as ex:
            preds = list(ex.map(fn, Q))
        res[name] = np.array([int(norm(p) == g) for p, g in zip(preds, G)])
        m, lo, hi = ci(res[name])
        print(f"  {name:18s} acc={m:.3f} [{lo:.3f},{hi:.3f}]", flush=True)

    print("\n  --- gains ---")
    g, lo, hi = gain(res["single_cot"], res["self_consistency"])
    print(f"  self_consistency - single = {g:+.3f} [{lo:+.3f},{hi:+.3f}]  "
          f"{'POSITIVE CONTROL PASSED (detects real gain)' if lo > 0 else 'n.s.'}")
    g, lo, hi = gain(res["single_cot"], res["debate"])
    print(f"  debate - single           = {g:+.3f} [{lo:+.3f},{hi:+.3f}]  "
          f"{'debate helps' if lo > 0 else 'TIE (no debate gain)'}")
    g, lo, hi = gain(res["debate"], res["self_consistency"])
    print(f"  self_consistency - debate = {g:+.3f} [{lo:+.3f},{hi:+.3f}]  "
          f"{'cheap SC beats debate' if lo > 0 else 'tie'}")
    print("GSM8K_DONE")


if __name__ == "__main__":
    main()
