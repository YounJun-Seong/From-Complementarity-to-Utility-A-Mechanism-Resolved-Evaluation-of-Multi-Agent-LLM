"""MDAgents bridge-ablation analysis (F0..F3): per run and across runs.

Per run      completion / crash rate (by route), accuracy on ALL questions (crash = wrong) and
             among completed, route distribution, calls / tokens / $ per question (per-model,
             from the logged API calls in the cache, keyed by nonce).
Across runs  accuracy mean ± SD, question-level answer agreement, route agreement,
             basic<->advanced switch rate.

  python experiments/mda_analyze.py F0 [F1 ...]
"""
import collections
import glob
import itertools
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
PRICE = {"gpt-4o-mini-2024-07-18": (0.15, 0.60), "gpt-3.5-turbo": (0.50, 1.50)}   # $/1M in, out
ROUTES = ("basic", "intermediate", "advanced")


def flat(resp):
    if isinstance(resp, dict):
        return "\n".join(flat(v) for v in resp.values())
    return "" if resp is None else str(resp)


# final-answer extraction, most explicit first; within a tier the LAST match wins
TIERS = [
    ("explicit", r"(?:final (?:answer|decision)|answer)\s*(?:is|:)?\s*[:\-]?\s*\**\s*\(([A-E])\)"),
    ("explicit", r"(?:final (?:answer|decision)|answer)\s*(?:is|:)\s*\**\s*([A-E])\b[).:]"),
    ("bold", r"\*\*\s*\(([A-E])\)"),
    ("paren", r"\(([A-E])\)"),
]


def extract(resp, options=None):
    """Pre-registered scorer; with `options` (sensitivity scorer) an answer given only as option
    text is matched to the option whose text is mentioned last in the tail."""
    t = flat(resp)
    tail = t[-600:]
    for src, pat in TIERS:
        m = re.findall(pat, tail, flags=re.I)
        if m:
            return m[-1].upper(), src
    if options:
        low = tail.lower()
        pos = {k: low.rfind(v.lower().rstrip(".")) for k, v in options.items()}
        k, p = max(pos.items(), key=lambda kv: kv[1])
        if p >= 0:
            return k, "option_text"
    return "", "none"


def cache_index(prefixes):
    """nonce -> (model, prompt_tokens, completion_tokens) for logged calls under the prefixes."""
    idx = {}
    for f in glob.glob(f"{ROOT}/data/llm_cache/*.json"):
        s = open(f).read()
        if '"nonce": "mda:' not in s:
            continue
        r = json.loads(s)
        n = r["request"].get("nonce", "")
        if any(n.startswith(p) for p in prefixes):
            u = r.get("usage") or {}
            last = r["request"]["messages"][-1]["content"]
            # released-code logic bug: intermediate moderator gets final_answer=None when no agent
            # asked to talk in round 1 (loop breaks before final answers are collected)
            empty_mod = "by taking majority vote" in last and "\nNone\n\nQuestion:" in last
            idx[n] = (r["request"]["model"], u.get("prompt_tokens", 0), u.get("completion_tokens", 0), empty_mod)
    return idx


def load_runs(cond):
    runs = {}
    for d in sorted(glob.glob(f"{ROOT}/runs/mda/{cond}/run*")):
        rows = {}
        for f in glob.glob(f"{d}/shard*.jsonl"):
            for l in open(f):
                r = json.loads(l)
                rows[r["idx"]] = r
        runs[os.path.basename(d)] = rows
    return runs


def per_question_cost(r, idx):
    key = r.get("nonce_prefix") or (f"mda:run{r['seed']}:rng-{r['rng']}:strict{int(r['strict_parse'])}"
                                    f":unify{int(r['unify'])}:adaptive:{r['idx']}")
    key += ":"
    usd, by_model = 0.0, collections.Counter()
    for k in range(1, r["n_calls"] + 1):
        m, pi, po, empty = idx.get(key + str(k), (None, 0, 0, False))
        if m is None:
            continue
        if empty:
            by_model["EMPTY_MODERATOR_INPUT"] += 1
        pin, pout = PRICE.get(m, PRICE["gpt-4o-mini-2024-07-18"])
        usd += (pi * pin + po * pout) / 1e6
        by_model[m] += 1
    return usd, by_model


def analyze(cond):
    runs = load_runs(cond)
    if not runs:
        print(f"{cond}: no runs"); return
    common = sorted(set.intersection(*(set(v) for v in runs.values())))
    idx = cache_index(["mda:"])
    print(f"=== {cond}: {len(runs)} runs, {len(common)} questions complete in every run ===")
    print(f"{'run':6s} {'n':>4s} {'compl':>6s} {'acc_all':>7s} {'acc_cmp':>7s} {'unparsed':>8s} "
          f"{'basic':>6s} {'inter':>6s} {'adv':>6s} {'crash b/i/a':>12s} {'calls/q':>7s} {'tok/q':>7s} {'$/q':>8s} {'noAns':>5s} models")
    acc_runs, ans, route = [], {}, {}
    for name, rows in runs.items():
        R = [rows[i] for i in common]
        pred = [extract(r["response"]) if r["error"] is None else ("", "crash") for r in R]
        ok = np.array([r["error"] is None for r in R])
        corr = np.array([p == r["label"] for (p, _), r in zip(pred, R)])
        rt = collections.Counter(r["difficulty"] for r in R)
        crash = collections.Counter(r["difficulty"] for r in R if r["error"] is not None)
        costs = [per_question_cost(r, idx) for r in R]
        models = sum((c[1] for c in costs), collections.Counter())
        n_empty = models.pop("EMPTY_MODERATOR_INPUT", 0)
        tok = np.mean([r["prompt_tokens"] + r["completion_tokens"] for r in R])
        unparsed = np.mean([p == "" for (p, s) in pred if s != "crash"]) if ok.any() else 0
        print(f"{name:6s} {len(R):4d} {ok.mean():6.3f} {corr.mean():7.3f} {corr[ok].mean():7.3f} {unparsed:8.3f} "
              f"{rt['basic']:6d} {rt['intermediate']:6d} {rt['advanced']:6d} "
              f"{crash['basic']:>4d}/{crash['intermediate']}/{crash['advanced']:<4d} "
              f"{np.mean([r['n_calls'] for r in R]):7.1f} {tok:7.0f} {np.mean([c[0] for c in costs]):8.5f} {n_empty:5d} "
              f"{dict(models)}")
        acc_runs.append(corr.mean())
        ans[name] = [p for p, _ in pred]
        route[name] = [r["difficulty"] for r in R]
    a = np.array(acc_runs)
    print(f"\naccuracy (crash = wrong): mean {a.mean():.3f} ± SD {a.std(ddof=1) if len(a) > 1 else 0:.3f} "
          f"(min {a.min():.3f}, max {a.max():.3f})")
    if len(runs) > 1:
        names = list(runs)
        pair_ans = np.mean([np.mean([x == y for x, y in zip(ans[p], ans[q])]) for p, q in itertools.combinations(names, 2)])
        pair_rt = np.mean([np.mean([x == y for x, y in zip(route[p], route[q])]) for p, q in itertools.combinations(names, 2)])
        all_ans = np.mean([len({ans[n][i] for n in names}) == 1 for i in range(len(common))])
        all_rt = np.mean([len({route[n][i] for n in names}) == 1 for i in range(len(common))])
        b2a = np.mean([{"basic", "advanced"} <= {route[n][i] for n in names} for i in range(len(common))])
        print(f"answer agreement: pairwise {pair_ans:.3f}, identical in all {len(names)} runs {all_ans:.3f}")
        print(f"route agreement:  pairwise {pair_rt:.3f}, identical in all {len(names)} runs {all_rt:.3f}; "
              f"questions routed both basic and advanced across runs {b2a:.3f}")
    print()


if __name__ == "__main__":
    for c in sys.argv[1:] or ["F0"]:
        analyze(c)
