"""Layer 2A report (no LLM calls): per Qwen model, S vs M1 on MedQA 0–199, 3 runs each.
Per-question score = mean over 3 runs; paired bootstrap over questions (10k, seed 0).
Crash / unparsed = wrong (end-to-end). Also completion / crash rate, answer stability
(identical answer in all 3 runs), M1 finish=length share of calls, calls / tokens per question.
Writes runs/layer2/2A/summary.csv (one row per model) for the capability figure.

  python experiments/layer2_2a_report.py
"""
import glob
import json
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
BASE = f"{ROOT}/runs/layer2/2A"
MODELS = ["Qwen2.5-1.5B-Instruct", "Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct",
          "Llama-3.1-8B-Instruct"]
RNG = np.random.default_rng(0)
Q = 200


def load_s(tag):
    runs = []
    for r in (1, 2, 3):
        f = f"{BASE}/{tag}/S/run{r}.jsonl"
        if not os.path.exists(f):
            return None
        runs.append([json.loads(l) for l in open(f)][:Q])
    return runs


def load_m1(tag):
    runs = []
    for r in (1, 2, 3):
        by = {}
        for d in glob.glob(f"{BASE}/{tag}/M1/run{r}/shard*"):
            s = int(os.path.basename(d)[5:])
            f = f"{d}/chatgpt-syn_verif"
            if os.path.exists(f):
                for k, l in enumerate(open(f)):
                    by[s * 20 + k] = json.loads(l)
        if len(by) < Q:
            return None
        runs.append([by[i] for i in range(Q)])
    return runs


def main():
    rows = []
    hdr = (f"{'model':22s} {'S acc':>6s} {'M1 acc':>7s} {'M1−S [95% CI]':>24s} {'crash':>6s} {'compl':>6s} "
           f"{'stab S':>6s} {'stab M1':>7s} {'M1 len%':>7s} {'M1 calls/q':>10s} {'M1 tok/q':>9s}")
    print(hdr)
    for tag in MODELS:
        S, M = load_s(tag), load_m1(tag)
        if S is None or M is None:
            print(f"{tag:22s} pending (S {'ok' if S else 'missing'}, M1 {'ok' if M else 'incomplete'})")
            continue
        s = np.array([[r["E3"] == r["gold"] for r in run] for run in S], float).mean(0)
        m = np.array([[r["pred_answer"] == r["gold_answer"] for r in run] for run in M], float).mean(0)
        d = m - s
        bs = np.array([d[RNG.integers(0, Q, Q)].mean() for _ in range(10000)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        crash = np.mean([r.get("error") is not None for run in M for r in run])
        stab_s = np.mean([len({run[i]["E3"] for run in S}) == 1 for i in range(Q)])
        stab_m = np.mean([len({run[i]["pred_answer"] for run in M}) == 1 for i in range(Q)])
        # per-call stats for M1 from the cache namespace of this model
        calls = tok = length = 0
        logs = glob.glob(f"{BASE}/{tag}/M1/run*/shard*/cost.jsonl")
        if logs:   # per-question COST_LOG (runs launched after it was added); no cache scan
            C = [json.loads(l) for f in logs for l in open(f)]
            calls, tok, length = sum(c["calls"] for c in C), sum(c["prompt_tokens"] + c["completion_tokens"] for c in C), float("nan")
        prev = {}
        if not logs and os.path.exists(f"{BASE}/summary.csv"):   # reuse per-call stats computed earlier (cache scan is slow)
            import csv
            prev = {r["model"]: r for r in csv.DictReader(open(f"{BASE}/summary.csv"))}.get(tag, {})
        if prev:
            n = 3 * Q
            calls, tok = float(prev["M1_calls_q"]) * n, float(prev["M1_tok_q"]) * n
            length = float(prev["M1_length_share"]) * calls
        for f in ([] if (logs or prev) else glob.glob(f"{ROOT}/data/llm_cache/*.json")):
            t = open(f).read()
            if f'"nonce": "2A:{tag}:M1:' not in t:
                continue
            rec = json.loads(t)
            calls += 1; length += rec.get("finish_reason") == "length"
            u = rec.get("usage") or {}; tok += u.get("prompt_tokens", 0) + u.get("completion_tokens", 0)
        n_q = 3 * Q
        row = dict(model=tag, S_acc=s.mean(), M1_acc=m.mean(), gap=d.mean(), ci_lo=lo, ci_hi=hi, crash=crash,
                   completion=1 - crash, stab_S=stab_s, stab_M1=stab_m, M1_length_share=(length / max(calls, 1)) if length == length else float("nan"),
                   M1_calls_q=calls / n_q, M1_tok_q=tok / n_q)
        rows.append(row)
        print(f"{tag:22s} {s.mean():6.3f} {m.mean():7.3f} {d.mean():+7.3f} [{lo:+.3f},{hi:+.3f}] {crash:6.3f} {1 - crash:6.3f} "
              f"{stab_s:6.3f} {stab_m:7.3f} {row['M1_length_share']:7.3f} {calls / n_q:10.1f} {tok / n_q:9.0f}")
    if rows:
        with open(f"{BASE}/summary.csv", "w") as f:
            f.write(",".join(rows[0]) + "\n")
            for r in rows:
                f.write(",".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r.values()) + "\n")
        print("\ncache calls are unique generations per namespace (cached repeats within a run are counted once)")


if __name__ == "__main__":
    main()
