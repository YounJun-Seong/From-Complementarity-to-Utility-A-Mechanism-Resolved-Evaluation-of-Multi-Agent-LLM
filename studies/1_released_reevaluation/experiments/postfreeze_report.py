"""Post-freeze report (POSTFREEZE_ROBUSTNESS.md; no LLM calls): R1 random subsets, R2 MedMCQA, R5 SC-7.
Paired bootstrap over questions (10k, seed 0); crash / unparsed = wrong. Missing pieces print 'pending'.

  python experiments/postfreeze_report.py
"""
import glob
import json
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
PF = f"{ROOT}/runs/postfreeze"
RNG = np.random.default_rng(0)
MODELS = ["Qwen2.5-14B-Instruct", "Llama-3.1-8B-Instruct"]


def jl(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else None


def ci(d):
    bs = np.array([d[RNG.integers(0, len(d), len(d))].mean() for _ in range(10000)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return f"{d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]"


def cci(d, qid, B=10000):
    """Cluster bootstrap over unique question ids (repeated occurrences stay together)."""
    ids = np.unique(qid)
    groups = [np.flatnonzero(qid == u) for u in ids]
    sums = np.array([d[g].sum() for g in groups]); counts = np.array([len(g) for g in groups])
    bs = []
    for _ in range(B):
        k = RNG.integers(0, len(ids), len(ids))
        bs.append(sums[k].sum() / counts[k].sum())
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return f"{d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]"


def m1(d, n):
    by = {}
    for sd in glob.glob(f"{d}/shard*"):
        s = int(os.path.basename(sd)[5:])
        f = f"{sd}/chatgpt-syn_verif"
        if os.path.exists(f):
            for k, l in enumerate(open(f)):
                by[s * 20 + k] = json.loads(l)
    if len(by) < n:
        return None, None
    rows = [by[i] for i in range(n)]
    C = [json.loads(l) for f in glob.glob(f"{d}/shard*/cost.jsonl") for l in open(f)]
    return rows, C


def tokline(name, rows, key_in, key_out, calls):
    ti = np.mean([r[key_in] for r in rows]); to = np.mean([r[key_out] for r in rows])
    return f"    {name:6s} input {ti:8.0f}  output {to:7.0f}  total {ti + to:8.0f}  calls {calls:5.1f}"


def main():
    for tag in MODELS:
        print(f"\n=== {tag} ===")
        # R1
        pooled = {"S": [], "M1": [], "SC5": []}
        for sub in "ABC":
            S = jl(f"{PF}/R1/{tag}/S_{sub}.jsonl"); SC = jl(f"{PF}/R1/{tag}/SC5_{sub}.jsonl")
            M, _ = m1(f"{PF}/R1/{tag}/M1_{sub}", 200)
            if S is None or M is None:
                print(f"  R1 subset {sub}: pending"); continue
            s = np.array([r["E3"] == r["gold"] for r in S], float)
            m = np.array([r["pred_answer"] == r["gold_answer"] for r in M], float)
            line = f"  R1 subset {sub}: S {s.mean():.3f}  M1 {m.mean():.3f}  M1 − S {ci(m - s)}"
            pooled["S"].append(s); pooled["M1"].append(m)
            if SC is not None:
                c = np.array([r["pred"] == r["gold"] for r in SC], float)
                pooled["SC5"].append(c)
                line += f"  | SC-5 {c.mean():.3f}  SC-5 − M1 {ci(c - m)}"
            crash = sum(r.get("error") is not None for r in M)
            print(line + f"  (M1 crash {crash})")
        if len(pooled["S"]) == 3:
            # subsets overlap (sampled independently): bootstrap clustered by unique MedQA question id;
            # every occurrence of a question across subsets is resampled together
            qid = np.concatenate([[json.loads(l)["orig_idx"] for l in open(f"{ROOT}/external/MedAgents/datasets/MedQA_rand{x}/test.jsonl")]
                                  for x in "ABC"])
            s, m = np.concatenate(pooled["S"]), np.concatenate(pooled["M1"])
            line = f"  R1 pooled (600 draws, {len(set(qid))} unique questions; cluster bootstrap by question): M1 − S {cci(m - s, qid)}"
            if len(pooled["SC5"]) == 3:
                line += f"  | SC-5 − M1 {cci(np.concatenate(pooled['SC5']) - m, qid)}"
            print(line)
        # R2 MedMCQA
        S = jl(f"{PF}/R2/{tag}/S.jsonl"); SC = jl(f"{PF}/R2/{tag}/SC5.jsonl"); M, C = m1(f"{PF}/R2/{tag}/M1", 500)
        if S is None or M is None or SC is None:
            print("  R2 MedMCQA: pending")
        else:
            s = np.array([r["E3"] == r["gold"] for r in S], float)
            m = np.array([r["pred_answer"] == r["gold_answer"] for r in M], float)
            c = np.array([r["pred"] == r["gold"] for r in SC], float)
            print(f"  R2 MedMCQA (500 q): S {s.mean():.3f}  M1 {m.mean():.3f}  SC-5 {c.mean():.3f}")
            print(f"    M1 − S {ci(m - s)}   SC-5 − M1 {ci(c - m)}   M1 crash {sum(r.get('error') is not None for r in M)}")
            print(tokline("S", S, "S_prompt_tokens", "S_completion_tokens", 2.0))
            print(tokline("SC-5", SC, "prompt_tokens", "completion_tokens", 6.0))
            if C:
                print(tokline("M1", C, "prompt_tokens", "completion_tokens", np.mean([c_["calls"] for c_ in C])))
    # R5 Llama SC-7 vs M1 (first 200, 3 runs)
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    import layer2_2a_report as A
    tag = "Llama-3.1-8B-Instruct"
    R = [jl(f) for f in sorted(glob.glob(f"{ROOT}/runs/layer2/2D/{tag}/sc-7/run*.jsonl"))]
    if len(R) == 3:
        c = np.array([[r["pred"] == r["gold"] for r in run] for run in R], float).mean(0)
        m = np.array([[r["pred_answer"] == r["gold_answer"] for r in run] for run in A.load_m1(tag)], float).mean(0)
        tok = np.mean([r["prompt_tokens"] + r["completion_tokens"] for run in R for r in run])
        print(f"\nR5 Llama-3.1-8B, first 200 x 3: SC-7 {c.mean():.3f} ({tok:.0f} tok/q) vs M1 {m.mean():.3f} (7561 tok/q): "
              f"SC-7 − M1 {ci(c - m)}")


if __name__ == "__main__":
    main()
