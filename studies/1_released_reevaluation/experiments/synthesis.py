"""Cross-framework synthesis (no LLM calls), GPT-4o-mini confirmatory layer + Qwen 2A.

 (A) summary table: gap to strong single, gap to released baseline, execution failure, compute
     overhead, main sensitivity.
 (B) collaboration win/loss: questions split by MAS vs strong-single correctness (question-level
     majority over runs); per group, strong-single difficulty (held-out runs where available),
     MAS answer stability, and framework-specific process features.
 (C) difficulty x gain: difficulty bins defined on held-out strong-single runs, gap measured with
     the remaining runs (guards against regression to the mean).

  python experiments/synthesis.py
"""
import collections
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
ROOT = os.path.join(os.path.dirname(__file__), "..")
RNG = np.random.default_rng(0)


def jl(p):
    return [json.loads(l) for l in open(p)]


def boot(d, B=10000):
    n = len(d)
    bs = np.array([d[RNG.integers(0, n, n)].mean() for _ in range(B)])
    return np.percentile(bs, [2.5, 97.5])


def fmt(d):
    lo, hi = boot(d)
    return f"{d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]"


# ------------------------------------------------------------------ loaders (question x run arrays)
def medagents():
    from p0_stats import medagents_rerun
    M = medagents_rerun(return_rows=True)
    m = np.array([[r["pred_answer"] == r["gold_answer"] for r in M]], float)          # 1 x 1273
    S = np.array([[r["E3"] == r["gold"] for r in jl(f"{ROOT}/runs/p0_extraction.jsonl")]], float)
    B1 = np.array([[r["pred"] == r["gold"] for r in jl(f"{ROOT}/runs/p0_sweep/empty_300.jsonl")]], float)
    S5 = np.array([[r["E3"] == r["gold"] for r in jl(f"{ROOT}/runs/mda/S5/run{k}.jsonl")] for k in range(1, 6)], float)
    trunc = np.array([r["finish_reason"] == "length" for r in jl(f"{ROOT}/runs/p0_sweep/empty_300.jsonl")])
    C = [json.loads(l) for f in glob.glob(f"{ROOT}/runs/m1_medagents/shards/*/cost.jsonl") for l in open(f)]
    C = {r["idx"]: r for r in C}
    tok = np.mean([C[i]["prompt_tokens"] + C[i]["completion_tokens"] for i in range(1273)])
    usd = np.mean([(C[i]["prompt_tokens"] * .15 + C[i]["completion_tokens"] * .6) / 1e6 for i in range(1273)])
    E = jl(f"{ROOT}/runs/p0_extraction.jsonl")
    stok = np.mean([r["S_prompt_tokens"] + r["S_completion_tokens"] for r in E])
    susd = np.mean([(r["S_prompt_tokens"] * .15 + r["S_completion_tokens"] * .6) / 1e6 for r in E])
    return dict(M=m, S=S, base=B1, S5_200=S5, trunc=trunc, tok=tok, usd=usd, stok=stok, susd=susd,
                crash=0.0, M_ans=[[r["pred_answer"] for r in M]])


def mdagents(cond="F2_final_v2"):
    from mda_analyze import extract, load_runs
    runs = list(load_runs(cond).values())
    M = np.array([[rows[i]["error"] is None and extract(rows[i]["response"])[0] == rows[i]["label"] for i in range(200)]
                  for rows in runs], float)
    ans = [[extract(rows[i]["response"])[0] if rows[i]["error"] is None else "" for i in range(200)] for rows in runs]
    disc = np.array([[rows[i]["error"] is None and rows[i]["difficulty"] == "intermediate" and not rows[i].get("noAns", False)
                      for i in range(200)] for rows in runs], float)
    lvl = np.array([[{"basic": 0, "intermediate": 1, "advanced": 2}.get(rows[i]["difficulty"], np.nan) for i in range(200)]
                    for rows in runs], float)
    crash = np.mean([rows[i]["error"] is not None for rows in runs for i in range(200)])
    tok = np.mean([rows[i]["prompt_tokens"] + rows[i]["completion_tokens"] for rows in runs for i in range(200)])
    S5 = np.array([[r["E3"] == r["gold"] for r in jl(f"{ROOT}/runs/mda/S5/run{k}.jsonl")] for k in range(1, 6)], float)
    S5r = [jl(f"{ROOT}/runs/mda/S5/run{k}.jsonl") for k in range(1, 6)]
    stok = np.mean([r["S_prompt_tokens"] + r["S_completion_tokens"] for run in S5r for r in run])
    return dict(M=M, S=S5, disc=disc, lvl=lvl, crash=crash, tok=tok, stok=stok, M_ans=ans)


def medla():
    Q = 1273
    def L(s, r):
        d = {x["idx"]: x for x in jl(f"{ROOT}/runs/medla_full/{s}/run{r}.jsonl")}
        return [d[i] for i in range(Q)]
    L0 = [L("L0", r) for r in (1, 2)]
    L1 = [L("L1", r) for r in (1, 2, 3)]
    S4 = [jl(f"{ROOT}/runs/medla_full/S4/run{r}.jsonl") for r in (1, 2, 3)]
    M = np.array([[x["final_decision"] == x["label"] for x in run] for run in L0], float)
    S = np.array([[x["E3"] == x["gold"] for x in run] for run in S4], float)
    base = np.array([[x["final_decision"] == x["label"] for x in run] for run in L1], float)
    tok = np.mean([x["prompt_tokens"] + x["completion_tokens"] for run in L0 for x in run])
    usd = np.mean([(x["prompt_tokens"] * .15 + x["completion_tokens"] * .6) / 1e6 for run in L0 for x in run])
    stok = np.mean([x["S_prompt_tokens"] + x["S_completion_tokens"] for run in S4 for x in run])
    susd = np.mean([(x["S_prompt_tokens"] * .15 + x["S_completion_tokens"] * .6) / 1e6 for run in S4 for x in run])
    calls = np.array([[x["n_calls"] for x in run] for run in L0], float)
    return dict(M=M, S=S, base=base, tok=tok, usd=usd, stok=stok, susd=susd, crash=0.0, calls=calls,
                M_ans=[[x["final_decision"] for x in run] for run in L0])


# ------------------------------------------------------------------ (A) summary table
def summary(ma, md, md0, ml):
    print("=" * 100)
    print("(A) Cross-framework summary — GPT-4o-mini-2024-07-18, MedQA")
    print("=" * 100)
    rows = [
        ("MedAgents", "1273 q, 1 run", ma["M"][0] - ma["S"][0], ma["M"][0] - ma["base"][0], "M1 − B1 (released 300-tok CoT)",
         f"{ma['crash']:.1%}", "generation budget of the released single baseline (300 tok)",
         ma["tok"] / ma["stok"], ma["usd"] / ma["susd"]),
        ("MDAgents F2 (parser-robust)", "200 q, 5 runs", md["M"].mean(0) - md["S"].mean(0), None, "—",
         f"{md['crash']:.1%} (released F0: {md0['crash']:.1%})", "parser robustness; routing / backbone (F2→F3)",
         md["tok"] / md["stok"], None),
        ("MedLA", "1273 q, L0 2 / S4 3 runs", ml["M"].mean(0) - ml["S"].mean(0), ml["M"].mean(0) - ml["base"].mean(0),
         "L0 − L1 (released single mode)", f"{ml['crash']:.1%}", "none detected (clean execution)",
         ml["tok"] / ml["stok"], ml["usd"] / ml["susd"]),
    ]
    for name, n, gap, gbase, blabel, crash, sens, tx, ux in rows:
        print(f"\n{name}  ({n})")
        print(f"  MAS − strong single      {fmt(gap)}")
        print(f"  MAS − released baseline  {fmt(gbase) if gbase is not None else 'n/a'}   ({blabel})")
        print(f"  execution failure        {crash}")
        print(f"  main sensitivity         {sens}")
        print(f"  compute overhead vs S    tokens {tx:.0f}x" + (f", $ {ux:.0f}x" if ux else " ($ not comparable: mixed models)"))
        lo, hi = boot(gap)
        print(f"  gain survives strong-single control? {'NO' if hi < 0.03 else 'unclear'} (upper CI {hi:+.3f})")


# ------------------------------------------------------------------ (B) win / loss
def winloss(name, M, S, feats):
    m = M.mean(0) >= 0.5; s = S.mean(0) >= 0.5
    groups = {"MAS wins": m & ~s, "Single wins": ~m & s, "Both correct": m & s, "Both wrong": ~m & ~s}
    print(f"\n{name}: question-level majority correctness (n = {len(m)})")
    hdr = f"  {'group':13s} {'n':>5s} {'share':>6s} " + " ".join(f"{k:>18s}" for k in feats)
    print(hdr)
    for g, k in groups.items():
        vals = " ".join(f"{(v[k].mean() if k.any() else float('nan')):18.3f}" for v in feats.values())
        print(f"  {g:13s} {k.sum():5d} {k.mean():6.1%} {vals}")
    w, l = groups["MAS wins"].sum(), groups["Single wins"].sum()
    print(f"  net: MAS wins {w} vs single wins {l}  (ratio {w / max(l, 1):.2f})")


def stability(ans):
    A = np.array(ans)
    return np.array([len(set(A[:, i])) == 1 for i in range(A.shape[1])], float) if A.shape[0] > 1 else np.full(A.shape[1], np.nan)


# ------------------------------------------------------------------ (C) difficulty x gain
def diff_gain(name, M, S, k_bin):
    """bin on S runs [:k_bin] (held-out), evaluate S on the rest."""
    Sb, Se = S[:k_bin].mean(0), S[k_bin:].mean(0)
    m = M.mean(0)
    print(f"\n{name}: bins from S runs 1–{k_bin}, S evaluated on runs {k_bin + 1}–{S.shape[0]}")
    for lab, k in (("hard   (S bin = 0)", Sb == 0), ("medium (0 < bin < 1)", (Sb > 0) & (Sb < 1)), ("easy   (S bin = 1)", Sb == 1)):
        if k.sum() == 0:
            continue
        d = m[k] - Se[k]
        lo, hi = boot(d, 5000) if k.sum() > 1 else (np.nan, np.nan)
        print(f"  {lab:22s} n={k.sum():5d}  S {Se[k].mean():.3f}  MAS {m[k].mean():.3f}  MAS − S {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")


def main():
    ma, md, md0, ml = medagents(), mdagents("F2_final_v2"), mdagents("F0"), medla()
    summary(ma, md, md0, ml)

    print("\n" + "=" * 100 + "\n(B) Collaboration win / loss\n" + "=" * 100)
    Sdiff = lambda S: 1 - S.mean(0)
    winloss("MDAgents F2 (200 q, 5 runs; S = strong single x5)", md["M"], md["S"],
            {"S difficulty": 1 - md["S"].mean(0), "discussion freq": md["disc"].mean(0),
             "route level": np.nanmean(md["lvl"], 0), "MAS stability": stability(md["M_ans"])})
    winloss("MedLA (1273 q, L0 x2 vs S4 x3)", ml["M"], ml["S"],
            {"S difficulty": 1 - ml["S"].mean(0), "L0 stability": stability(ml["M_ans"]),
             "L0 calls / 100": ml["calls"].mean(0) / 100})
    winloss("MedAgents (1273 q, M1 x1 vs S x1)", ma["M"], ma["S"],
            {"B1 truncated": ma["trunc"].astype(float), "B1 correct": ma["base"][0]})

    print("\n" + "=" * 100 + "\n(C) Difficulty x MAS gain (held-out strong-single bins)\n" + "=" * 100)
    diff_gain("MDAgents F2", md["M"], md["S"], 2)
    diff_gain("MedLA", ml["M"], ml["S"], 1)
    diff_gain("MedAgents (questions 0–199, S x5)", ma["M"][:, :200], ma["S5_200"], 2)
    print("\n" + "=" * 100 + "\n(D) MedLA across backbones (4-option MedQA-US, 1,273 q)\n" + "=" * 100)
    for label, base_dir, lruns, sruns in (("GPT-4o-mini (L0 x2, L1/S4 x3)", "runs/medla_full", (1, 2), (1, 2, 3)),
                                          ("Llama-3.1-8B (paper backbone; 1 run each)", "runs/layer3", (1,), (1,))):
        def L(s, r):
            d = {x["idx"]: x for x in jl(f"{ROOT}/{base_dir}/{s}/run{r}.jsonl")}
            return [d[i] for i in range(1273)]
        L0 = np.array([[x["final_decision"] == x["label"] for x in L("L0", r)] for r in lruns], float).mean(0)
        L1 = np.array([[x["final_decision"] == x["label"] for x in L("L1", r)] for r in sruns], float).mean(0)
        S4 = np.array([[x["E3"] == x["gold"] for x in jl(f"{ROOT}/{base_dir}/S4/run{r}.jsonl")] for r in sruns], float).mean(0)
        t0 = np.mean([x["prompt_tokens"] + x["completion_tokens"] for r in lruns for x in L("L0", r)])
        ts = np.mean([x["S_prompt_tokens"] + x["S_completion_tokens"] for r in sruns for x in jl(f"{ROOT}/{base_dir}/S4/run{r}.jsonl")])
        print(f"\n{label}: L0 {L0.mean():.3f} | L1 {L1.mean():.3f} | S4 {S4.mean():.3f} | L0 tokens {t0 / ts:.0f}x S4")
        print(f"  L0 − L1 {fmt(L0 - L1)}   L0 − S4 {fmt(L0 - S4)}")
    import layer2_2a_report as A
    for tag in A.MODELS:
        S, M = A.load_s(tag), A.load_m1(tag)
        if S is None or M is None:
            continue
        s = np.array([[r["E3"] == r["gold"] for r in run] for run in S], float)
        m = np.array([[r["pred_answer"] == r["gold_answer"] for r in run] for run in M], float)
        diff_gain(f"2A {tag} (M1 vs S, 200 q x 3)", m, s, 1)


if __name__ == "__main__":
    main()
