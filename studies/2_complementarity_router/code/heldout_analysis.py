"""Sealed HELD-OUT analysis (PHASE2_FREEZE.md §5, §7, Amendments 1–2). Qwen2.5-14B, 339 questions, one
execution of every arm. Committed and hashed before any HELD-OUT output exists. No LLM calls.

Policies are built offline from the same outputs (no separate runs per policy). Per-question cost is the
measured tokens: probe = 4/22 of that question's SC-22 tokens.
  Ours-Collab    STOP: S, cost S+probe | SAMPLE: SC-22, cost S+SC | COLLABORATE: M1, cost S+probe+M1
  Ours-Sampling  same routing; COLLABORATE -> SC-22, cost S+SC
  Ours-Hetero    same routing; COLLABORATE -> HET-22, cost S+probe+HET
Output order is fixed: (1) Criterion B (primary), (2) Criterion A, (3) Criterion B′, (4) fixed strategies
and routing baselines, (5) descriptive diagnostics. Thresholds are never revisited after reading (1).

  python studies/2_complementarity_router/code/heldout_analysis.py   ->  studies/2_complementarity_router/results/heldout/RESULTS.{txt,json}
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import features as F            # noqa: E402
import router_policy as RP      # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
TAG = "Qwen2.5-14B-Instruct"
BASE = f"{ROOT}/studies/2_complementarity_router/results/heldout/{TAG}"
N, K, B, SEED = 339, 22, 10000, 0
MARGIN = 0.01                       # Criterion A non-inferiority margin (1 pp), one-sided α = .05


def jl(p):
    return [json.loads(l) for l in open(p)]


def boot(d, rng, q=(2.5, 97.5)):
    d = np.asarray(d, float)
    bs = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return float(d.mean()), *map(float, np.percentile(bs, q))


def load():
    S = jl(f"{BASE}/S/run1.jsonl")
    SC = jl(f"{BASE}/SC/run1.jsonl")
    HET = jl(f"{BASE}/HET/run1.jsonl")
    by = {}
    for sd in glob.glob(f"{BASE}/M1/run1/shard*"):
        s = int(os.path.basename(sd)[5:])
        for k, l in enumerate(open(f"{sd}/chatgpt-syn_verif")):
            by[s * 20 + k] = json.loads(l)
    M = [by[i] for i in range(N)]
    cost = {}
    for sd in glob.glob(f"{BASE}/M1/run1/shard*"):
        s = int(os.path.basename(sd)[5:])
        for k, l in enumerate(open(f"{sd}/cost.jsonl")):
            cost[s * 20 + k] = json.loads(l)
    assert len(S) == len(SC) == len(HET) == len(M) == len(cost) == N
    gold = np.array([r["gold"] for r in S])
    assert all(r["gold"] == g for rows in (SC, HET) for r, g in zip(rows, gold))
    assert all(r["gold_answer"] == g for r, g in zip(M, gold))
    assert all(len(r["votes"]) == K for r in SC + HET)
    return S, SC, HET, M, [cost[i] for i in range(N)], gold


def main():
    rng = np.random.default_rng(SEED)
    frozen = RP.load()
    assert frozen["k_star"] == K
    S, SC, HET, M, cost, gold = load()
    idx = json.load(open(f"{ROOT}/studies/2_complementarity_router/results/splits/heldout.json"))["idx"]
    qs = [r["question"] for r in jl(f"{ROOT}/data/splits/MedQA_p2heldout/test.jsonl")]

    c = dict(S=np.array([r["E3"] == r["gold"] for r in S], float),
             SC=np.array([r["pred"] == r["gold"] for r in SC], float),
             HET=np.array([r["pred"] == r["gold"] for r in HET], float),
             M=np.array([r["pred_answer"] == r["gold_answer"] for r in M], float))
    t = dict(S=np.array([r["S_prompt_tokens"] + r["S_completion_tokens"] for r in S], float),
             SC=np.array([r["prompt_tokens"] + r["completion_tokens"] for r in SC], float),
             HET=np.array([r["prompt_tokens"] + r["completion_tokens"] for r in HET], float),
             M=np.array([r["prompt_tokens"] + r["completion_tokens"] for r in cost], float))
    probe = t["SC"] * 4 / K

    X = F.build(S, np.array([r["votes"] for r in SC])[:, :4], idx, qs)
    a, qC, qA = RP.act(frozen, X)
    STOP, SAMPLE, COLLAB = RP.STOP, RP.SAMPLE, RP.COLLABORATE

    def routed(collab_c, collab_t):
        acc = np.select([a == STOP, a == SAMPLE], [c["S"], c["SC"]], collab_c)
        tok = np.select([a == STOP, a == SAMPLE], [t["S"] + probe, t["S"] + t["SC"]], collab_t)
        return acc, tok

    ours_c, ours_t = routed(c["M"], t["S"] + probe + t["M"])
    samp_c, samp_t = routed(c["SC"], t["S"] + t["SC"])
    het_c, het_t = routed(c["HET"], t["S"] + probe + t["HET"])

    res, L = {}, []
    pp = lambda v: f"{v[0]*100:+.1f} pp [{v[1]*100:+.1f}, {v[2]*100:+.1f}]"

    # (1) Criterion B — primary
    dB = boot(ours_c - samp_c, rng)
    rB = ours_t.mean() / samp_t.mean()
    passB = dB[1] > 0 and 0.9 <= rB <= 1.1
    res["B"] = dict(diff=dB, token_ratio=rB, passed=passB)
    L += ["(1) PRIMARY — Criterion B: Ours-Collab − Ours-Sampling",
          f"    Δ_B = {pp(dB)};  token ratio {rB:.3f} (window [0.9, 1.1])  -> {'PASS' if passB else 'FAIL'}"]

    # (2) Criterion A — non-inferiority to always-on M1 (one-sided 95% lower bound > −1 pp) and ≤ 0.5× tokens
    dA = boot(ours_c - c["M"], rng, q=(5, 95))
    rA = ours_t.mean() / t["M"].mean()
    passA = dA[1] > -MARGIN and rA <= 0.5
    res["A"] = dict(diff_90ci=dA, token_ratio_vs_M1=rA, passed=passA)
    L += ["(2) Criterion A: Ours-Collab vs always-on M1",
          f"    diff = {pp(dA)} (90% CI; non-inferior if lower > −1.0 pp);  tokens {rA:.3f}× M1 (≤ 0.5)"
          f"  -> {'PASS' if passA else 'FAIL'}"]

    # (3) Criterion B′ — interaction beyond heterogeneity
    dBp = boot(ours_c - het_c, rng)
    rBp = ours_t.mean() / het_t.mean()
    passBp = dBp[1] > 0 and 0.9 <= rBp <= 1.1
    res["B_prime"] = dict(diff=dBp, token_ratio=rBp, passed=passBp)
    L += ["(3) Criterion B′: Ours-Collab − Ours-Hetero",
          f"    Δ_B′ = {pp(dBp)};  token ratio {rBp:.3f}  -> {'PASS' if passBp else 'FAIL'}"]
    reading = ("A+B: selective collaboration" if passA and passB else
               "A only: adaptivity, not interaction, drives the gain" if passA else
               "B only: interaction signal without practical efficiency" if passB else
               "neither A nor B: adaptive method dropped; mechanism finding only")
    if passB:
        reading += " | B′ " + ("PASS: interaction-specific" if passBp else "FAIL: explained by heterogeneous expertise")
    res["reading"] = reading
    L += [f"    pre-declared reading: {reading}", ""]

    # (4) fixed strategies and routing baselines
    rows = [("S", c["S"], t["S"]), ("SC-22", c["SC"], t["SC"]), ("HET-22", c["HET"], t["HET"]),
            ("always-on M1", c["M"], t["M"]), ("Ours-Collab", ours_c, ours_t),
            ("Ours-Sampling", samp_c, samp_t), ("Ours-Hetero", het_c, het_t)]
    pmax = X[:, F.NAMES.index("probe_max_share")]
    esc = pmax < frozen["difficulty_tau"]
    rows += [("Difficulty→SC (a)", np.where(esc, c["SC"], c["S"]), np.where(esc, t["S"] + t["SC"], t["S"] + probe)),
             ("Difficulty→M1 (b)", np.where(esc, c["M"], c["S"]), np.where(esc, t["S"] + probe + t["M"], t["S"] + probe))]
    rr = np.random.default_rng(SEED)
    accs, toks = [], []
    for _ in range(1000):                       # random routing with Ours' held-out action counts
        p = rr.permutation(a)
        accs.append(np.select([p == STOP, p == SAMPLE], [c["S"], c["SC"]], c["M"]).mean())
        toks.append(np.select([p == STOP, p == SAMPLE], [t["S"] + probe, t["S"] + t["SC"]], t["S"] + probe + t["M"]).mean())
    L += ["(4) fixed strategies and routing baselines (accuracy, tokens/q, diff vs Ours-Collab [95% CI])"]
    res["table"] = {}
    for name, acc, tok in rows:
        d = boot(ours_c - acc, rng) if name != "Ours-Collab" else None
        res["table"][name] = dict(acc=float(acc.mean()), tokens=float(tok.mean()), ours_minus=d)
        L.append(f"    {name:20s} {acc.mean():.3f}  {tok.mean():8.0f}" + (f"   Ours − this {pp(d)}" if d else ""))
    res["table"]["Random routing"] = dict(acc=float(np.mean(accs)), acc_95=list(np.percentile(accs, [2.5, 97.5])),
                                          tokens=float(np.mean(toks)))
    L += [f"    {'Random routing':20s} {np.mean(accs):.3f}  {np.mean(toks):8.0f}   (1,000 permutations; 95% range "
          f"{np.percentile(accs, 2.5):.3f}–{np.percentile(accs, 97.5):.3f})", ""]

    # (5) descriptive diagnostics
    sel = a == COLLAB
    res["diag"] = dict(share={n: float((a == j).mean()) for j, n in enumerate(("STOP", "SAMPLE", "COLLABORATE"))},
                       n_collab=int(sel.sum()),
                       collab_selected_acc={k_: float(v[sel].mean()) if sel.any() else None for k_, v in c.items()})
    L += ["(5) diagnostics",
          "    action shares: " + "  ".join(f"{k_} {v:.3f}" for k_, v in res["diag"]["share"].items())
          + f"  (COLLABORATE n = {int(sel.sum())})",
          "    on COLLABORATE-selected questions: " + ("  ".join(f"{k_} {v:.3f}" for k_, v in res["diag"]["collab_selected_acc"].items())
                                                      if sel.any() else "none selected")]
    out = f"{ROOT}/studies/2_complementarity_router/results/heldout"
    open(f"{out}/RESULTS.txt", "w").write("\n".join(L) + "\n")
    json.dump(res, open(f"{out}/RESULTS.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("\n".join(L))


if __name__ == "__main__":
    main()
