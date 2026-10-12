"""Amendment 1 PILOT diagnostic (sealed; committed before any HET output existed). No LLM calls.
H_het = cross-fitted empirical-oracle headroom of {S, SC, HET} over {S, SC}; primary contrast H − H_het
(paired over questions). Also on type-C questions: HET_E − SC_E and M1_run3 − HET_E. No effect on GO.

  python studies/2_complementarity_router/code/het_analysis.py   ->  studies/2_complementarity_router/results/pilot/HET_RESULTS.{txt,json}
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import oracle_analysis as OA    # noqa: E402
import pilot_analysis as P      # noqa: E402

K_STAR = {"Qwen2.5-14B-Instruct": 22, "Llama-3.1-8B-Instruct": 7}    # from RESULTS.json (§3.3)


def main():
    res_all, lines = {}, []
    for tag, decides in P.MODELS:
        rng = np.random.default_rng(OA.SEED)
        S, SC, M, _ = P.load(tag)
        H = [P.jl(f"{P.BASE}/{tag}/HET/run{r}.jsonl") for r in (1, 2, 3)]
        assert all(len(run) == P.N for run in H)
        gold = np.array([r["gold"] for r in S[0]])
        assert all(r["gold"] == g for run in H for r, g in zip(run, gold))
        k = K_STAR[tag]
        assert json.load(open(f"{P.BASE}/RESULTS.json"))[tag]["k_star"] == k
        Sm = np.array([[r["E3"] == r["gold"] for r in run] for run in S], float)
        Mm = np.array([[r["pred_answer"] == r["gold_answer"] for r in run] for run in M], float)
        sc = np.concatenate([np.array([r["votes"] for r in run]) for run in SC[:3]], axis=1)
        het = np.concatenate([np.array([r["votes"] for r in run]) for run in H], axis=1)      # Q x 63
        pS, pM = Sm[:2].mean(0), Mm[:2].mean(0)
        pSC, eSC = OA.p_sc(sc[:, :40], gold, k, rng), OA.p_sc(sc[:, 40:], gold, k, rng)
        pH, eH = OA.p_sc(het[:, :42], gold, k, rng), OA.p_sc(het[:, 42:], gold, k, rng)
        eS, eM = Sm[2], Mm[2]
        Q = np.arange(len(gold))
        base = np.c_[eS, eSC][Q, OA.pick(np.c_[pS, pSC])]
        h_m = np.c_[eS, eSC, eM][Q, OA.pick(np.c_[pS, pSC, pM])] - base
        h_h = np.c_[eS, eSC, eH][Q, OA.pick(np.c_[pS, pSC, pH])] - base
        c = OA.types(pS, pSC, pM) == "C"
        tok_het = np.mean([r["prompt_tokens"] + r["completion_tokens"] for run in H for r in run]) / 21
        r = dict(k_star=k, het_tokens_per_sample=float(tok_het), n_C=int(c.sum()),
                 H=OA.boot_ci(h_m, rng), H_het=OA.boot_ci(h_h, rng), H_minus_H_het=OA.boot_ci(h_m - h_h, rng),
                 C_het_minus_sc=OA.boot_ci(eH[c] - eSC[c], rng), C_m_minus_het=OA.boot_ci(eM[c] - eH[c], rng),
                 acc=dict(S=float(Sm.mean()), SC=float(eSC.mean()), HET=float(eH.mean()), M1=float(Mm.mean())))
        res_all[tag] = r
        f = lambda v: f"{v[0]*100:+.1f} pp [{v[1]*100:+.1f}, {v[2]*100:+.1f}]"
        lines += [f"=== {tag} {'(decision model)' if decides else ''} — k* = {k}, HET tokens/sample {tok_het:.0f} ===",
                  f"acc (eval half / all runs): S {r['acc']['S']:.3f}  SC-{k} {r['acc']['SC']:.3f}  HET-{k} {r['acc']['HET']:.3f}  M1 {r['acc']['M1']:.3f}",
                  f"H (M1)          {f(r['H'])}",
                  f"H_het (HET)     {f(r['H_het'])}",
                  f"H − H_het       {f(r['H_minus_H_het'])}   <- headroom attributable to interaction beyond heterogeneity",
                  f"type C (n={r['n_C']}): HET_E − SC_E {f(r['C_het_minus_sc'])};  M1_run3 − HET_E {f(r['C_m_minus_het'])}", ""]
    open(f"{P.BASE}/HET_RESULTS.txt", "w").write("\n".join(lines) + "\n")
    json.dump(res_all, open(f"{P.BASE}/HET_RESULTS.json", "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
