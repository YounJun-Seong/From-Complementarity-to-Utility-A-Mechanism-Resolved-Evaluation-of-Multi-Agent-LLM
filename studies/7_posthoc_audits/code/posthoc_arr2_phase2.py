"""Post-hoc ARR2-E1/E2/E3 (POSTHOC_ARR2.md; definitions committed before this script was run). No LLM calls.

  E1  compute-corrected C* (generation prompt charged per SC sample), PILOT + HELD-OUT sensitivity
  E2  C* across the candidate budget k (Phase II PILOT; Phase V MedARC replication), naive / union oracles
  E3  router rate–accuracy–cost curves and routing baselines (PILOT cross-fitted; HELD-OUT exploratory)

  python studies/7_posthoc_audits/code/posthoc_arr2_phase2.py  ->  studies/7_posthoc_audits/results/arr2/E{1,2,3}.{txt,json}
"""
import glob
import json
import os
import sys

import numpy as np
from sklearn.model_selection import KFold

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..", "..")
sys.path.insert(0, f"{HERE}/../../2_complementarity_router/code"); sys.path.insert(0, f"{HERE}/../../5_medarc_transfer/code")
import features as F            # noqa: E402
import oracle_analysis as OA    # noqa: E402
import pilot_analysis as P      # noqa: E402
import router as RT             # noqa: E402
import router_policy as RP      # noqa: E402

B, SEED = 10000, 0
OUT = f"{ROOT}/studies/7_posthoc_audits/results/arr2"
QWEN, LLAMA = "Qwen2.5-14B-Instruct", "Llama-3.1-8B-Instruct"
pp = lambda v: f"{v[0]*100:+.1f} [{v[1]*100:+.1f}, {v[2]*100:+.1f}]" if v is not None else "n/a"


def boot(d, rng):
    d = np.asarray(d, float)
    bs = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return [float(d.mean()), *map(float, np.percentile(bs, [2.5, 97.5]))]


# ---------------------------------------------------------------- Phase II PILOT data
def pilot(tag):
    S, SC, M, cost = P.load(tag)
    H = [P.jl(f"{P.BASE}/{tag}/HET/run{r}.jsonl") for r in (1, 2, 3)]
    gold = np.array([r["gold"] for r in S[0]])
    Sm = np.array([[r["E3"] == r["gold"] for r in run] for run in S], float)
    Mm = np.array([[r["pred_answer"] == r["gold_answer"] for r in run] for run in M], float)
    votes = [np.array([r["votes"] for r in run]) for run in SC]
    het = np.concatenate([np.array([r["votes"] for r in run]) for run in H], axis=1)
    n = 20
    s_prompt = np.mean([r["S_prompt_tokens"] for run in S for r in run])
    sc_prompt = np.mean([r["prompt_tokens"] for run in SC[:3] for r in run])
    gen_p = (n * s_prompt - sc_prompt) / (n - 1)
    tok = dict(S=np.mean([r["S_prompt_tokens"] + r["S_completion_tokens"] for run in S for r in run]),
               SC=np.mean([(r["prompt_tokens"] + r["completion_tokens"]) / n for run in SC[:3] for r in run]),
               M=np.mean([c["prompt_tokens"] + c["completion_tokens"] for c in cost]),
               HET=np.mean([(r["prompt_tokens"] + r["completion_tokens"]) / 21 for run in H for r in run]))
    roles = np.mean([r["calls"] - 2 - 21 for run in H for r in run])
    tok_c = dict(tok, SC=tok["SC"] + gen_p * (n - 1) / n, HET=tok["HET"] + gen_p * (21 - roles) / 21)
    idx = json.load(open(f"{ROOT}/studies/2_complementarity_router/results/splits/pilot.json"))["idx"]
    qs = [r["question"] for r in P.jl(f"{ROOT}/data/splits/MedQA_p2pilot/test.jsonl")]
    X = F.build(S[0], votes[0][:, :4], idx, qs)
    return dict(S=S, Sm=Sm, Mm=Mm, votes=votes, het=het, gold=gold, tok=tok, tok_c=tok_c, gen_p=gen_p,
                roles=roles, X=X)


def het_headroom(d, k):
    """het_analysis.py at budget k (same rng order)."""
    rng = np.random.default_rng(OA.SEED)
    sc = np.concatenate(d["votes"][:3], axis=1)
    gold, Sm, Mm, het = d["gold"], d["Sm"], d["Mm"], d["het"]
    pS, pM = Sm[:2].mean(0), Mm[:2].mean(0)
    pSC, eSC = OA.p_sc(sc[:, :40], gold, k, rng), OA.p_sc(sc[:, 40:], gold, k, rng)
    pH, eH = OA.p_sc(het[:, :42], gold, k, rng), OA.p_sc(het[:, 42:], gold, k, rng)
    eS, eM = Sm[2], Mm[2]
    Q = np.arange(len(gold))
    base = np.c_[eS, eSC][Q, OA.pick(np.c_[pS, pSC])]
    h_m = np.c_[eS, eSC, eM][Q, OA.pick(np.c_[pS, pSC, pM])] - base
    h_h = np.c_[eS, eSC, eH][Q, OA.pick(np.c_[pS, pSC, pH])] - base
    return dict(H=OA.boot_ci(h_m, rng), H_het=OA.boot_ci(h_h, rng), C_het=OA.boot_ci(h_m - h_h, rng),
                acc_HET_E=float(eH.mean()))


def oracle_at(Sm, votes3, gold, Mm, placebo, k, tok, pool_split, X=None):
    """Sealed analyze() at k, plus naive (non-cross-fitted) and union-oracle headroom and diversity."""
    Q = len(gold)
    r = OA.analyze(Sm, votes3, gold, Mm, tok, X if X is not None else np.zeros((Q, 2)), k=k,
                   placebo_votes=placebo, pool_split=pool_split)
    rng = np.random.default_rng(SEED)
    pS, pM = Sm.mean(0), Mm.mean(0)
    pSC_all = OA.p_sc(votes3, gold, k, rng)
    pP_all = OA.p_sc(placebo, gold, k, rng)
    eSC = OA.p_sc(votes3[:, pool_split:], gold, k, rng)
    eP = OA.p_sc(placebo[:, placebo.shape[1] // 2:], gold, k, rng)
    eS, eM = Sm[2], Mm[2]
    base_n = np.maximum(pS, pSC_all)
    hn, hnp = np.maximum(base_n, pM) - base_n, np.maximum(base_n, pP_all) - base_n
    base_u = np.maximum(eS, eSC)
    u, up = np.maximum(base_u, eM) - base_u, np.maximum(base_u, eP) - base_u
    brng = np.random.default_rng(SEED)
    return dict(k=k, acc_SC_E=float(eSC.mean()), acc_M=float(Mm.mean()), acc_S=float(Sm.mean()),
                H=r["H"], H_plc=r["H_plc"], C_star=r["H_minus_H_plc"], n_C=r["n_C"], G3=r["G3_contrast"],
                acc_oracle_CS=r["acc_oracle_CS"], acc_oracle_S=r["acc_oracle_S"], share_M_chosen=r["share_M_chosen"],
                H_naive=boot(hn, brng), H_naive_plc=boot(hnp, brng), C_naive=boot(hn - hnp, brng),
                U=boot(u, brng), U_plc=boot(up, brng), U_minus_plc=boot(u - up, brng),
                M_only=float(np.mean(eM * (1 - eSC))), SC_only=float(np.mean((1 - eM) * eSC)), _r=r)


# ---------------------------------------------------------------- E1
def e1():
    res, L = {}, ["=== ARR2-E1 compute-corrected C* (Phase II PILOT) ==="]
    for tag in (QWEN, LLAMA):
        d = pilot(tag)
        frozen = json.load(open(f"{P.BASE}/RESULTS.json"))[tag]
        k_fz = OA.k_star(d["tok"]["M"], d["tok"]["SC"], 60)
        k_c = OA.k_star(d["tok_c"]["M"], d["tok_c"]["SC"], 60)
        k_hc = OA.k_star(d["tok_c"]["M"], d["tok_c"]["HET"], 63)
        assert k_fz == frozen["k_star"]
        votes3, plc = np.concatenate(d["votes"][:3], 1), np.concatenate(d["votes"][3:], 1)
        rows = {}
        for k in sorted({*range(min(k_c, k_fz), max(k_c, k_fz) + 1), k_hc}):
            o = oracle_at(d["Sm"], votes3, d["gold"], d["Mm"], plc, k, d["tok"], 40, d["X"])
            h = het_headroom(d, k)
            rr = o.pop("_r")
            for name, t in (("frozen", d["tok"]), ("corrected", d["tok_c"])):
                o[f"tok_SC_over_M_{name}"] = float(k * t["SC"] / t["M"])
            o.update(H_het=h["H_het"], C_star_het=h["C_het"], acc_HET_E=h["acc_HET_E"])
            rows[k] = o
            if k == k_fz:
                for key, fk in (("H", "H"), ("H_plc", "H_plc"), ("C_star", "H_minus_H_plc")):
                    assert np.allclose(o[key], frozen[fk]), (tag, key)
                assert np.allclose(rr["G4"]["auc"], frozen["G4"]["auc"])
        hf = json.load(open(f"{P.BASE}/HET_RESULTS.json"))[tag]
        assert np.allclose(rows[k_fz]["C_star_het"], hf["H_minus_H_het"]), "HET reproduction"
        res[tag] = dict(gen_prompt=float(d["gen_p"]), roles=float(d["roles"]), tok=d["tok"], tok_corrected=d["tok_c"],
                        k_star=k_fz, k_star_corrected=k_c, k_star_het_corrected=k_hc, rows=rows)
        L += [f"--- {tag}: gen prompt ≈ {d['gen_p']:.0f} tok; SC/sample {d['tok']['SC']:.0f} → {d['tok_c']['SC']:.0f}; "
              f"HET/sample {d['tok']['HET']:.0f} → {d['tok_c']['HET']:.0f} (roles {d['roles']:.2f}); M1 {d['tok']['M']:.0f}",
              f"    k* frozen {k_fz} → corrected {k_c} (HET corrected {k_hc}). Reproduction at k={k_fz}: H, H_plc, C*, AUC, C*_het OK",
              f"    {'k':>3} {'SC/M1 tok (fz|corr)':>20} {'acc SC_E':>8} {'H':>20} {'H_plc':>20} {'C*':>20} {'C*_het':>20} {'n_C':>4} {'G3':>22}"]
        for k, o in rows.items():
            tagk = " ← k*′" if k == k_c else (" ← k* frozen" if k == k_fz else (" ← k*′het" if k == k_hc else ""))
            L.append(f"    {k:>3} {o['tok_SC_over_M_frozen']:>9.3f} | {o['tok_SC_over_M_corrected']:<8.3f} {o['acc_SC_E']:>8.3f} "
                     f"{pp(o['H']):>20} {pp(o['H_plc']):>20} {pp(o['C_star']):>20} {pp(o['C_star_het']):>20} {o['n_C']:>4} {pp(o['G3']):>22}{tagk}")
        L.append("")
    res["HELDOUT"], hl = e1_heldout(res[QWEN]["k_star_corrected"])
    L += hl
    return res, L


def heldout():
    D = f"{ROOT}/studies/2_complementarity_router/results/heldout/{QWEN}"
    S = P.jl(f"{D}/S/run1.jsonl"); SC = P.jl(f"{D}/SC/run1.jsonl"); HET = P.jl(f"{D}/HET/run1.jsonl")
    by, cost = {}, {}
    for sd in glob.glob(f"{D}/M1/run1/shard*"):
        s = int(os.path.basename(sd)[5:])
        for k, l in enumerate(open(f"{sd}/chatgpt-syn_verif")):
            by[s * 20 + k] = json.loads(l)
        for k, l in enumerate(open(f"{sd}/cost.jsonl")):
            cost[s * 20 + k] = json.loads(l)
    N = len(S)
    M = [by[i] for i in range(N)]
    gold = np.array([r["gold"] for r in S])
    idx = json.load(open(f"{ROOT}/studies/2_complementarity_router/results/splits/heldout.json"))["idx"]
    qs = [r["question"] for r in P.jl(f"{ROOT}/data/splits/MedQA_p2heldout/test.jsonl")]
    v = np.array([r["votes"] for r in SC])
    X = F.build(S, v[:, :4], idx, qs)
    c = dict(S=np.array([r["E3"] == r["gold"] for r in S], float), SC=np.array([r["pred"] == r["gold"] for r in SC], float),
             HET=np.array([r["pred"] == r["gold"] for r in HET], float),
             M=np.array([r["pred_answer"] == r["gold_answer"] for r in M], float))
    t = dict(S=np.array([r["S_prompt_tokens"] + r["S_completion_tokens"] for r in S], float),
             SC=np.array([r["prompt_tokens"] + r["completion_tokens"] for r in SC], float),
             HET=np.array([r["prompt_tokens"] + r["completion_tokens"] for r in HET], float),
             M=np.array([cost[i]["prompt_tokens"] + cost[i]["completion_tokens"] for i in range(N)], float))
    n = 22
    gen_q = (n * np.array([r["S_prompt_tokens"] for r in S], float) - np.array([r["prompt_tokens"] for r in SC], float)) / (n - 1)
    return dict(c=c, t=t, X=X, votes=v, gold=gold, gen_q=gen_q, N=N, HETrows=HET)


def first_k(v, gold, k):
    out = []
    for row, g in zip(v[:, :k], gold):
        labs = [x for x in row if x]
        if not labs:
            out.append(0.0); continue
        cnt = {a: labs.count(a) for a in labs}; top = max(cnt.values())
        out.append(float(next(a for a in labs if cnt[a] == top) == g))
    return np.array(out)


def e1_heldout(k_c):
    rng = np.random.default_rng(SEED)
    h = heldout()
    c, t = h["c"], h["t"]
    assert np.array_equal(first_k(h["votes"], h["gold"], 22), c["SC"]), "SC-22 first-k must equal the sealed pred"
    a, _, _ = RP.act(RP.load(), h["X"])
    per22 = t["SC"] / 22
    per_c = per22 + h["gen_q"] * 21 / 22
    res, L = {}, [f"--- HELD-OUT sensitivity (Qwen, n={h['N']}): per-question gen prompt mean {h['gen_q'].mean():.0f} tok"]
    for k, acct in ((22, "frozen"), (k_c, "corrected")):
        cSC = first_k(h["votes"], h["gold"], k)
        per = per22 if acct == "frozen" else per_c
        tSC, probe = per * k, per * 4
        sel = lambda col, ct: (np.select([a == 0, a == 1], [c["S"], cSC], col),
                               np.select([a == 0, a == 1], [t["S"] + probe, t["S"] + tSC], ct))
        oc, ot = sel(c["M"], t["S"] + probe + t["M"])
        sc_, st = sel(cSC, t["S"] + tSC)
        d = boot(oc - sc_, rng)
        res[f"k{k}_{acct}"] = dict(k=k, acc_SC=float(cSC.mean()), acc_M=float(c["M"].mean()), M_minus_SC=boot(c["M"] - cSC, rng),
                                   tok_SC_over_M=float(tSC.mean() / t["M"].mean()), ours_collab=float(oc.mean()),
                                   ours_sampling=float(sc_.mean()), B_diff=d, B_token_ratio=float(ot.mean() / st.mean()))
        r = res[f"k{k}_{acct}"]
        L.append(f"    SC-{k} ({acct} accounting): acc SC {r['acc_SC']:.3f} vs M1 {r['acc_M']:.3f}, M1 − SC {pp(r['M_minus_SC'])}; "
                 f"SC/M1 tokens {r['tok_SC_over_M']:.3f}; Criterion B Ours-Collab − Ours-Sampling {pp(d)}, token ratio {r['B_token_ratio']:.3f}")
    return res, L + [""]


# ---------------------------------------------------------------- E2
def phase5():
    sys.path.insert(0, f"{HERE}/../../5_medarc_transfer/code")
    import aggregate as AG      # noqa: E402
    import parsers as PS        # noqa: E402
    D = f"{ROOT}/studies/5_medarc_transfer/results/main"
    K = json.load(open(f"{ROOT}/studies/5_medarc_transfer/results/K_FROZEN.json"))["k_star"]
    lab = json.load(open(f"{ROOT}/studies/5_medarc_transfer/results/labels.json"))
    M = [P.jl(f"{D}/medarc_run{r}.jsonl") for r in (1, 2, 3)]
    COT = [P.jl(f"{D}/cot_run{r}.jsonl") for r in (1, 2, 3)]
    SC = [P.jl(f"{D}/sc_run{r}.jsonl") for r in range(1, 7)]
    gold = np.array([lab[str(x["pubid"])] for x in M[0]])
    c = lambda preds: np.array([p == g for p, g in zip(preds, gold)], float)
    Mm = np.array([c([AG.a_paper_no_reinit(x)[0] for x in run]) for run in M])
    Sm = np.array([c([PS.answer(x["text"]) for x in run]) for run in COT])
    votes = [np.array([[PS.answer(t) or "" for t in x["texts"]] for x in run]) for run in SC]
    return K, gold, Sm, Mm, votes


def e2():
    res, L = {}, ["=== ARR2-E2 C* across the candidate budget k ==="]
    hdr = (f"    {'k':>3} {'SC/M1 tok':>9} {'accSC_E':>7} {'H (xfit)':>19} {'H_plc':>19} {'C*':>19} {'C*_het':>19} "
           f"{'H_naive':>19} {'C*_naive':>19} {'U (union)':>19} {'U−U_plc':>19} {'P(M✓SC✗)':>8} {'P(M✗SC✓)':>8}")
    d = pilot(QWEN)
    votes3, plc = np.concatenate(d["votes"][:3], 1), np.concatenate(d["votes"][3:], 1)
    rows = {}
    for k in (1, 3, 5, 7, 10, 15, 18, 20, 22):
        o = oracle_at(d["Sm"], votes3, d["gold"], d["Mm"], plc, k, d["tok"], 40, d["X"]); o.pop("_r")
        h = het_headroom(d, k)
        o.update(C_star_het=h["C_het"], H_het=h["H_het"], tok_SC_over_M=float(k * d["tok"]["SC"] / d["tok"]["M"]),
                 tok_SC_over_M_corrected=float(k * d["tok_c"]["SC"] / d["tok"]["M"]))
        rows[k] = o
    frozen = json.load(open(f"{P.BASE}/RESULTS.json"))[QWEN]
    assert np.allclose(rows[22]["C_star"], frozen["H_minus_H_plc"])
    res["PhaseII_Qwen"] = rows
    L += [f"--- Phase II PILOT, Qwen2.5-14B (n=300; M1 fixed; SC/M1 tokens frozen accounting, corrected in JSON)", hdr]
    for k, o in rows.items():
        L.append(f"    {k:>3} {o['tok_SC_over_M']:>9.3f} {o['acc_SC_E']:>7.3f} {pp(o['H']):>19} {pp(o['H_plc']):>19} {pp(o['C_star']):>19} "
                 f"{pp(o['C_star_het']):>19} {pp(o['H_naive']):>19} {pp(o['C_naive']):>19} {pp(o['U']):>19} {pp(o['U_minus_plc']):>19} "
                 f"{o['M_only']:>8.3f} {o['SC_only']:>8.3f}")
    # Phase V MedARC replication
    K, gold, Sm, Mm, votes = phase5()
    v3, pl = np.concatenate(votes[:3], 1), np.concatenate(votes[3:], 1)
    rows5 = {}
    for k in (1, 3, 5, 7, 10, 15, 20, 25):
        o = oracle_at(Sm, v3, gold, Mm, pl, k, dict(S=1, SC=1, M=1), 2 * K); o.pop("_r")
        rows5[k] = o
    sealed = json.load(open(f"{ROOT}/studies/5_medarc_transfer/results/RESULTS.json"))["G2"]
    assert np.allclose(rows5[K]["C_star"], sealed["C_star"]), "Phase V reproduction"
    res["PhaseV_MedARC"] = rows5
    L += ["", f"--- Phase V MedARC, Qwen2.5-14B, PubMedQA (n={len(gold)}; sealed k*={K}; reproduction OK)",
          hdr.replace(f"{'C*_het':>19} ", "").replace(f"{'SC/M1 tok':>9} ", "")]
    for k, o in rows5.items():
        L.append(f"    {k:>3} {o['acc_SC_E']:>7.3f} {pp(o['H']):>19} {pp(o['H_plc']):>19} {pp(o['C_star']):>19} "
                 f"{pp(o['H_naive']):>19} {pp(o['C_naive']):>19} {pp(o['U']):>19} {pp(o['U_minus_plc']):>19} "
                 f"{o['M_only']:>8.3f} {o['SC_only']:>8.3f}")
    return res, L + [""]


# ---------------------------------------------------------------- E3
RATES = (0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0)
TAUS = (0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.075, 0.05, 0.025)


def pilot_oof(d, k=22):
    """Out-of-fold q_C, q_A exactly as router.crossfit_vc (λ = 0), returning probabilities."""
    gold, votes = d["gold"], d["votes"]
    yS, yM = d["Sm"].mean(0), d["Mm"].mean(0)
    ySC = OA.p_sc(np.concatenate(votes[:3], axis=1), gold, k, np.random.default_rng(0))
    Y = np.c_[yS, ySC, yM]
    t = d["tok"]
    cost = np.array([t["S"] + 4 * t["SC"], t["S"] + k * t["SC"], t["S"] + 4 * t["SC"] + t["M"]])
    yC, yA = RT.vc_labels(Y, cost, 0.0)
    X = d["X"]
    q = np.zeros((2, len(Y)))
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
        for j, y in enumerate((yC, yA)):
            q[j, te] = (np.full(len(te), float(y[tr][0])) if y[tr].min() == y[tr].max()
                        else RT.make_model("logreg").fit(X[tr], y[tr]).predict_proba(X[te])[:, 1])
    a = np.where(q[0] > .5, 2, np.where(q[1] > .5, 1, 0))
    assert np.array_equal(a, RT.crossfit_vc(X, Y, cost, 0.0)), "OOF reproduction of crossfit_vc"
    yH = OA.p_sc(d["het"], gold, k, np.random.default_rng(0))
    return Y, yH, q[0], q[1], cost


def order(score, tie=None):
    keys = [np.arange(len(score))] + ([tie] if tie is not None else []) + [-score]
    return np.lexsort(keys)                      # descending score, then tie-breaker ascending, then index


def curves(base_c, base_t, collab, samp, het, selectors, rng):
    """base: non-escalated accuracy/tokens per question; collab/samp/het: (acc, tokens) if escalated."""
    Q = len(base_c)
    out = {}
    for name, ordr in selectors.items():
        rows = {}
        for r in RATES:
            n = int(np.ceil(r * Q))
            if name == "random":
                accs = {"collab": [], "samp": [], "het": [], "tc": [], "ts": []}
                diffs = np.zeros(Q)
                for _ in range(1000):
                    esc = np.zeros(Q, bool); esc[rng.permutation(Q)[:n]] = True
                    cc = np.where(esc, collab[0], base_c); cs = np.where(esc, samp[0], base_c)
                    accs["collab"].append(cc.mean()); accs["samp"].append(cs.mean())
                    accs["het"].append(np.where(esc, het[0], base_c).mean())
                    accs["tc"].append(np.where(esc, collab[1], base_t).mean()); accs["ts"].append(np.where(esc, samp[1], base_t).mean())
                    diffs += (cc - cs) / 1000
                d = np.array(accs["collab"]) - np.array(accs["samp"])
                rows[r] = dict(n=n, collab=float(np.mean(accs["collab"])), samp=float(np.mean(accs["samp"])),
                               het=float(np.mean(accs["het"])), tok_collab=float(np.mean(accs["tc"])),
                               tok_ratio=float(np.mean(accs["tc"]) / np.mean(accs["ts"])),
                               diff=[float(d.mean()), *map(float, np.percentile(d, [2.5, 97.5]))])
                continue
            esc = np.zeros(Q, bool); esc[ordr[:n]] = True
            cc, cs, ch = (np.where(esc, x[0], base_c) for x in (collab, samp, het))
            tc, ts = np.where(esc, collab[1], base_t), np.where(esc, samp[1], base_t)
            rows[r] = dict(n=n, collab=float(cc.mean()), samp=float(cs.mean()), het=float(ch.mean()),
                           tok_collab=float(tc.mean()), tok_ratio=float(tc.mean() / ts.mean()), diff=boot(cc - cs, rng),
                           diff_het=boot(cc - ch, rng), prec=float(np.mean(collab[0][esc] - samp[0][esc])) if n else None)
        out[name] = rows
    return out


def fmt_curves(cv, title):
    L = [title, f"    {'selector':12s} {'rate':>5} {'n':>4} {'Collab':>6} {'Samp':>6} {'Het':>6} {'tok/q':>7} {'ratio':>5}  Collab − Sampling [95%]"]
    for name, rows in cv.items():
        for r, o in rows.items():
            L.append(f"    {name:12s} {r*100:>4.0f}% {o['n']:>4} {o['collab']:>6.3f} {o['samp']:>6.3f} {o['het']:>6.3f} "
                     f"{o['tok_collab']:>7.0f} {o['tok_ratio']:>5.3f}  {pp(o['diff'])}")
    return L


def e3():
    rng = np.random.default_rng(SEED)
    res, L = {}, ["=== ARR2-E3 router rate–accuracy–cost curves ==="]
    # PILOT (cross-fitted)
    d = pilot(QWEN)
    Y, yH, qC, qA, cost = pilot_oof(d)
    X = d["X"]
    base_a = np.where(qA > .5, 1, 0)
    base_c, base_t = Y[np.arange(len(Y)), base_a], cost[base_a]
    collab, samp, het = (Y[:, 2], np.full(len(Y), cost[2])), (Y[:, 1], np.full(len(Y), cost[1])), \
        (yH, np.full(len(Y), d["tok"]["S"] + 4 * d["tok"]["SC"] + 22 * d["tok"]["HET"]))
    ent, pmax = X[:, F.NAMES.index("probe_entropy")], X[:, F.NAMES.index("probe_max_share")]
    rlen = X[:, F.NAMES.index("log_reasoning_chars")]
    sel = lambda qc: {"q_C": order(qc), "uncertainty": order(ent, pmax), "difficulty": order(rlen), "random": None}
    cvP = curves(base_c, base_t, collab, samp, het, sel(qC), rng)
    res["PILOT_curves"] = cvP
    # τ* on PILOT
    tau_rows = {}
    for tau in TAUS:
        esc = qC > tau
        cc, cs = np.where(esc, collab[0], base_c), np.where(esc, samp[0], base_c)
        ratio = np.where(esc, collab[1], base_t).mean() / np.where(esc, samp[1], base_t).mean()
        tau_rows[tau] = dict(n=int(esc.sum()), diff=float((cc - cs).mean()), ratio=float(ratio))
    ok = [t for t in TAUS if 0.9 <= tau_rows[t]["ratio"] <= 1.1]
    best = max(ok, key=lambda t: (round(tau_rows[t]["diff"], 12), t))
    res["PILOT_tau"] = dict(rows=tau_rows, tau_star=best)
    L += [f"--- PILOT (n=300), cross-fitted q_C (OOF reproduction of crossfit_vc OK); escalated → M1 / SC-22 / HET-22; others keep q_A rule",
          "    τ grid on PILOT: " + "  ".join(f"{t}: n={o['n']}, Δ={o['diff']*100:+.2f}pp, ratio {o['ratio']:.3f}" for t, o in tau_rows.items()),
          f"    τ* (max PILOT Collab − Sampling, ratio in [0.9, 1.1], ties → higher τ) = {best}"]
    L += fmt_curves(cvP, "    PILOT curves (expected accuracy over runs; cross-fitted, out-of-sample w.r.t. the router fit)")
    # HELD-OUT
    h = heldout()
    c, t, X = h["c"], h["t"], h["X"]
    a, qCh, qAh = RP.act(RP.load(), X)
    probe = t["SC"] * 4 / 22
    base_a = np.where(qAh > .5, 1, 0)
    base_c = np.where(base_a == 1, c["SC"], c["S"]); base_t = np.where(base_a == 1, t["S"] + t["SC"], t["S"] + probe)
    collab, samp, het = (c["M"], t["S"] + probe + t["M"]), (c["SC"], t["S"] + t["SC"]), (c["HET"], t["S"] + probe + t["HET"])
    ent, pmax = X[:, F.NAMES.index("probe_entropy")], X[:, F.NAMES.index("probe_max_share")]
    rlen = X[:, F.NAMES.index("log_reasoning_chars")]
    cvH = curves(base_c, base_t, collab, samp, het, sel(qCh), rng)
    res["HELDOUT_curves_exploratory"] = cvH
    esc = qCh > best
    cc, cs, ch = (np.where(esc, x[0], base_c) for x in (collab, samp, het))
    tc, ts = np.where(esc, collab[1], base_t), np.where(esc, samp[1], base_t)
    frz = np.select([a == 0, a == 1], [c["S"], c["SC"]], c["M"])
    res["HELDOUT_tau_star"] = dict(tau=best, n=int(esc.sum()), collab=float(cc.mean()), samp=float(cs.mean()), het=float(ch.mean()),
                                   diff=boot(cc - cs, rng), diff_het=boot(cc - ch, rng), tok=float(tc.mean()),
                                   tok_ratio=float(tc.mean() / ts.mean()), frozen_router_acc=float(frz.mean()))
    o = res["HELDOUT_tau_star"]
    L += ["", f"--- HELD-OUT (n=339). PILOT-selected τ* = {best} applied once (post-hoc, quasi-prospective; NOT prospective):",
          f"    COLLAB n = {o['n']} ({o['n']/339*100:.1f}%); Ours-Collab {o['collab']:.3f} vs Ours-Sampling {o['samp']:.3f}: {pp(o['diff'])}; "
          f"vs Ours-Hetero {o['het']:.3f}: {pp(o['diff_het'])}; tokens {o['tok']:.0f}/q, ratio {o['tok_ratio']:.3f} (frozen router acc {o['frozen_router_acc']:.3f})"]
    L += fmt_curves(cvH, "    HELD-OUT curves — EXPLORATORY (rates not selected on PILOT; for description only)")
    return res, L + [""]


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, fn in (("E1", e1), ("E2", e2), ("E3", e3)):
        res, L = fn()
        open(f"{OUT}/{name}.txt", "w").write("\n".join(L) + "\n")
        json.dump(res, open(f"{OUT}/{name}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
        print("\n".join(L))


if __name__ == "__main__":
    main()
