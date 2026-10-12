"""Sealed Phase VI analysis (PHASE6_FREEZE.md §4). Only script that reads labels; run once.
  python sealed_analysis.py  ->  studies/6_medarc_matched_ablation/results/RESULTS.{txt,json}"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..", "..")
sys.path.insert(0, os.path.join(HERE, "..", "..", "5_medarc_transfer", "code")); sys.path.insert(0, os.path.join(HERE, "..", "..", "2_complementarity_router", "code"))
import aggregate as AG        # noqa: E402
import parsers as P           # noqa: E402
import oracle_analysis as OA  # noqa: E402

D, K, B, SEED, EQ = f"{ROOT}/studies/6_medarc_matched_ablation/results/main", 25, 10000, 0, 0.10


def jl(p):
    return [json.loads(l) for l in open(p)]


def tok(r):
    return sum(c["input_tokens"] + c["output_tokens"] for c in r["calls"])


def boot(d, rng):
    d = np.asarray(d, float)
    bs = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return [float(d.mean()), *map(float, np.percentile(bs, [2.5, 97.5]))]


def main():
    rng = np.random.default_rng(SEED)
    labels = json.load(open(f"{ROOT}/studies/6_medarc_matched_ablation/results/labels.json"))
    arms = {m: [sorted(jl(f"{D}/{m}_run{r}.jsonl"), key=lambda x: x["idx"]) for r in (1, 2, 3)] for m in "ABC"}
    cot = [sorted(jl(f"{D}/cot_run{r}.jsonl"), key=lambda x: x["idx"]) for r in (1, 2, 3)]
    sc = [sorted(jl(f"{D}/sc_run{r}.jsonl"), key=lambda x: x["idx"]) for r in range(1, 7)]
    for runs in list(arms.values()) + [cot, sc]:
        assert all(len(run) == 472 and all(x["error"] is None for x in run) for run in runs)
    gold = np.array([labels[str(x["pubid"])] for x in arms["A"][0]]); Q = len(gold)
    corr = lambda preds: np.array([p == g for p, g in zip(preds, gold)], float)
    acc = {m: np.array([corr([AG.a_paper_no_reinit(x)[0] for x in run]) for run in arms[m]]) for m in "ABC"}
    r2 = {m: np.array([corr([AG.a_majority(x, 2) for x in run]) for run in arms[m]]).mean(0) for m in "ABC"}
    bs = np.array([corr([P.answer(x["text"]) for x in run]) for run in cot])
    votes = [np.array([[P.answer(t) or "" for t in x["texts"]] for x in run]) for run in sc]
    p_sc = OA.p_sc(np.concatenate(votes[:3], 1), gold, K, np.random.default_rng(SEED))
    mean = {m: acc[m].mean(0) for m in "ABC"}
    res = dict(n=Q, accuracy={m: float(mean[m].mean()) for m in "ABC"} | dict(SC25=float(p_sc.mean()), B_strong=float(bs.mean())),
               R2MV={m: float(r2[m].mean()) for m in "ABC"})
    res["primary_A_minus_B"] = boot(mean["A"] - mean["B"], rng)
    res["robust"] = dict(A_minus_C=boot(mean["A"] - mean["C"], rng), B_minus_C=boot(mean["B"] - mean["C"], rng))
    res["R2MV_A_minus_B"] = boot(r2["A"] - r2["B"], rng)
    # coverage
    cov = {}
    for m in "ABC":
        kg, kp, gn, ls = [], [], [], []
        for qi in range(Q):
            g = gold[qi]; per = []
            for run in arms[m]:
                ans = AG.controlled(run[qi])[0]
                per.append((g in {a[0] for a in ans}, g in {a[2] for a in ans}))
            kg.append(np.mean([x for x, _ in per])); kp.append(np.mean([y for _, y in per]))
            gn.append(np.mean([(not x) and y for x, y in per])); ls.append(np.mean([x and (not y) for x, y in per]))
        cov[m] = dict(kg=np.array(kg), kp=np.array(kp), gain=np.array(gn), loss=np.array(ls))
    res["coverage"] = {m: dict(kappa_gen=float(c["kg"].mean()), kappa_post=float(c["kp"].mean()), gain=float(c["gain"].mean()),
                               loss=float(c["loss"].mean()), delta=boot(c["kp"] - c["kg"], rng)) for m, c in cov.items()}
    res["coverage_A_minus_B"] = dict(kappa_post=boot(cov["A"]["kp"] - cov["B"]["kp"], rng),
                                     delta_kappa=boot((cov["A"]["kp"] - cov["A"]["kg"]) - (cov["B"]["kp"] - cov["B"]["kg"]), rng),
                                     kappa_gen_identical=bool(np.allclose(cov["A"]["kg"], cov["B"]["kg"]) and np.allclose(cov["A"]["kg"], cov["C"]["kg"])))
    # revision (pooled transitions, cluster by question)
    def revision(m):
        wn, wd, cn, cd, sw, n = (np.zeros(Q) for _ in range(6))
        for qi in range(Q):
            g = gold[qi]
            for run in arms[m]:
                ans = AG.controlled(run[qi])[0]
                for t in (0, 1):
                    for i in range(3):
                        cur, nxt = ans[i][t], ans[i][t + 1]; oth = [ans[j][t] for j in range(3) if j != i]
                        n[qi] += 1; sw[qi] += cur != nxt
                        if cur != g and any(x == g for x in oth):
                            wd[qi] += 1; wn[qi] += nxt == g
                        if cur == g and any(x != g for x in oth):
                            cd[qi] += 1; cn[qi] += nxt != g
        idx = rng.integers(0, Q, (B, Q))
        bsd = wn[idx].sum(1) / np.maximum(wd[idx].sum(1), 1) - cn[idx].sum(1) / np.maximum(cd[idx].sum(1), 1)
        brr, hrr = wn.sum() / max(wd.sum(), 1), cn.sum() / max(cd.sum(), 1)
        di = [float(brr - hrr), *map(float, np.percentile(bsd, [2.5, 97.5]))]
        lab = "truth-discriminating" if di[1] > 0 else "non-discriminating" if (di[1] >= -EQ and di[2] <= EQ) else "inconclusive"
        return dict(BRR=float(brr), HRR=float(hrr), DI=di, label=lab, switch=float(sw.sum() / n.sum()), n_W=int(wd.sum()), n_C=int(cd.sum()))
    res["revision"] = {m: revision(m) for m in "ABC"}
    # G_int and C*
    res["G_int"] = {m: boot(mean[m] - p_sc, rng) for m in "ABC"}
    tkS = float(np.mean([tok(x) for run in cot for x in run])); tkSC = float(np.mean([tok(x) / K for run in sc[:3] for x in run]))
    res["C_star"] = {}
    for m in "ABC":
        tkM = float(np.mean([tok(x) for run in arms[m] for x in run]))
        g2 = OA.analyze(bs, np.concatenate(votes[:3], 1), gold, acc[m], dict(S=tkS, SC=tkSC, M=tkM), np.zeros((Q, 2)),
                        k=K, placebo_votes=np.concatenate(votes[3:], 1), pool_split=2 * K)
        res["C_star"][m] = dict(H=g2["H"], H_plc=g2["H_plc"], C_star=g2["H_minus_H_plc"])
    res["tokens"] = {m: float(np.mean([tok(x) for run in arms[m] for x in run])) for m in "ABC"} | dict(SC25=float(np.mean([tok(x) for run in sc for x in run])), B_strong=tkS)
    res["none_pass"] = {m: float(np.mean([AG.a_paper_no_reinit(x)[1] for run in arms[m] for x in run])) for m in "ABC"}
    res["unparsed_turns"] = {m: float(np.mean([P.answer(t) is None for run in arms[m] for x in run for a in AG.turns(x) for t in a])) for m in "ABC"}
    pp = lambda v: f"{v[0]*100:+.1f} pp [{v[1]*100:+.1f}, {v[2]*100:+.1f}]"
    L = [f"=== Phase VI: matched interaction ablation, MedARC homogeneous Qwen2.5-14B, PubMedQA PQA-L fresh n={Q} ===",
         "accuracy (A_paper_no-reinit, mean of 3 runs): " + ", ".join(f"{k} {v:.3f}" for k, v in res["accuracy"].items()),
         f"PRIMARY  Acc(A) − Acc(B) = {pp(res['primary_A_minus_B'])}",
         f"key secondary  κ_post(A) − κ_post(B) = {pp(res['coverage_A_minus_B']['kappa_post'])};  Δκ(A) − Δκ(B) = {pp(res['coverage_A_minus_B']['delta_kappa'])};  κ_gen identical across arms: {res['coverage_A_minus_B']['kappa_gen_identical']}"]
    for m in "ABC":
        c = res["coverage"][m]; r = res["revision"][m]
        L.append(f"  arm {m}: κ_gen {c['kappa_gen']:.3f} → κ_post {c['kappa_post']:.3f} (gain {c['gain']:.3f}, loss {c['loss']:.3f}); "
                 f"BRR {r['BRR']:.3f} HRR {r['HRR']:.3f} DI {pp(r['DI'])} [{r['label']}] switch {r['switch']:.3f}; "
                 f"G_int {pp(res['G_int'][m])}; C* {pp(res['C_star'][m]['C_star'])} (H {res['C_star'][m]['H'][0]*100:+.1f}); "
                 f"R2MV {res['R2MV'][m]:.3f}; tokens/q {res['tokens'][m]:.0f}; none-pass {res['none_pass'][m]:.3f}; unparsed {res['unparsed_turns'][m]:.3f}")
    L += [f"robustness  A − C {pp(res['robust']['A_minus_C'])};  B − C {pp(res['robust']['B_minus_C'])};  R2MV A − B {pp(res['R2MV_A_minus_B'])}",
          f"tokens/q: SC-25 {res['tokens']['SC25']:.0f}, B_strong {res['tokens']['B_strong']:.0f}"]
    open(f"{ROOT}/studies/6_medarc_matched_ablation/results/RESULTS.txt", "w").write("\n".join(L) + "\n")
    json.dump(res, open(f"{ROOT}/studies/6_medarc_matched_ablation/results/RESULTS.json", "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
