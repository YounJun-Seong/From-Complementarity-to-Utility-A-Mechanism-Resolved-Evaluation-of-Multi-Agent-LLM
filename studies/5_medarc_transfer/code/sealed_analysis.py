"""Sealed Phase V analysis (PHASE5_FREEZE.md §4–5). The only Phase V script that reads labels. Run once.

  python sealed_analysis.py  ->  studies/5_medarc_transfer/results/RESULTS.{txt,json}
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..", "..")
sys.path.insert(0, f"{ROOT}/studies/2_complementarity_router/code")
import oracle_analysis as OA   # noqa: E402
import aggregate as AG          # noqa: E402
import parsers as P             # noqa: E402
import released as R            # noqa: E402

D = f"{ROOT}/studies/5_medarc_transfer/results/main"
B, SEED, EQ = 10000, 0, 0.10


def jl(p):
    return [json.loads(l) for l in open(p)]


def tok(r):
    return sum(c["input_tokens"] + c["output_tokens"] for c in r["calls"])


def boot(d, rng):
    d = np.asarray(d, float)
    bs = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return [float(d.mean()), *map(float, np.percentile(bs, [2.5, 97.5]))]


def ratio_boot(num, den, rng):
    """Cluster bootstrap of sum(num)/sum(den) over questions (arrays are per-question sums)."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    idx = rng.integers(0, len(num), (B, len(num)))
    bs = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1e-12)
    return [float(num.sum() / den.sum()), *map(float, np.percentile(bs, [2.5, 97.5]))]


def main():
    rng = np.random.default_rng(SEED)
    K = json.load(open(f"{ROOT}/studies/5_medarc_transfer/results/K_FROZEN.json"))["k_star"]
    labels = json.load(open(f"{ROOT}/studies/5_medarc_transfer/results/labels.json"))
    M = [jl(f"{D}/medarc_run{r}.jsonl") for r in (1, 2, 3)]
    ZS = [jl(f"{D}/zeroshot_run{r}.jsonl") for r in (1, 2, 3)]
    COT = [jl(f"{D}/cot_run{r}.jsonl") for r in (1, 2, 3)]
    SC = [jl(f"{D}/sc_run{r}.jsonl") for r in range(1, 7)]
    for arm in M + ZS + COT + SC:
        assert len(arm) == 498 and all(x["error"] is None for x in arm), "incomplete or failed run"
    gold = np.array([labels[str(x["pubid"])] for x in M[0]])
    Q = len(gold)
    c = lambda preds: np.array([p == g for p, g in zip(preds, gold)], float)

    acc = {}
    acc["A_released"] = np.array([c([AG.a_released(x) for x in run]) for run in M])
    acc["A_released_controlled_parser"] = np.array([c([AG.a_released_controlled_parser(x) for x in run]) for run in M])
    paper = [[AG.a_paper_no_reinit(x) for x in run] for run in M]
    acc["A_paper_no_reinit"] = np.array([c([p for p, _ in run]) for run in paper])
    acc["A_R2MV"] = np.array([c([AG.a_majority(x, 2) for x in run]) for run in M])
    acc["A_R0MV"] = np.array([c([AG.a_majority(x, 0) for x in run]) for run in M])
    acc["B_released"] = np.array([c([R.released_zeroshot(x["text"]) for x in run]) for run in ZS])
    acc["B_strong"] = np.array([c([P.answer(x["text"]) for x in run]) for run in COT])
    votes = [np.array([[P.answer(t) or "" for t in x["texts"]] for x in run]) for run in SC]
    assert all(v.shape == (Q, K) for v in votes)
    p_sc = OA.p_sc(np.concatenate(votes[:3], 1), gold, K, np.random.default_rng(SEED))
    mean = {k: v.mean(0) for k, v in acc.items()}
    res = dict(k_star=K, n=Q, accuracy={k: float(v.mean()) for k, v in mean.items()} | {"SC-k*": float(p_sc.mean())})

    # G1
    res["G1"] = dict(G_app=boot(mean["A_released"] - mean["B_released"], rng),
                     G_base=boot(mean["A_paper_no_reinit"] - mean["B_strong"], rng),
                     G_int=boot(mean["A_paper_no_reinit"] - p_sc, rng), G_het="N/A")
    # G2 (unchanged Phase II analyze)
    tk = dict(S=float(np.mean([tok(x) for run in COT for x in run])),
              SC=float(np.mean([tok(x) / K for run in SC[:3] for x in run])),
              M=float(np.mean([tok(x) for run in M for x in run])))
    g2 = OA.analyze(acc["B_strong"], np.concatenate(votes[:3], 1), gold, acc["A_paper_no_reinit"], tk, np.zeros((Q, 2)),
                    k=K, placebo_votes=np.concatenate(votes[3:], 1), pool_split=2 * K)
    res["G2"] = dict(H=g2["H"], H_plc=g2["H_plc"], C_star=g2["H_minus_H_plc"], share_M_chosen=g2["share_M_chosen"])
    # G4
    kg, kp, gain, loss = [], [], [], []
    for qi in range(Q):
        a = [[AG.controlled(run[qi])[0][i] for i in range(3)] for run in M]       # per run: agent -> rounds
        g = gold[qi]
        A0 = [g in {ag[0] for ag in run} for run in a]
        A2 = [g in {ag[2] for ag in run} for run in a]
        kg.append(np.mean(A0)); kp.append(np.mean(A2))
        gain.append(np.mean([(not x) and y for x, y in zip(A0, A2)])); loss.append(np.mean([x and (not y) for x, y in zip(A0, A2)]))
    res["G4"] = dict(kappa_gen=boot(kg, rng), kappa_post=boot(kp, rng), delta_kappa=boot(np.array(kp) - np.array(kg), rng),
                     kappa_gain=boot(gain, rng), kappa_loss=boot(loss, rng))
    # G5 (per-question sums for the cluster bootstrap)
    obs = {t: dict(wn=np.zeros(Q), wd=np.zeros(Q), cn=np.zeros(Q), cd=np.zeros(Q), sw=np.zeros(Q), n=np.zeros(Q)) for t in (0, 1)}
    for qi in range(Q):
        g = gold[qi]
        for run in M:
            ans = AG.controlled(run[qi])[0]
            for t in (0, 1):
                for i in range(3):
                    cur, nxt = ans[i][t], ans[i][t + 1]
                    others = [ans[j][t] for j in range(3) if j != i]
                    o = obs[t]
                    o["n"][qi] += 1; o["sw"][qi] += cur != nxt
                    if cur != g and any(x == g for x in others):
                        o["wd"][qi] += 1; o["wn"][qi] += nxt == g
                    if cur == g and any(x != g for x in others):
                        o["cd"][qi] += 1; o["cn"][qi] += nxt != g

    def di(ts):
        wn = sum(obs[t]["wn"] for t in ts); wd = sum(obs[t]["wd"] for t in ts)
        cn = sum(obs[t]["cn"] for t in ts); cd = sum(obs[t]["cd"] for t in ts)
        idx = rng.integers(0, Q, (B, Q))
        bs = wn[idx].sum(1) / np.maximum(wd[idx].sum(1), 1) - cn[idx].sum(1) / np.maximum(cd[idx].sum(1), 1)
        brr, hrr = wn.sum() / max(wd.sum(), 1), cn.sum() / max(cd.sum(), 1)
        return dict(DI=[float(brr - hrr), *map(float, np.percentile(bs, [2.5, 97.5]))], BRR=float(brr), HRR=float(hrr),
                    n_W=int(wd.sum()), n_C=int(cd.sum()),
                    switch=ratio_boot(sum(obs[t]["sw"] for t in ts), sum(obs[t]["n"] for t in ts), rng))
    g5 = dict(pooled=di((0, 1)), r0_r1=di((0,)), r1_r2=di((1,)))
    d = g5["pooled"]["DI"]
    g5["verdict"] = "truth-discriminating" if d[1] > 0 else "non-discriminating" if (d[1] >= -EQ and d[2] <= EQ) else "inconclusive"
    res["G5"] = g5
    # G6
    res["G6"] = dict(R_int=boot(mean["A_R2MV"] - mean["A_R0MV"], rng),
                     paper_minus_R2MV=boot(mean["A_paper_no_reinit"] - mean["A_R2MV"], rng),
                     released_minus_R2MV=boot(mean["A_released"] - mean["A_R2MV"], rng),
                     parser_sensitivity_released=boot(mean["A_released"] - mean["A_released_controlled_parser"], rng),
                     none_pass_rate=float(np.mean([np_ for run in paper for _, np_ in run])))
    # descriptive
    unp = lambda texts: float(np.mean([P.answer(t) is None for t in texts]))
    res["descriptive"] = dict(
        paper_reported_deepseek=dict(zero_shot=0.729, medarc=0.772, gain=0.043),
        tokens_per_question=dict(medarc=tk["M"], B_strong=tk["S"], SC_k=float(np.mean([tok(x) for run in SC for x in run])),
                                 B_released=float(np.mean([tok(x) for run in ZS for x in run]))),
        realized_token_ratio_medarc_over_sc=tk["M"] / float(np.mean([tok(x) for run in SC for x in run])),
        unparsed_controlled=dict(medarc_turns=unp([t for run in M for x in run for a in AG.turns(x) for t in a]),
                                 B_strong=unp([x["text"] for run in COT for x in run]),
                                 SC=unp([t for run in SC for x in run for t in x["texts"]])),
        truncated=dict(medarc=float(np.mean([cl["finish"] == "length" for run in M for x in run for cl in x["calls"]])),
                       SC=float(np.mean([f == "length" for run in SC for x in run for f in x["finish"]]))))
    pp = lambda v: f"{v[0]*100:+.1f} pp [{v[1]*100:+.1f}, {v[2]*100:+.1f}]"
    a_ = res["accuracy"]
    L = [f"=== Phase V: MedARC (released homogeneous configuration, Qwen2.5-14B), PubMedQA n={Q}, k*={K} ===",
         "accuracy: " + ", ".join(f"{k} {v:.3f}" for k, v in a_.items()),
         f"G1  G_app (released-pipeline apparent gain, common backbone) {pp(res['G1']['G_app'])}",
         f"G1  G_base {pp(res['G1']['G_base'])}    G_int {pp(res['G1']['G_int'])}    G_het N/A",
         f"G2  H {pp(res['G2']['H'])}  H_plc {pp(res['G2']['H_plc'])}  C* {pp(res['G2']['C_star'])}",
         "G3  N/A (always-on)",
         f"G4  kappa_gen {res['G4']['kappa_gen'][0]:.3f}  kappa_post {res['G4']['kappa_post'][0]:.3f}  delta {pp(res['G4']['delta_kappa'])}  "
         f"gain {res['G4']['kappa_gain'][0]:.3f}  loss {res['G4']['kappa_loss'][0]:.3f}",
         f"G5  pooled DI {pp(g5['pooled']['DI'])} (BRR {g5['pooled']['BRR']:.3f}, HRR {g5['pooled']['HRR']:.3f}, n_W {g5['pooled']['n_W']}, n_C {g5['pooled']['n_C']}) -> {g5['verdict']}",
         f"    DI 0→1 {pp(g5['r0_r1']['DI'])}; DI 1→2 {pp(g5['r1_r2']['DI'])}; switch pooled {g5['pooled']['switch'][0]:.3f}",
         f"G6  R_int {pp(res['G6']['R_int'])}; paper−R2MV {pp(res['G6']['paper_minus_R2MV'])}; released−R2MV {pp(res['G6']['released_minus_R2MV'])}; "
         f"released parser − controlled parser {pp(res['G6']['parser_sensitivity_released'])}; none-pass {res['G6']['none_pass_rate']:.3f}",
         "descriptive: " + json.dumps(res["descriptive"])]
    open(f"{ROOT}/studies/5_medarc_transfer/results/RESULTS.txt", "w").write("\n".join(L) + "\n")
    json.dump(res, open(f"{ROOT}/studies/5_medarc_transfer/results/RESULTS.json", "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
