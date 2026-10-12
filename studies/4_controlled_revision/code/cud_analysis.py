"""Collaboration Utility Decomposition (CUD) over the frozen Part I / Phase II / Phase III results.
Post-hoc, descriptive; no LLM calls; nothing here feeds back into any frozen decision.

  G_app       Acc(M) − Acc(B_reported)                 apparent gain
  G_base      Acc(M) − Acc(S_strong)                   baseline-adjusted gain
  G_int       Acc(M) − Acc(SC at ≈ M's tokens)         compute-adjusted interaction gain
  C_oracle    cross-fitted empirical-oracle headroom of adding M to {S, SC}   (Phase II H)
  C_realized  Acc(router with M) − Acc(same router, M -> SC), ≈ matched tokens (Phase II B / Phase III P1)
  CUG         C_oracle − C_realized                    complementarity–utility gap

Also Phase III revision-level utility (BRR, HRR) and rescue/harm by disagreement state.

  python experiments/cud/cud_analysis.py  ->  studies/4_controlled_revision/results/cud/CUD.md, studies/4_controlled_revision/results/cud/CUD.json
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
sys.path.insert(0, f"{ROOT}/studies/2_complementarity_router/code")
sys.path.insert(0, f"{ROOT}/studies/3_disagreement_intervention/code")
RNG = np.random.default_rng(0)


def boot(d):
    d = np.asarray(d, float)
    bs = d[RNG.integers(0, len(d), (10000, len(d)))].mean(1)
    return [float(d.mean()), *map(float, np.percentile(bs, [2.5, 97.5]))]


def phase2():
    import pilot_analysis as P
    import oracle_analysis as OA
    S, SC, M, _ = P.load("Qwen2.5-14B-Instruct")
    gold = np.array([r["gold"] for r in S[0]])
    s = np.array([[r["E3"] == r["gold"] for r in run] for run in S], float).mean(0)
    m = np.array([[r["pred_answer"] == r["gold_answer"] for r in run] for run in M], float).mean(0)
    sc = OA.p_sc(np.concatenate([np.array([r["votes"] for r in run]) for run in SC[:3]], 1), gold, 22, RNG)
    R = json.load(open(f"{ROOT}/studies/2_complementarity_router/results/pilot/RESULTS.json"))["Qwen2.5-14B-Instruct"]
    H = json.load(open(f"{ROOT}/studies/2_complementarity_router/results/heldout/RESULTS.json"))
    return dict(G_base=boot(m - s), G_int=boot(m - sc), C_oracle=R["H"], C_oracle_placebo=R["H_plc"],
                C_realized=H["B"]["diff"], CUG_point=R["H"][0] - H["B"]["diff"][0])


def phase3():
    from common import jl
    d = f"{ROOT}/studies/3_disagreement_intervention/results/pilot/main"
    gold = {int(k): v for k, v in json.load(open(f"{ROOT}/studies/3_disagreement_intervention/results/splits/pilot_labels.json")).items()}
    P = {p["qid"]: p for p in jl(f"{d}/probe.jsonl")}
    T = {r["qid"]: r for r in jl(f"{d}/tdc.jsonl")}
    SM = {r["qid"]: r for r in jl(f"{d}/sample_more.jsonl")}
    HM = {r["qid"]: r for r in jl(f"{d}/het.jsonl")}
    q = list(T)
    c = lambda A: np.array([A[i]["final_answer"] == gold[i] for i in q], float)
    t, s, h = c(T), c(SM), c(HM)
    # naive (single-run, not cross-fitted) oracle headroom over SampleMore; HeteroMore as a placebo arm
    res = dict(n=len(q), G_int=boot(t - s), C_oracle_naive=float(np.mean((t == 1) & (s == 0))),
               C_oracle_naive_placebo_HET=float(np.mean((h == 1) & (s == 0))))
    res["C_realized"] = res["G_int"]
    res["CUG_point"] = res["C_oracle_naive"] - res["G_int"][0]
    # revision-level utility: each representative's pre -> post answer vs gold
    rev = []
    for i in q:
        r = T[i]
        if r.get("single_answer") or r["error"]:
            continue
        for side in ("revision_x", "revision_y"):
            pre, post = r[side]["before"] == gold[i], r[side]["after"] == gold[i]
            rev.append((r["state"], pre, post, r[side]["before"] != r[side]["after"]))
    rv = np.array([(p, o, sw) for _, p, o, sw in rev], float)
    res["revisions"] = dict(n=len(rev), switch_rate=float(rv[:, 2].mean()),
                            BRR=float(np.mean((rv[:, 0] == 0) & (rv[:, 1] == 1))),
                            HRR=float(np.mean((rv[:, 0] == 1) & (rv[:, 1] == 0))),
                            P_wrong_to_correct_given_wrong=float(np.mean(rv[rv[:, 0] == 0, 1])),
                            P_correct_to_wrong_given_correct=float(np.mean(1 - rv[rv[:, 0] == 1, 1])))
    res["revisions"]["NetRevisionUtility"] = res["revisions"]["BRR"] - res["revisions"]["HRR"]
    by = {}
    for st in ("weak", "split", "fragmented"):
        idx = [k for k, i in enumerate(q) if T[i]["state"] == st]
        rr = [x for x in rev if x[0] == st]
        by[st] = dict(n=len(idx), rescue=int(sum((t[k] == 1) and (s[k] == 0) for k in idx)),
                      harm=int(sum((t[k] == 0) and (s[k] == 1) for k in idx)),
                      BRR=float(np.mean([(not p) and o for _, p, o, _ in rr])), HRR=float(np.mean([p and (not o) for _, p, o, _ in rr])))
    res["by_state"] = by
    # was the correct answer initially X (majority / top-1) or Y (minority / top-2)?
    pos = dict(X=0, Y=0, neither=0)
    for i in q:
        r = T[i]
        pos["X" if r["X"] == gold[i] else "Y" if r["Y"] == gold[i] else "neither"] += 1
    res["gold_position"] = pos
    return res


def main():
    p1 = dict(  # Part I frozen numbers (runs/SYNTHESIS.txt, P0_RESULTS.txt, F_vs_S.txt, layer3, PARETO / R5)
        MedAgents_GPT=dict(G_app=[.107, .082, .131], G_base=[-.005, -.027, .015]),
        MDAgents_GPT=dict(G_base=[-.120, -.156, -.085]),
        MedLA_GPT=dict(G_app=[.007, -.013, .026], G_base=[-.048, -.065, -.031]),
        MedLA_Llama=dict(G_app=[.027, -.004, .057], G_base=[-.110, -.138, -.081]),
        MedAgents_Qwen14B_first200=dict(G_base=[-.040, None, None], G_int=[-.032, -.075, .010]),
        MedAgents_Llama8B_first200=dict(G_base=[-.095, -.150, -.040], G_int=[-.133, -.187, -.080]))
    out = dict(part1=p1, phase2_MedAgents_Qwen14B_MedQA=phase2(), phase3_TDC_Qwen14B_MedMCQA=phase3())
    json.dump(out, open(f"{ROOT}/studies/4_controlled_revision/results/cud/CUD.json", "w"), indent=1)
    f = lambda v: "—" if v is None else (f"{v[0]*100:+.1f}" + ("" if v[1] is None else f" [{v[1]*100:+.1f}, {v[2]*100:+.1f}]"))
    L = ["| System (backbone, data) | G_app | G_base | G_int | C_oracle | C_realized | CUG |", "|---|---|---|---|---|---|---|"]
    for k, v in p1.items():
        L.append(f"| {k} | {f(v.get('G_app'))} | {f(v.get('G_base'))} | {f(v.get('G_int'))} | — | — | — |")
    a = out["phase2_MedAgents_Qwen14B_MedQA"]
    L.append(f"| Phase II: MedAgents vs router (Qwen-14B, MedQA PILOT/HELD-OUT) | — | {f(a['G_base'])} | {f(a['G_int'])} | "
             f"{f(a['C_oracle'])} (placebo {a['C_oracle_placebo'][0]*100:+.1f}) | {f(a['C_realized'])} | {a['CUG_point']*100:+.1f} |")
    b = out["phase3_TDC_Qwen14B_MedMCQA"]
    L.append(f"| Phase III: TDC (Qwen-14B, MedMCQA PILOT, n={b['n']} triggered) | — | — | {f(b['G_int'])} | "
             f"{b['C_oracle_naive']*100:+.1f} naive (HET placebo {b['C_oracle_naive_placebo_HET']*100:+.1f}) | {f(b['C_realized'])} | {b['CUG_point']*100:+.1f} |")
    rv = b["revisions"]
    L += ["", f"Phase III revisions (n={rv['n']} representative revisions): switch {rv['switch_rate']:.3f}, BRR {rv['BRR']:.3f}, "
          f"HRR {rv['HRR']:.3f}, NetRevisionUtility {rv['NetRevisionUtility']*100:+.1f} pp; "
          f"P(W→C | W) {rv['P_wrong_to_correct_given_wrong']:.3f}, P(C→W | C) {rv['P_correct_to_wrong_given_correct']:.3f}",
          "", "| state | n | rescue | harm | BRR | HRR |", "|---|---|---|---|---|---|"]
    for st, v in b["by_state"].items():
        L.append(f"| {st} | {v['n']} | {v['rescue']} | {v['harm']} | {v['BRR']:.3f} | {v['HRR']:.3f} |")
    L += ["", f"gold among the top-2 answers on triggered questions: {b['gold_position']}"]
    open(f"{ROOT}/studies/4_controlled_revision/results/cud/CUD.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
