"""Sealed Phase III analysis (PHASE3_FREEZE.md §6). The only Phase III script that reads labels.
Committed before any PILOT output exists; FINAL uses it unchanged with --split final.

  python sealed_analysis.py --split pilot   ->  studies/3_disagreement_intervention/results/<split>/RESULTS.{txt,json}
"""
import argparse
import glob
import json
import os
from collections import Counter

import numpy as np

from common import ROOT, jl
from run_tdc import TRIGGERED

B, SEED = 10000, 0


def boot(d, rng):
    d = np.asarray(d, float)
    bs = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return float(d.mean()), *map(float, np.percentile(bs, [2.5, 97.5]))


def tot(t):
    return t["input_tokens"] + t["output_tokens"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    a = ap.parse_args()
    d = f"{ROOT}/studies/3_disagreement_intervention/results/{a.split}/main"
    gold = {int(k): v for k, v in json.load(open(f"{ROOT}/studies/3_disagreement_intervention/results/splits/{a.split}_labels.json")).items()}
    P = jl(f"{d}/probe.jsonl")
    trig = [p for p in P if p["state"] in TRIGGERED]
    qids = [p["qid"] for p in trig]
    arms = {name: {r["qid"]: r for r in jl(f"{d}/{f}.jsonl")} for name, f in
            (("TDC", "tdc"), ("SampleMore", "sample_more"), ("HeteroMore", "het"))}
    m1q = json.load(open(f"{d}/m1_input/qids.json"))
    m1 = {}
    for sd in glob.glob(f"{d}/M1/shard*"):
        s = int(os.path.basename(sd)[5:])
        rows = [json.loads(l) for l in open(f"{sd}/chatgpt-syn_verif")]
        cost = [json.loads(l) for l in open(f"{sd}/cost.jsonl")]
        for k, (r, c) in enumerate(zip(rows, cost)):
            m1[m1q[s * 20 + k]] = dict(final_answer=r.get("pred_answer", ""),
                                       tokens=dict(input_tokens=c["prompt_tokens"], output_tokens=c["completion_tokens"]))
    arms["Full MAS"] = m1
    assert all(set(qids) <= set(v) for v in arms.values()), "missing arm outputs"
    g = np.array([gold[q] for q in qids])
    C = {n: np.array([v[q]["final_answer"] == gold[q] for q in qids], float) for n, v in arms.items()}
    T = {n: np.array([tot(v[q]["tokens"]) for q in qids], float) for n, v in arms.items()}
    rng = np.random.default_rng(SEED)
    pp = lambda v: f"{v[0]*100:+.1f} pp [{v[1]*100:+.1f}, {v[2]*100:+.1f}]"
    L, res = [], dict(split=a.split, n_questions=len(P), n_triggered=len(qids))

    # P1 / P2 / P3
    dP1 = boot(C["TDC"] - C["SampleMore"], rng)
    ratio = T["TDC"].mean() / T["SampleMore"].mean()
    rescue = int(((C["SampleMore"] == 0) & (C["TDC"] == 1)).sum())
    harm = int(((C["SampleMore"] == 1) & (C["TDC"] == 0)).sum())
    P1 = dP1[0] >= 0.02 and dP1[1] > 0
    P2 = 0.9 <= ratio <= 1.1
    res.update(P1=dict(diff=dP1, passed=P1), P2=dict(ratio=ratio, passed=P2),
               P3=dict(rescue=rescue, harm=harm, passed=rescue > harm),
               decision="GO" if P1 and P2 else "NO-GO" + ("" if P2 else " (comparison not compute-matched)"))
    L += [f"=== Phase III {a.split}: {len(P)} questions, {len(qids)} triggered ===",
          f"P1  Acc(TDC) − Acc(SampleMore) = {pp(dP1)}   (GO needs ≥ +2.0 pp and CI lower > 0) -> {'PASS' if P1 else 'FAIL'}",
          f"P2  incremental tokens TDC / SampleMore = {ratio:.3f}   (window [0.9, 1.1]) -> {'PASS' if P2 else 'FAIL'}",
          f"P3  rescue {rescue}  vs  harm {harm}   -> {'Rescue > Harm' if rescue > harm else 'Rescue ≤ Harm'}",
          f"DECISION: {res['decision']}", ""]

    # Table 1
    L.append("Table 1 — triggered questions: condition, accuracy, incremental tokens/q, Δ vs SampleMore [95% CI]")
    res["table1"] = {}
    for n in ("SampleMore", "HeteroMore", "TDC", "Full MAS"):
        dv = boot(C[n] - C["SampleMore"], rng) if n != "SampleMore" else None
        res["table1"][n] = dict(acc=float(C[n].mean()), tokens=float(T[n].mean()), diff=dv)
        L.append(f"  {n:11s} {C[n].mean():.3f}  {T[n].mean():8.0f}   " + (pp(dv) if dv else "—"))
    for lab, x, y in (("TDC − HeteroMore", "TDC", "HeteroMore"), ("HeteroMore − SampleMore", "HeteroMore", "SampleMore")):
        v = boot(C[x] - C[y], rng); res[lab] = v
        L.append(f"  {lab}: {pp(v)}")
    L.append("")

    # Table 2
    st = np.array([p["state"] for p in trig])
    L.append("Table 2 — by state: n, SampleMore, TDC, Δ [95% CI]")
    res["table2"] = {}
    for s in ("weak", "split", "fragmented"):
        m = st == s
        if m.sum() == 0:
            continue
        v = boot(C["TDC"][m] - C["SampleMore"][m], rng)
        res["table2"][s] = dict(n=int(m.sum()), sm=float(C["SampleMore"][m].mean()), tdc=float(C["TDC"][m].mean()), diff=v)
        L.append(f"  {s:10s} n={m.sum():4d}  {C['SampleMore'][m].mean():.3f}  {C['TDC'][m].mean():.3f}  {pp(v)}")
    L.append("")

    # Table 3
    bc = int(((C["SampleMore"] == 1) & (C["TDC"] == 1)).sum()); bw = int(((C["SampleMore"] == 0) & (C["TDC"] == 0)).sum())
    res["table3"] = dict(rescue=rescue, harm=harm, both_correct=bc, both_wrong=bw)
    L += ["Table 3 — transitions SampleMore → TDC",
          f"  SampleMore wrong → TDC correct (rescue) {rescue}", f"  SampleMore correct → TDC wrong (harm)   {harm}",
          f"  both correct {bc}", f"  both wrong   {bw}", ""]

    # diagnostics
    TD = [arms["TDC"][q] for q in qids]
    D = [r for r in TD if not r.get("single_answer") and r["error"] is None]
    parse = sum(bool(x) for p in P for x in p["answers"]) / (4 * len(P))
    conv = lambda rs: np.mean([r["flow"]["joint"].startswith("converge") for r in rs]) if rs else float("nan")
    res["diag"] = dict(
        states=dict(Counter(p["state"] for p in P)), probe_parse=parse,
        extraction_parsed=float(np.mean([r["disagreement"]["parsed_all"] for r in D])) if D else None,
        core_fallback=float(np.mean([r["disagreement"]["core_fallback"] for r in D])) if D else None,
        claim_missing=float(np.mean([r["disagreement"]["claim_x_missing"] or r["disagreement"]["claim_y_missing"] for r in D])) if D else None,
        revision_e3_failed=float(np.mean([r["revision_x"]["e3_failed"] for r in D] + [r["revision_y"]["e3_failed"] for r in D])) if D else None,
        single_answer=sum(bool(r.get("single_answer")) for r in TD), tdc_errors=sum(r["error"] is not None for r in TD),
        flow_x=dict(Counter(r["flow"]["x"] for r in D)), flow_y=dict(Counter(r["flow"]["y"] for r in D)),
        flow_joint=dict(Counter(r["flow"]["joint"] for r in D)),
        convergence=dict(all=conv(D), **{s: conv([r for r in D if r["state"] == s]) for s in ("weak", "split", "fragmented")}))
    sys_c = lambda arm: np.mean([(arms[arm][p["qid"]]["final_answer"] if p["state"] in TRIGGERED else
                                  (p["answers"][0] if p["state"] == "unanimous" else "")) == gold[p["qid"]] for p in P])
    res["overall"] = dict(TDC_system=float(sys_c("TDC")), SampleMore_system=float(sys_c("SampleMore")))
    dg = res["diag"]
    L += ["Diagnostics",
          f"  states {dg['states']}; probe parse success {dg['probe_parse']:.3f}",
          f"  extraction parsed {dg['extraction_parsed']}, CORE fallback {dg['core_fallback']}, claim missing {dg['claim_missing']}, "
          f"revision E3 failed {dg['revision_e3_failed']}, single-answer {dg['single_answer']}, TDC errors {dg['tdc_errors']}",
          f"  flow X {dg['flow_x']}", f"  flow Y {dg['flow_y']}", f"  joint {dg['flow_joint']}",
          f"  P(convergence) {dg['convergence']}",
          f"  overall accuracy (all {len(P)} questions): TDC system {res['overall']['TDC_system']:.3f}, "
          f"SampleMore system {res['overall']['SampleMore_system']:.3f}"]
    open(f"{ROOT}/studies/3_disagreement_intervention/results/{a.split}/RESULTS.txt", "w").write("\n".join(L) + "\n")
    json.dump(res, open(f"{ROOT}/studies/3_disagreement_intervention/results/{a.split}/RESULTS.json", "w"), indent=1,
              default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("\n".join(L))


if __name__ == "__main__":
    main()
