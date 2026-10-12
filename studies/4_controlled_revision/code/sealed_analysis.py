"""Sealed Phase IV analysis (PHASE4_FREEZE.md §3–4). The only Phase IV script that reads labels.
Committed before the full run; run once.

  python studies/4_controlled_revision/code/sealed_analysis.py [--dir studies/4_controlled_revision/results/main]  ->  <dir>/RESULTS.{txt,json}, fig_brr_hrr.{pdf,png}
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "3_disagreement_intervention", "code"))
import common as C3          # noqa: E402
import aggregate as AG       # noqa: E402
from run_tdc import TRIGGERED  # noqa: E402

ROOT = C3.ROOT
B, SEED, EQ = 10000, 0, 0.10
CONDS = ("control", "blinded", "opponent", "irrelevant")


def boot(d, rng):
    d = np.asarray(d, float)
    bs = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return [float(d.mean()), *map(float, np.percentile(bs, [2.5, 97.5]))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=f"{ROOT}/studies/4_controlled_revision/results/main")
    ap.add_argument("--labels", default=f"{ROOT}/studies/3_disagreement_intervention/results/splits/final_labels.json")
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    gold = {int(k): v for k, v in json.load(open(a.labels)).items()}
    P = {p["qid"]: p for p in C3.jl(f"{a.dir}/probe.jsonl")}
    R = [r for r in C3.jl(f"{a.dir}/revisions.jsonl")]
    assert all(r["error"] is None for r in R), "revision errors present"
    trig = [p for p in P.values() if p["state"] in TRIGGERED]

    # ---- discordant pairs
    D = []
    for r in R:
        g, x, y = gold[r["qid"]], r["extraction"]["X"], r["extraction"]["Y"]
        if (x == g) != (y == g):
            c, w = ("x", "y") if x == g else ("y", "x")
            D.append(dict(qid=r["qid"], state=r["state"], g=g, c=c, w=w, rev=r["revisions"]))
    b = {k: np.array([d["rev"][k][d["w"]]["after"] == d["g"] for d in D], float) for k in CONDS}
    h = {k: np.array([d["rev"][k][d["c"]]["after"] != d["g"] for d in D], float) for k in CONDS}
    sw = {k: np.array([np.mean([d["rev"][k][s]["after"] != d["rev"][k][s]["before"] for s in ("x", "y")]) for d in D])
          for k in CONDS}
    res = dict(n_questions=len(P), n_triggered=len(trig), n_two_answers=len(R), n_discordant=len(D))
    res["by_condition"] = {k: dict(BRR=boot(b[k], rng), HRR=boot(h[k], rng), DI=boot(b[k] - h[k], rng),
                                   switch=boot(sw[k], rng)) for k in CONDS}
    pairs = (("opponent", "control"), ("opponent", "blinded"), ("blinded", "control"))
    res["contrasts"] = {f"{u}-{v}": dict(dBRR=boot(b[u] - b[v], rng), dHRR=boot(h[u] - h[v], rng),
                                         dSwitch=boot(sw[u] - sw[v], rng)) for u, v in pairs}
    DI = res["by_condition"]["opponent"]["DI"]
    t1 = "truth-discriminating" if DI[1] > 0 else "non-discriminating" if (DI[1] >= -EQ and DI[2] <= EQ) else "inconclusive"
    oc = res["contrasts"]["opponent-control"]
    dB, dH = oc["dBRR"], oc["dHRR"]
    if dB[1] > 0 and dH[1] > 0:
        t2 = "mixed"
    elif dB[1] > 0 and (dH[2] <= 0 or dH[1] <= 0):
        t2 = "corrective"
    elif dH[1] > 0 and dB[1] <= 0:
        t2 = "corruptive"
    else:
        t2 = "no detectable effect"
    named = ("persuasion-consistent" if t1 == "non-discriminating" and t2 in ("corruptive", "mixed") else
             "correction-sensitive" if t1 == "truth-discriminating" and t2 == "corrective" else f"{t1} + {t2}")
    res["verdict"] = dict(truth_discrimination=t1, counterclaim_effect=t2, reading=named)
    res["DI_opp_by_state"] = {s: boot(b["opponent"][m] - h["opponent"][m], rng) if m.sum() else None
                              for s in ("weak", "split", "fragmented")
                              for m in [np.array([d["state"] == s for d in D])]}
    res["n_by_state"] = {s: int(sum(d["state"] == s for d in D)) for s in ("weak", "split", "fragmented")}

    # ---- coverage (all triggered with two parsed answers) and implementation diagnostics
    cov = dict(gold_in_pair=0, gold_outside=0)
    for r in R:
        cov["gold_in_pair" if gold[r["qid"]] in (r["extraction"]["X"], r["extraction"]["Y"]) else "gold_outside"] += 1
    res["coverage"] = dict(**cov, single_answer_triggered=len(trig) - len(R))
    res["diag"] = dict(
        probe_parse=sum(bool(x) for p in P.values() for x in p["answers"]) / (4 * len(P)),
        extraction_parsed=float(np.mean([r["extraction"]["parsed_all"] for r in R])),
        core_fallback=float(np.mean([r["extraction"]["core_fallback"] for r in R])),
        e3_failed={k: float(np.mean([r["revisions"][k][s]["e3_failed"] for r in R for s in ("x", "y")])) for k in CONDS},
        truncated={k: float(np.mean([r["revisions"][k][s]["finish_reason"] == "length" for r in R for s in ("x", "y")]))
                   for k in CONDS})

    # ---- exploratory: local vs system-level utility (opponent revisions + frozen TDC aggregation)
    acc = lambda f: float(np.mean([f(r) == gold[r["qid"]] for r in R]))
    loc_b = np.mean([r["revisions"]["opponent"][s]["after"] == gold[r["qid"]] and r["revisions"]["opponent"][s]["before"] != gold[r["qid"]]
                     for r in R for s in ("x", "y")])
    loc_h = np.mean([r["revisions"]["opponent"][s]["after"] != gold[r["qid"]] and r["revisions"]["opponent"][s]["before"] == gold[r["qid"]]
                     for r in R for s in ("x", "y")])
    res["exploratory"] = dict(
        per_revision_BRR_minus_HRR_opponent=float(loc_b - loc_h),
        probe_majority_acc=acc(lambda r: AG.majority(P[r["qid"]]["answers"])),
        tdc_final_acc_opponent=acc(lambda r: AG.tdc_final(P[r["qid"]]["answers"], r["revisions"]["opponent"]["x"]["after"],
                                                          r["revisions"]["opponent"]["y"]["after"])),
        revised_votes_only_acc_opponent=acc(lambda r: AG.majority([r["revisions"]["opponent"]["x"]["after"],
                                                                   r["revisions"]["opponent"]["y"]["after"]])))

    # ---- report
    pp = lambda v: "n/a" if v is None else f"{v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]"
    L = [f"=== Phase IV: {len(P)} questions, {len(trig)} triggered, {len(R)} with two parsed answers, {len(D)} discordant pairs ===",
         f"PRIMARY  DI_opp = BRR_opp − HRR_opp = {pp(DI)}",
         f"VERDICT  truth discrimination: {t1};  counterclaim effect (Opponent vs Control): {t2};  reading: {named}", "",
         "condition    BRR (W→C)                 HRR (C→W)                 DI                         switch"]
    for k in CONDS:
        v = res["by_condition"][k]
        L.append(f"{k:11s}  {pp(v['BRR'])}  {pp(v['HRR'])}  {pp(v['DI'])}  {pp(v['switch'])}")
    L += ["", "contrasts                ΔBRR                      ΔHRR                      Δswitch"]
    for k, v in res["contrasts"].items():
        L.append(f"{k:22s}  {pp(v['dBRR'])}  {pp(v['dHRR'])}  {pp(v['dSwitch'])}")
    L += ["", "DI_opp by state: " + "; ".join(f"{s} (n={res['n_by_state'][s]}) {pp(v)}" for s, v in res["DI_opp_by_state"].items()),
          f"coverage: {res['coverage']}",
          f"diagnostics: probe parse {res['diag']['probe_parse']:.3f}, extraction parsed {res['diag']['extraction_parsed']:.3f}, "
          f"CORE fallback {res['diag']['core_fallback']:.3f}, E3 failed {res['diag']['e3_failed']}, truncated {res['diag']['truncated']}",
          "", "EXPLORATORY (local vs system level; sensitivity only): " + json.dumps(res["exploratory"])]
    open(f"{a.dir}/RESULTS.txt", "w").write("\n".join(L) + "\n")
    json.dump(res, open(f"{a.dir}/RESULTS.json", "w"), indent=1)
    print("\n".join(L))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    xs = np.arange(3)
    for key, col, mk, lab in (("BRR", "#2a78d6", "o", "wrong → correct (BRR)"), ("HRR", "#eb6834", "s", "correct → wrong (HRR)")):
        v = np.array([res["by_condition"][k][key] for k in CONDS[:3]])
        ax.errorbar(xs, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], color=col, marker=mk, lw=2, capsize=3, label=lab)
        iv = res["by_condition"]["irrelevant"][key]
        ax.errorbar([3], [iv[0]], yerr=[[iv[0] - iv[1]], [iv[2] - iv[0]]], color=col, marker=mk, mfc="white", ls="none", capsize=3)
    ax.set_xticks(range(4), ["Control", "Blinded\ncritique", "Opponent\nclaim", "Irrelevant\nclaim"])
    ax.set_ylabel("rate (discordant pairs)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{a.dir}/fig_brr_hrr.{ext}", dpi=300)


if __name__ == "__main__":
    main()
