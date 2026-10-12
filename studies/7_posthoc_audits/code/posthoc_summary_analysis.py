"""E2′ analysis (POSTHOC_EXPERIMENTS.md P1–P4 + human check). Reads judge outputs and gold; no LLM calls.

  python experiments/posthoc_summary_analysis.py          ->  studies/7_posthoc_audits/results/SUMMARY_E2.{txt,json}
  python experiments/posthoc_summary_analysis.py human    ->  agreement on the filled human_check/sheet.csv
"""
import collections
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..", "..")
sys.path.insert(0, f"{HERE}/../../5_medarc_transfer/code"); sys.path.insert(0, f"{HERE}/../../2_complementarity_router/code")
import aggregate as AG          # noqa: E402
import oracle_analysis as OA    # noqa: E402
import parsers as P             # noqa: E402

B, SEED = 10000, 0
OUT = os.environ.get("POSTHOC_OUT", f"{ROOT}/studies/7_posthoc_audits/results")
LABELS = ("yes", "no", "maybe")


def jl(p):
    return [json.loads(l) for l in open(p)]


class Ratio:
    """Accumulate per-question numerator/denominator; cluster-bootstrap the pooled ratio."""
    def __init__(self):
        self.n, self.d = collections.defaultdict(float), collections.defaultdict(float)

    def add(self, q, hit, w=1.0):
        self.d[q] += w; self.n[q] += hit * w

    def ci(self, rng):
        qs = list(self.d)
        if not qs or sum(self.d.values()) == 0:
            return dict(p=None, n=0)
        n, d = np.array([self.n[q] for q in qs]), np.array([self.d[q] for q in qs])
        idx = rng.integers(0, len(qs), (B, len(qs)))
        bs = n[idx].sum(1) / np.maximum(d[idx].sum(1), 1e-12)
        return dict(p=float(n.sum() / d.sum()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
                    n=float(d.sum()), n_q=len(qs))


def load(tag):
    if tag == "VI_A":
        lab = json.load(open(f"{ROOT}/studies/6_medarc_matched_ablation/results/labels.json"))
        recs = [jl(f"{ROOT}/studies/6_medarc_matched_ablation/results/main/A_run{r}.jsonl") for r in (1, 2, 3)]
        sc = [f"{ROOT}/studies/6_medarc_matched_ablation/results/main/sc_run{r}.jsonl" for r in (1, 2, 3)]
        recs = [sorted(x, key=lambda z: z["idx"]) for x in recs]
        srt = lambda rows: sorted(rows, key=lambda z: z["idx"])
    else:
        lab = json.load(open(f"{ROOT}/studies/5_medarc_transfer/results/labels.json"))
        recs = [jl(f"{ROOT}/studies/5_medarc_transfer/results/main/medarc_run{r}.jsonl") for r in (1, 2, 3)]
        sc = [f"{ROOT}/studies/5_medarc_transfer/results/main/sc_run{r}.jsonl" for r in (1, 2, 3)]
        srt = lambda rows: rows
    gold = [lab[str(x["pubid"])] for x in recs[0]]
    votes = np.concatenate([np.array([[P.answer(t) or "" for t in x["texts"]] for x in srt(jl(p))]) for p in sc], 1)
    p_sc = OA.p_sc(votes, np.array(gold), 25, np.random.default_rng(SEED))
    return recs, gold, p_sc


def state(j1row):
    s, L, ev = j1row["src_label"], j1row["answer_in_summary"], j1row["evidence_preserved"]
    if s is None or L is None or ev is None:
        return None
    if L == s:
        return "conveyed+evidence" if ev in ("full", "partial") else "answer_only"
    return "not_conveyed"


def analyse(tag, j1, j2, rng):
    recs, gold, p_sc = load(tag)
    res = {}
    # sanity: judge's reading of the original's answer vs controlled parser
    rows = [r for r in j1.values() if r["set"] == tag and r["src_label"] and r["final_answer_original"]]
    res["sanity_parser_agreement"] = float(np.mean([r["final_answer_original"] == r["src_label"] for r in rows]))
    res["judge_unparsed"] = float(np.mean([state(r) is None for r in j1.values() if r["set"] == tag and r["src_label"]]))

    P1 = collections.defaultdict(Ratio)
    P2 = collections.defaultdict(Ratio); P2eng = collections.Counter()
    P3 = collections.defaultdict(Ratio)
    P4 = collections.defaultdict(Ratio); P4n = collections.Counter()
    conv = Ratio()
    for run_i, run in enumerate(recs, 1):
        for qi, rec in enumerate(run):
            g, q = gold[qi], qi
            ans = AG.controlled(rec)[0]
            ev = {}
            for r in (1, 2):
                for i in range(3):
                    srcs = [j1.get((tag, run_i, rec["idx"], r, i, j)) for j in (1, 2)]
                    ev[(r, i)] = srcs
                    st = [state(s) if s else None for s in srcs]
                    labs = [s["src_label"] for s in srcs]
                    disagree = labs[0] != labs[1]
                    # P1 preservation, per source
                    for s, x in zip(srcs, st):
                        if x is None:
                            continue
                        grp = "gold_src" if s["src_label"] == g else "wrong_src"
                        for k in ("conveyed+evidence", "answer_only", "not_conveyed"):
                            P1[f"{grp}|{k}"].add(q, x == k)
                            if disagree:
                                P1[f"{grp}|disagree|{k}"].add(q, x == k)
                        P1[f"{grp}|label_mentioned"].add(q, x != "not_conveyed")
                        P1[f"{grp}|misrepresented"].add(q, bool(s["misrepresented"]))
                        P1[f"{grp}|evidence_full"].add(q, s["evidence_preserved"] == "full")
                    said = [s["answer_in_summary"] if s else None for s in srcs]
                    gold_states = [x for s, x in zip(srcs, st) if s and s["src_label"] == g and x]
                    # P2 uptake: agent wrong at r-1, some source carries gold
                    if ans[i][r - 1] != g and gold_states:
                        best = min(gold_states, key=("conveyed+evidence", "answer_only", "not_conveyed").index)
                        P2[best].add(q, ans[i][r] == g); P2["all"].add(q, ans[i][r] == g)
                        for j in (1, 2):
                            e2 = j2.get((tag, run_i, rec["idx"], r, i, j))
                            if e2 and e2["label"] == g and e2["engagement"]:
                                P2eng[e2["engagement"]] += 1
                    # P3 wrong-information path: agent gold at r-1
                    if ans[i][r - 1] == g:
                        wrong_only = any(x in LABELS and x != g for x in said) and g not in said
                        P3["wrong_only" if wrong_only else "other"].add(q, ans[i][r] != g)
            R = [{a[t] for a in ans} for t in range(3)]
            if g in R[0]:
                conv.add(q, len(R[2]) == 1 and g not in R[2])
                if g not in R[2]:                              # P4 lost-candidate trace
                    t = 2 if g in R[1] else 1
                    holders = [h for h in range(3) if ans[h][t - 1] == g]
                    cls = []
                    for h in holders:
                        srcs = ev[(t, h)]
                        if any(s is None or state(s) is None for s in srcs):
                            cls.append("unjudged"); continue
                        gs = [s for s in srcs if s["src_label"] == g]
                        said = [s["answer_in_summary"] for s in srcs]
                        if gs:
                            c = [state(s) for s in gs]
                            cls.append("conveyed+evidence_but_dropped" if "conveyed+evidence" in c else
                                       "answer_only_but_dropped" if "answer_only" in c else "erased")
                        else:
                            cls.append("wrong_only" if any(x in LABELS for x in said) else "no_label")
                    for sub in ("all", "comp") if p_sc[qi] < 0.5 else ("all",):
                        P4n[f"{sub}|cases"] += 1; P4n[f"{sub}|t{t}"] += 1; P4n[f"{sub}|holders{len(holders)}"] += 1
                        for k in ("erased", "conveyed+evidence_but_dropped", "answer_only_but_dropped", "wrong_only", "no_label", "unjudged"):
                            P4[f"{sub}|{k}"].add(q, cls.count(k) / len(cls))
    res["P1"] = {k: v.ci(rng) for k, v in sorted(P1.items())}
    res["P2"] = {k: v.ci(rng) for k, v in P2.items()} | dict(engagement_when_gold_conveyed=dict(P2eng))
    res["P3"] = {k: v.ci(rng) for k, v in P3.items()}
    res["P4"] = {k: v.ci(rng) for k, v in sorted(P4.items())} | dict(counts=dict(P4n))
    res["incorrect_convergence"] = conv.ci(rng)
    return res


def convergence_bc(rng):
    """Incorrect convergence for Phase VI B and C (no judge needed)."""
    lab = json.load(open(f"{ROOT}/studies/6_medarc_matched_ablation/results/labels.json"))
    out = {}
    for m in "BC":
        c = Ratio()
        for run in (1, 2, 3):
            for rec in jl(f"{ROOT}/studies/6_medarc_matched_ablation/results/main/{m}_run{run}.jsonl"):
                g = lab[str(rec["pubid"])]; ans = AG.controlled(rec)[0]
                if g in {a[0] for a in ans}:
                    R2 = {a[2] for a in ans}
                    c.add(rec["idx"], len(R2) == 1 and g not in R2)
        out[m] = c.ci(rng)
    return out


def exposure_abc(rng, rounds=(1, 2)):
    """Added 10-11 after J1 (no judge): gold holder at r−1; P(leaves gold at r) by what the other two held at r−1
    (both wrong / ≥1 gold), in arms A, B, C (identical round-0). A−B, A−C: paired cluster bootstrap over questions."""
    lab = json.load(open(f"{ROOT}/studies/6_medarc_matched_ablation/results/labels.json"))
    acc = {m: {k: (collections.defaultdict(float), collections.defaultdict(float)) for k in ("others_wrong", "other_gold")} for m in "ABC"}
    for m in "ABC":
        for run in (1, 2, 3):
            for rec in jl(f"{ROOT}/studies/6_medarc_matched_ablation/results/main/{m}_run{run}.jsonl"):
                g = lab[str(rec["pubid"])]; ans = AG.controlled(rec)[0]
                for r in rounds:
                    for i in range(3):
                        if ans[i][r - 1] != g:
                            continue
                        k = "other_gold" if any(ans[o][r - 1] == g for o in range(3) if o != i) else "others_wrong"
                        n, d = acc[m][k]; d[rec["idx"]] += 1; n[rec["idx"]] += ans[i][r] != g
    out = {}
    for k in ("others_wrong", "other_gold"):
        qs = sorted(set().union(*(acc[m][k][1] for m in "ABC")))
        arr = {m: (np.array([acc[m][k][0][q] for q in qs]), np.array([acc[m][k][1][q] for q in qs])) for m in "ABC"}
        idx = rng.integers(0, len(qs), (B, len(qs)))
        rate = lambda m, ix=None: (arr[m][0][ix].sum(-1) / np.maximum(arr[m][1][ix].sum(-1), 1e-12)) if ix is not None else arr[m][0].sum() / arr[m][1].sum()
        o = {m: dict(p=float(rate(m)), lo=float(np.percentile(rate(m, idx), 2.5)), hi=float(np.percentile(rate(m, idx), 97.5)), n=float(arr[m][1].sum())) for m in "ABC"}
        for a, b in (("A", "B"), ("A", "C")):
            bs = rate(a, idx) - rate(b, idx)
            o[f"{a}-{b}"] = dict(p=float(rate(a) - rate(b)), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), n=float(arr[a][1].sum()))
        out[k] = o
    return out


def human(path=None):
    from sklearn.metrics import cohen_kappa_score
    key = json.load(open(f"{OUT}/human_check/key.json"))
    j1 = {(r["set"], r["run"], r["idx"], r["round"], r["agent"], r["j"]): r for r in jl(f"{OUT}/judge_j1.jsonl")}
    rows = list(csv.DictReader(open(path or f"{OUT}/human_check/sheet_labeled.csv", encoding="utf-8-sig")))   # Excel may add a BOM
    res = {}
    for f in ("final_answer_original", "answer_in_summary", "evidence_preserved", "misrepresented"):
        h, m = [], []
        for row, k in zip(rows, key):
            v = row[f].strip().lower()
            if v:
                h.append(v); m.append(str(j1[(k["set"], k["run"], k["idx"], k["round"], k["agent"], k["j"])][f]).lower())
        if h:
            res[f] = dict(n=len(h), agreement=float(np.mean([a == b for a, b in zip(h, m)])), kappa=float(cohen_kappa_score(h, m)))
    json.dump(res, open(f"{OUT}/human_check/AGREEMENT.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


def main():
    if sys.argv[1:2] == ["human"]:
        return human(*sys.argv[2:3])
    rng = np.random.default_rng(SEED)
    j1 = {(r["set"], r["run"], r["idx"], r["round"], r["agent"], r["j"]): r for r in jl(f"{OUT}/judge_j1.jsonl")}
    j2 = {(r["set"], r["run"], r["idx"], r["round"], r["agent"], r["j"]): r for r in jl(f"{OUT}/judge_j2.jsonl")}
    res = {tag: analyse(tag, j1, j2, rng) for tag in ("VI_A", "V")}
    res["incorrect_convergence_VI_BC"] = convergence_bc(rng)
    res["exposure_ABC"] = exposure_abc(rng)
    res["exposure_ABC_r1"] = exposure_abc(rng, (1,))   # r=1 conditions on round-0, identical across arms
    res["judge_control"] = json.load(open(f"{OUT}/judge_control.json")) if os.path.exists(f"{OUT}/judge_control.json") else None
    json.dump(res, open(f"{OUT}/SUMMARY_E2.json", "w"), indent=1)
    f = lambda c: "n/a" if c.get("p") is None else f"{c['p']*100:.1f}% [{c['lo']*100:.1f}, {c['hi']*100:.1f}] (n={c['n']:.0f})"
    L = ["=== E2′ summary-level preservation and uptake (post-hoc, observational; judge Qwen3-32B-FP8, human check pending) ==="]
    for tag in ("VI_A", "V"):
        r = res[tag]
        L += [f"[{tag}] judge↔parser answer agreement {r['sanity_parser_agreement']:.3f}; unjudged {r['judge_unparsed']:.3f}",
              "  P1 preservation (per summarized response):"]
        for grp in ("gold_src", "wrong_src"):
            L.append(f"    {grp}: " + "; ".join(f"{k} {f(r['P1'][f'{grp}|{k}'])}" for k in
                     ("label_mentioned", "conveyed+evidence", "answer_only", "not_conveyed", "evidence_full", "misrepresented")))
            L.append(f"    {grp} (disagreement events): " + "; ".join(f"{k} {f(r['P1'][f'{grp}|disagree|{k}'])}" for k in
                     ("conveyed+evidence", "answer_only", "not_conveyed")))
        L.append("  P2 uptake (agent wrong at r−1, gold among sources) P(agent → gold): " +
                 "; ".join(f"{k} {f(v)}" for k, v in r["P2"].items() if k != "engagement_when_gold_conveyed"))
        L.append(f"     J2 engagement with conveyed gold: {r['P2']['engagement_when_gold_conveyed']}")
        L.append("  P3 (agent gold at r−1) P(leave gold): " + "; ".join(f"{k} {f(v)}" for k, v in r["P3"].items()))
        L.append(f"  incorrect convergence (R2 unanimous wrong | gold ∈ R0): {f(r['incorrect_convergence'])}")
        for sub in ("all", "comp"):
            c = r["P4"]["counts"]
            L.append(f"  P4 lost-candidate trace [{sub}] cases {c.get(f'{sub}|cases', 0)} (drop round t1 {c.get(f'{sub}|t1', 0)}, t2 {c.get(f'{sub}|t2', 0)}; "
                     f"holders 1/2/3: {c.get(f'{sub}|holders1', 0)}/{c.get(f'{sub}|holders2', 0)}/{c.get(f'{sub}|holders3', 0)})")
            L.append("    " + "; ".join(f"{k} {f(r['P4'][f'{sub}|{k}'])}" for k in
                     ("erased", "conveyed+evidence_but_dropped", "answer_only_but_dropped", "wrong_only", "no_label", "unjudged")))
    L.append("incorrect convergence Phase VI B / C: " + " / ".join(f(res["incorrect_convergence_VI_BC"][m]) for m in "BC"))
    for tag_, ex_ in (("r1+r2", res["exposure_ABC"]), ("r1 only (pre-treatment condition)", res["exposure_ABC_r1"])):
      for k, o in ex_.items():
        L.append(f"holder exposure {tag_} [{k} at r−1] P(leave gold): " + "; ".join(f"{m} {f(o[m])}" for m in "ABC") +
                 "; " + "; ".join(f"{d} {o[d]['p']*100:+.1f} pp [{o[d]['lo']*100:+.1f}, {o[d]['hi']*100:+.1f}]" for d in ("A-B", "A-C")))
    jc = res["judge_control"]
    if jc:
        L.append(f"judge control (500 VI_A items): label=source true {jc['true_summary']['label_equals_source']:.3f} vs mismatched "
                 f"{jc['mismatched_summary']['label_equals_source']:.3f}; evidence true {jc['true_summary']['evidence']} vs mismatched "
                 f"{jc['mismatched_summary']['evidence']}; misrepresented {jc['true_summary']['misrepresented']:.3f} vs {jc['mismatched_summary']['misrepresented']:.3f}")
    open(f"{OUT}/SUMMARY_E2.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
