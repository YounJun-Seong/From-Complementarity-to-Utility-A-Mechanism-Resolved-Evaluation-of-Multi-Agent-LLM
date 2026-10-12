"""Post-hoc audit (POSTHOC_TRANSITION.md): question-level G2→G4→G6 transitions on MedARC (Phase V, Phase VI A/B/C)
and the Phase II SC token-accounting sensitivity. No LLM calls.

  python experiments/posthoc_transition.py  ->  studies/7_posthoc_audits/results/TRANSITION.{txt,json}
"""
import glob
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
OUT = f"{ROOT}/studies/7_posthoc_audits/results"


def jl(p):
    return [json.loads(l) for l in open(p)]


def ratio_boot(num, den, rng):
    num, den = np.asarray(num, float), np.asarray(den, float)
    idx = rng.integers(0, len(num), (B, len(num)))
    bs = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1e-12)
    return [float(num.sum() / max(den.sum(), 1e-12)), *map(float, np.percentile(bs, [2.5, 97.5]))]


def diff_boot(n1, d1, n2, d2, rng):
    """Cluster bootstrap of ratio1 − ratio2 (same questions)."""
    a = [np.asarray(x, float) for x in (n1, d1, n2, d2)]
    idx = rng.integers(0, len(a[0]), (B, len(a[0])))
    r = lambda n, d, i=None: (n[i].sum(-1) if i is not None else n.sum()) / np.maximum(d[i].sum(-1) if i is not None else d.sum(), 1e-12)
    bs = r(a[0], a[1], idx) - r(a[2], a[3], idx)
    return [float(r(a[0], a[1]) - r(a[2], a[3])), *map(float, np.percentile(bs, [2.5, 97.5]))]


def counts(runs, gold, hard):
    """Per-question counts over runs. hard[q]: independent baseline fails on q."""
    Q = len(gold)
    keys = ("n", "avail", "comp", "real", "selmiss", "lost", "c_real", "c_selmiss", "c_lost",
            "wrong", "notavail", "gained", "gained_ok", "r2_real", "c_r2_real")
    c = {k: np.zeros(Q) for k in keys}
    for qi in range(Q):
        g = gold[qi]
        for run in runs:
            rec = run[qi]
            ans = AG.controlled(rec)[0]
            r0, r2 = {a[0] for a in ans}, {a[2] for a in ans}
            ok = AG.a_paper_no_reinit(rec)[0] == g
            mv = AG.a_majority(rec, 2) == g
            av, pr = g in r0, g in r2
            assert not ((not pr) and ok), "final answer must be a round-2 answer"
            c["n"][qi] += 1; c["wrong"][qi] += not ok
            if av:
                c["avail"][qi] += 1
                c["real"][qi] += pr and ok; c["selmiss"][qi] += pr and not ok; c["lost"][qi] += not pr
                c["r2_real"][qi] += mv
                if hard[qi]:
                    c["comp"][qi] += 1
                    c["c_real"][qi] += pr and ok; c["c_selmiss"][qi] += pr and not ok; c["c_lost"][qi] += not pr
                    c["c_r2_real"][qi] += mv
            else:
                c["notavail"][qi] += 1; c["gained"][qi] += pr; c["gained_ok"][qi] += pr and ok
    return c


def summarize(c, rng):
    R = lambda n, d: ratio_boot(c[n], c[d], rng)
    wa = c["wrong"].copy()
    fail_avail = c["selmiss"] + c["lost"]
    return dict(
        P_avail=R("avail", "n"), P_comp=R("comp", "n"),
        on_avail=dict(realized=R("real", "avail"), selection_miss=R("selmiss", "avail"), lost=R("lost", "avail"),
                      realized_R2MV=R("r2_real", "avail")),
        on_comp=dict(n_units=int(c["comp"].sum()), n_questions=int((c["comp"] > 0).sum()),
                     realized=R("c_real", "comp"), selection_miss=R("c_selmiss", "comp"), lost=R("c_lost", "comp"),
                     realized_R2MV=R("c_r2_real", "comp")),
        failures=dict(n_wrong=int(wa.sum()), share_gold_in_R0=ratio_boot(fail_avail, wa, rng),
                      share_lost=ratio_boot(c["lost"], wa, rng), share_selection_miss=ratio_boot(c["selmiss"], wa, rng)),
        on_notavail=dict(gained=R("gained", "notavail"), gained_and_correct=R("gained_ok", "notavail")))


def transitions(rng):
    res = {}
    # Phase V
    lab = json.load(open(f"{ROOT}/studies/5_medarc_transfer/results/labels.json"))
    D = f"{ROOT}/studies/5_medarc_transfer/results/main"
    M = [jl(f"{D}/medarc_run{r}.jsonl") for r in (1, 2, 3)]
    gold = np.array([lab[str(x["pubid"])] for x in M[0]])
    assert all([x["pubid"] for x in run] == [x["pubid"] for x in M[0]] for run in M)
    sets = {"PhaseV": dict(gold=gold, arms={"MedARC": M}, D=D, sc=[f"{D}/sc_run{r}.jsonl" for r in (1, 2, 3)],
                           cot=[f"{D}/cot_run{r}.jsonl" for r in (1, 2, 3)], key="pubid", K=25)}
    # Phase VI
    lab = json.load(open(f"{ROOT}/studies/6_medarc_matched_ablation/results/labels.json"))
    D = f"{ROOT}/studies/6_medarc_matched_ablation/results/main"
    arms = {m: [sorted(jl(f"{D}/{m}_run{r}.jsonl"), key=lambda x: x["idx"]) for r in (1, 2, 3)] for m in "ABC"}
    gold6 = np.array([lab[str(x["pubid"])] for x in arms["A"][0]])
    sets["PhaseVI"] = dict(gold=gold6, arms=arms, sc=[f"{D}/sc_run{r}.jsonl" for r in (1, 2, 3)],
                           cot=[f"{D}/cot_run{r}.jsonl" for r in (1, 2, 3)], key="idx", K=25)
    for name, s in sets.items():
        gold = s["gold"]; Q = len(gold)
        srt = (lambda rows: sorted(rows, key=lambda x: x["idx"])) if s["key"] == "idx" else (lambda rows: rows)
        sc = [srt(jl(p)) for p in s["sc"]]
        votes = np.concatenate([np.array([[P.answer(t) or "" for t in x["texts"]] for x in run]) for run in sc], 1)
        p_sc = OA.p_sc(votes, gold, s["K"], np.random.default_rng(SEED))
        cot = [srt(jl(p)) for p in s["cot"]]
        bs = np.array([[P.answer(x["text"]) == g for x, g in zip(run, gold)] for run in cot], float).mean(0)
        hard = {"SC25": p_sc < 0.5, "B_strong": bs < 0.5}
        out = dict(n=Q, P_SC25_wrong=float(hard["SC25"].mean()), P_Bstrong_wrong=float(hard["B_strong"].mean()))
        C = {}
        for m, runs in s["arms"].items():
            C[m] = {h: counts(runs, gold, hard[h]) for h in hard}
            out[m] = {h: summarize(C[m][h], rng) for h in hard}
        if name == "PhaseVI":
            out["contrasts"] = {}
            for a, b in (("A", "B"), ("A", "C")):
                ca, cb = C[a]["SC25"], C[b]["SC25"]
                out["contrasts"][f"{a}-{b}"] = dict(
                    comp_realized=diff_boot(ca["c_real"], ca["comp"], cb["c_real"], cb["comp"], rng),
                    comp_lost=diff_boot(ca["c_lost"], ca["comp"], cb["c_lost"], cb["comp"], rng),
                    avail_lost=diff_boot(ca["lost"], ca["avail"], cb["lost"], cb["avail"], rng))
        res[name] = out
    return res


def sc_sensitivity():
    """Phase II: generation prompt charged once per n-sample request; alternative = per sample."""
    base = f"{ROOT}/studies/2_complementarity_router/results"
    res = {}
    # PILOT
    d = f"{base}/pilot/Qwen2.5-14B-Instruct"
    S = [jl(f"{d}/S/run{r}.jsonl") for r in (1, 2, 3)]
    SC = [jl(f"{d}/SC/run{r}.jsonl") for r in range(1, 7)]
    cost = [json.loads(l) for f in glob.glob(f"{d}/M1/run*/shard*/cost.jsonl") for l in open(f)]
    sp = np.mean([r["S_prompt_tokens"] for run in S for r in run])
    scp = np.mean([r["prompt_tokens"] for run in SC[:3] for r in run])
    n = 20
    gen_p = (n * sp - scp) / (n - 1)                     # S = gen + e3 ; SC = gen + n*e3
    per = np.mean([(r["prompt_tokens"] + r["completion_tokens"]) / n for run in SC[:3] for r in run])
    tM = np.mean([c["prompt_tokens"] + c["completion_tokens"] for c in cost])
    per_alt = per + gen_p * (n - 1) / n
    k_star, k_alt = OA.k_star(tM, per, 60), OA.k_star(tM, per_alt, 60)
    gold = np.array([r["gold"] for r in S[0]])
    M = []
    for r in (1, 2, 3):
        by = {}
        for sd in glob.glob(f"{d}/M1/run{r}/shard*"):
            s = int(os.path.basename(sd)[5:])
            for k, l in enumerate(open(f"{sd}/chatgpt-syn_verif")):
                by[s * 20 + k] = json.loads(l)
        M.append(np.array([by[i]["pred_answer"] == by[i]["gold_answer"] for i in range(len(gold))], float))
    accM = float(np.mean(M))
    pool = np.concatenate([np.array([r["votes"] for r in run]) for run in SC[:3]], 1)
    curve = {k: float(OA.p_sc(pool, gold, k, np.random.default_rng(SEED)).mean()) for k in range(k_alt, k_star + 1)}
    res["PILOT"] = dict(gen_prompt_est=float(gen_p), per_sample=float(per), per_sample_alt=float(per_alt), M1_tokens=float(tM),
                        k_star=k_star, k_star_alt=k_alt, M1_acc=accM, SC_acc=curve,
                        SC_tokens_alt={k: float(k * per_alt) for k in curve})
    # HELD-OUT (one SC run of 22)
    d = f"{base}/heldout/Qwen2.5-14B-Instruct"
    SCh = jl(f"{d}/SC/run1.jsonl")
    goldh = np.array([r["gold"] for r in SCh])
    v = np.array([r["votes"] for r in SCh])
    Mh = json.load(open(f"{base}/heldout/RESULTS.json"))
    def first_k(k):
        out = []
        for row, g in zip(v[:, :k], goldh):
            labs = [x for x in row if x]
            if not labs:
                out.append(0.0); continue
            cnt = {a: labs.count(a) for a in labs}; top = max(cnt.values())
            out.append(float(next(a for a in labs if cnt[a] == top) == g))
        return float(np.mean(out))
    res["HELDOUT"] = dict(k_star=22, k_star_alt=k_alt, SC_first_k={k: first_k(k) for k in range(k_alt, 23)},
                          SC_subset_avg={k: float(OA.p_sc(v, goldh, k, np.random.default_rng(SEED)).mean()) for k in range(k_alt, 23)},
                          table_rows=[r for r in Mh.get("table", []) if isinstance(r, (list, dict))][:4])
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    res = dict(transitions=transitions(rng), sc_sensitivity=sc_sensitivity())
    json.dump(res, open(f"{OUT}/TRANSITION.json", "w"), indent=1)
    f = lambda v: f"{v[0]*100:.1f}% [{v[1]*100:.1f}, {v[2]*100:.1f}]"
    g = lambda v: f"{v[0]*100:+.1f} pp [{v[1]*100:+.1f}, {v[2]*100:+.1f}]"
    L = ["=== Post-hoc G2→G4→G6 transitions (descriptive; lost ⇒ wrong by construction) ==="]
    for name, out in res["transitions"].items():
        L.append(f"[{name}] n={out['n']}  P(SC25 wrong) {out['P_SC25_wrong']:.3f}  P(B_strong wrong) {out['P_Bstrong_wrong']:.3f}")
        for m, byh in out.items():
            if not isinstance(byh, dict) or "SC25" not in byh:
                continue
            for h, s in byh.items():
                a, c, fl, na = s["on_avail"], s["on_comp"], s["failures"], s["on_notavail"]
                L += [f"  {m} | comp vs {h}: P(avail) {f(s['P_avail'])}  P(comp) {f(s['P_comp'])} "
                      f"({c['n_units']} q×run, {c['n_questions']} q)",
                      f"    avail: realized {f(a['realized'])}  sel-miss {f(a['selection_miss'])}  lost {f(a['lost'])}  (R2MV realized {a['realized_R2MV'][0]*100:.1f}%)",
                      f"    comp : realized {f(c['realized'])}  sel-miss {f(c['selection_miss'])}  lost {f(c['lost'])}  (R2MV realized {c['realized_R2MV'][0]*100:.1f}%)",
                      f"    failures (n={fl['n_wrong']}): gold in R0 {f(fl['share_gold_in_R0'])} = lost {f(fl['share_lost'])} + sel-miss {f(fl['share_selection_miss'])}",
                      f"    not avail: gained {f(na['gained'])}, gained∧correct {f(na['gained_and_correct'])}"]
        for k, v in out.get("contrasts", {}).items():
            L.append(f"  contrast {k} (comp vs SC25): realized {g(v['comp_realized'])}  lost {g(v['comp_lost'])}; avail lost {g(v['avail_lost'])}")
    s = res["sc_sensitivity"]
    p, h = s["PILOT"], s["HELDOUT"]
    L += ["", "=== Phase II SC token-accounting sensitivity (Qwen2.5-14B) ===",
          f"PILOT: gen prompt ≈ {p['gen_prompt_est']:.0f} tok; per-sample {p['per_sample']:.0f} (as frozen) vs {p['per_sample_alt']:.0f} (gen prompt per sample); "
          f"M1 {p['M1_tokens']:.0f} tok -> k* {p['k_star']} vs k*′ {p['k_star_alt']}",
          f"  M1 acc {p['M1_acc']:.3f}; SC-k acc " + ", ".join(f"k={k}: {v:.3f}" for k, v in p["SC_acc"].items()),
          "HELD-OUT SC-k (first k / subset avg): " + ", ".join(f"k={k}: {h['SC_first_k'][k]:.3f}/{h['SC_subset_avg'][k]:.3f}" for k in h["SC_first_k"])]
    open(f"{OUT}/TRANSITION.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
