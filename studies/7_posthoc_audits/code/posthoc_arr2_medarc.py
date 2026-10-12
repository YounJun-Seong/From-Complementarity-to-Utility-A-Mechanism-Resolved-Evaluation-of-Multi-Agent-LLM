"""Post-hoc ARR2-E4 (POSTHOC_ARR2.md; definitions committed before this script was run): MedARC failure trace.
Phase V (n=498) and Phase VI arms A/B/C (n=472), 3 runs each. No LLM calls.

  T1 agent-level G/W trajectories over rounds 0→1→2      T2 #gold holders R0 × R2 (q×run), P(final correct)
  T3 lone-correct fate by peer configuration and gold     T4 gate mapping (G2, G4, G5, G6)
  Case studies chosen at random (seed 0) among eligible units -> E4_CASES.md

  python studies/7_posthoc_audits/code/posthoc_arr2_medarc.py  ->  studies/7_posthoc_audits/results/arr2/E4.{txt,json}, E4_CASES.md
"""
import collections
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..", "..")
sys.path.insert(0, f"{HERE}/../../5_medarc_transfer/code")
import aggregate as AG          # noqa: E402

B, SEED = 10000, 0
OUT = f"{ROOT}/studies/7_posthoc_audits/results/arr2"
PATTERNS = ["".join(p) for p in __import__("itertools").product("GW", repeat=3)]


def jl(p):
    return [json.loads(l) for l in open(p)]


def load(arm):
    if arm == "V":
        lab = json.load(open(f"{ROOT}/studies/5_medarc_transfer/results/labels.json"))
        runs = [jl(f"{ROOT}/studies/5_medarc_transfer/results/main/medarc_run{r}.jsonl") for r in (1, 2, 3)]
        inp = {x["pubid"]: x for x in jl(f"{ROOT}/studies/5_medarc_transfer/results/data/eval_inputs.jsonl")}
    else:
        lab = json.load(open(f"{ROOT}/studies/6_medarc_matched_ablation/results/labels.json"))
        runs = [sorted(jl(f"{ROOT}/studies/6_medarc_matched_ablation/results/main/{arm}_run{r}.jsonl"), key=lambda z: z["idx"]) for r in (1, 2, 3)]
        inp = {x["pubid"]: x for x in jl(f"{ROOT}/studies/6_medarc_matched_ablation/results/data/inputs.jsonl")}
    for run in runs:
        assert all(x["error"] is None for x in run) and [x["pubid"] for x in run] == [x["pubid"] for x in runs[0]]
    gold = [lab[str(x["pubid"])] for x in runs[0]]
    units = []                                   # one per (q, run)
    for r, run in enumerate(runs):
        for qi, rec in enumerate(run):
            ans, conf = AG.controlled(rec)
            units.append(dict(q=qi, run=r + 1, gold=gold[qi], ans=ans, conf=conf, final=AG.a_paper_no_reinit(rec)[0],
                              r2mv=AG.a_majority(rec, 2), rec=rec, question=inp[rec["pubid"]]["question"]))
    return units, len(gold)


def cluster_ratio(num, den, Q, rng):
    """num, den: per-question sums (length Q). Ratio with cluster-bootstrap CI."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    if den.sum() == 0:
        return [None, None, None, 0]
    idx = rng.integers(0, Q, (B, Q))
    bs = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1e-12)
    return [float(num.sum() / den.sum()), *map(float, np.percentile(bs, [2.5, 97.5])), int(den.sum())]


def per_q(units, Q, pred, cond=lambda u: True):
    n, d = np.zeros(Q), np.zeros(Q)
    for u in units:
        if cond(u):
            d[u["q"]] += 1; n[u["q"]] += bool(pred(u))
    return n, d


def lone(u):
    h = [i for i in range(3) if u["ans"][i][0] == u["gold"]]
    return h[0] if len(h) == 1 else None


def peer_cfg(u, h):
    peers = [u["ans"][i][0] for i in range(3) if i != h]
    return "unanimous_wrong" if peers[0] is not None and peers[0] == peers[1] else "split_wrong"


def fate(u):
    g = u["gold"]
    if u["final"] == g:
        return "realized"
    return "selection_miss" if any(u["ans"][i][2] == g for i in range(3)) else "lost"


def analyse(arm, rng):
    units, Q = load(arm)
    G = lambda u, i, r: u["ans"][i][r] == u["gold"]
    res = {}
    # T1
    pat = collections.Counter("".join("G" if G(u, i, r) else "W" for r in range(3)) for u in units for i in range(3))
    tot = sum(pat.values())
    res["T1_patterns"] = {p: pat[p] / tot for p in PATTERNS}
    for a, b in ((0, 1), (1, 2)):
        res[f"T1_{a}{b}"] = dict(
            abandon=cluster_ratio(*_agent(units, Q, lambda u, i: G(u, i, a), lambda u, i: not G(u, i, b)), Q, rng),
            adopt=cluster_ratio(*_agent(units, Q, lambda u, i: not G(u, i, a), lambda u, i: G(u, i, b)), Q, rng))
    # T2
    m = np.zeros((4, 4)); ok = np.zeros((4, 4))
    for u in units:
        h0 = sum(G(u, i, 0) for i in range(3)); h2 = sum(G(u, i, 2) for i in range(3))
        m[h0, h2] += 1; ok[h0, h2] += u["final"] == u["gold"]
    res["T2"] = dict(share=(m / m.sum()).tolist(), n=m.tolist(), p_final_correct=np.where(m > 0, ok / np.maximum(m, 1), np.nan).tolist())
    # T3 lone-correct
    L = [u for u in units if lone(u) is not None]
    t3 = {}
    for cfg in ("all", "unanimous_wrong", "split_wrong"):
        sel = lambda u: lone(u) is not None and (cfg == "all" or peer_cfg(u, lone(u)) == cfg)
        row = dict(n_units=sum(map(sel, units)),
                   abandon_r1=cluster_ratio(*per_q(units, Q, lambda u: not G(u, lone(u), 1), sel), Q, rng),
                   holder_gold_r2=cluster_ratio(*per_q(units, Q, lambda u: G(u, lone(u), 2), sel), Q, rng))
        for f in ("realized", "selection_miss", "lost"):
            row[f] = cluster_ratio(*per_q(units, Q, lambda u, f=f: fate(u) == f, sel), Q, rng)
        t3[cfg] = row
    for g in ("yes", "no", "maybe"):
        sel = lambda u, g=g: lone(u) is not None and u["gold"] == g
        t3[f"gold={g}"] = dict(n_units=sum(map(sel, units)),
                               abandon_r1=cluster_ratio(*per_q(units, Q, lambda u: not G(u, lone(u), 1), sel), Q, rng),
                               lost=cluster_ratio(*per_q(units, Q, lambda u: fate(u) == "lost", sel), Q, rng),
                               realized=cluster_ratio(*per_q(units, Q, lambda u: fate(u) == "realized", sel), Q, rng))
    res["T3"] = t3
    res["T3_share_lone_of_avail"] = len(L) / max(1, sum(any(G(u, i, 0) for i in range(3)) for u in units))
    # T4
    av = lambda u: any(G(u, i, 0) for i in range(3))
    pr = lambda u: any(G(u, i, 2) for i in range(3))
    ad = _agent2(units, Q, lambda u, i, r: not G(u, i, r - 1), lambda u, i, r: G(u, i, r))
    ab = _agent2(units, Q, lambda u, i, r: G(u, i, r - 1), lambda u, i, r: not G(u, i, r))
    res["T4"] = dict(G2_avail=cluster_ratio(*per_q(units, Q, av), Q, rng),
                     G4_preserved_given_avail=cluster_ratio(*per_q(units, Q, pr, av), Q, rng),
                     G5_adopt_WtoG=cluster_ratio(*ad, Q, rng), G5_abandon_GtoW=cluster_ratio(*ab, Q, rng),
                     G6_selection_miss_given_preserved=cluster_ratio(*per_q(units, Q, lambda u: u["final"] != u["gold"], pr), Q, rng),
                     final_acc=cluster_ratio(*per_q(units, Q, lambda u: u["final"] == u["gold"]), Q, rng),
                     final_acc_R2MV=cluster_ratio(*per_q(units, Q, lambda u: u["r2mv"] == u["gold"]), Q, rng))
    return res, units, Q


def _agent(units, Q, cond, ev):
    n, d = np.zeros(Q), np.zeros(Q)
    for u in units:
        for i in range(3):
            if cond(u, i):
                d[u["q"]] += 1; n[u["q"]] += bool(ev(u, i))
    return n, d


def _agent2(units, Q, cond, ev):
    n, d = np.zeros(Q), np.zeros(Q)
    for u in units:
        for i in range(3):
            for r in (1, 2):
                if cond(u, i, r):
                    d[u["q"]] += 1; n[u["q"]] += bool(ev(u, i, r))
    return n, d


def contrast_abandon(U, Qn, rng):
    """A − B, A − C on P(lone holder abandons gold at r = 1); identical R0 across Phase VI arms (asserted)."""
    out = {}
    for cfg in ("all", "unanimous_wrong", "split_wrong"):
        sums = {}
        for m in "ABC":
            n, d = np.zeros(Qn), np.zeros(Qn)
            for u in U[m]:
                h = lone(u)
                if h is not None and (cfg == "all" or peer_cfg(u, h) == cfg):
                    d[u["q"]] += 1; n[u["q"]] += u["ans"][h][1] != u["gold"]
            sums[m] = (n, d)
        idx = rng.integers(0, Qn, (B, Qn))
        rate = lambda m, ix=None: (sums[m][0][ix].sum(-1) / np.maximum(sums[m][1][ix].sum(-1), 1e-12)) if ix is not None else sums[m][0].sum() / sums[m][1].sum()
        for a, b in (("A", "B"), ("A", "C")):
            bs = rate(a, idx) - rate(b, idx)
            out[f"{cfg}:{a}-{b}"] = [float(rate(a) - rate(b)), *map(float, np.percentile(bs, [2.5, 97.5]))]
    return out


# ---------------------------------------------------------------- case studies
def trunc(s, n):
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n] + " …"


def summary_for(rec, r, i):
    for s in rec.get("summaries", []):
        if s["round"] == r and s["agent"] == i:
            return s["text"]
    return None


def turn(rec, i, r):
    return [m["content"] for m in rec["agent_contexts"][i] if m["role"] == "assistant"][r]


def case_md(u, title, h=None, contrast=None):
    L = [f"### {title}", f"- Question (q={u['q']}, run {u['run']}): {trunc(u['question'], 300)}", f"- Gold: **{u['gold']}**"]
    L.append("- Answers (confidence) by round:")
    for i in range(3):
        L.append(f"  - agent {i}{' (lone holder)' if i == h else ''}: " + " → ".join(
            f"R{r} {u['ans'][i][r]} ({u['conf'][i][r] if u['conf'][i][r] is not None else '?'})" for r in range(3)))
    L.append(f"- Final (A_paper_no_reinit): **{u['final']}** ({'correct' if u['final'] == u['gold'] else 'wrong'}); fate: {fate(u)}; R2 majority {u['r2mv']}")
    if h is not None:
        s = summary_for(u["rec"], 1, h)
        if s:
            L.append(f"- Summary read by agent {h} at round 1 (truncated): > {trunc(s, 700)}")
        L.append(f"- Agent {h} round-1 turn (truncated): > {trunc(turn(u['rec'], h, 1), 500)}")
    if contrast is not None:
        L.append(f"- Same (q, run) in arm B (no summary): agent {h} " + " → ".join(f"R{r} {contrast['ans'][h][r]}" for r in range(3))
                 + f"; final {contrast['final']} ({fate(contrast)})")
    return L + [""]


def cases(U):
    rng = np.random.default_rng(SEED)
    key = lambda u: (u["q"], u["run"])
    Bk = {key(u): u for u in U["B"]}
    pick = lambda xs, k=2: [xs[i] for i in sorted(rng.choice(len(xs), min(k, len(xs)), replace=False))] if xs else []
    A = U["A"]
    a = [u for u in A if lone(u) is not None and peer_cfg(u, lone(u)) == "unanimous_wrong" and fate(u) == "lost"
         and Bk[key(u)]["ans"][lone(u)][2] == u["gold"]]
    b = [u for u in A if lone(u) is not None and fate(u) == "realized"]
    c = [u for u in A if fate(u) == "selection_miss"]
    d = [u for u in U["V"] if lone(u) is not None and fate(u) == "lost"]
    L = ["# ARR2-E4 case studies (random among eligible units, seed 0; not hand-picked)", "",
         f"Eligible counts: (a) {len(a)}, (b) {len(b)}, (c) {len(c)}, (d) {len(d)}", "",
         "## (a) Phase VI A: lone-correct holder, unanimous wrong peers, gold lost; same unit in arm B keeps gold", ""]
    for u in pick(a):
        L += case_md(u, "Case a", lone(u), Bk[key(u)])
    L += ["## (b) Phase VI A: lone-correct at R0, final correct (success)", ""]
    for u in pick(b):
        L += case_md(u, "Case b", lone(u))
    L += ["## (c) Phase VI A: gold in R2 but final wrong (selection miss)", ""]
    for u in pick(c):
        L += case_md(u, "Case c")
    L += ["## (d) Phase V MedARC: lone-correct, gold lost", ""]
    for u in pick(d):
        L += case_md(u, "Case d", lone(u))
    return L, dict(a=len(a), b=len(b), c=len(c), d=len(d))


def fmt(v):
    return "n/a" if v[0] is None else f"{v[0]*100:.1f}% [{v[1]*100:.1f}, {v[2]*100:.1f}] (n={v[3]})"


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    res, U, Qs = {}, {}, {}
    for arm in ("V", "A", "B", "C"):
        res[arm], U[arm], Qs[arm] = analyse(arm, rng)
    for m in "BC":                                  # identical round-0 answers across Phase VI arms
        assert all(x["ans"][i][0] == y["ans"][i][0] for x, y in zip(U["A"], U[m]) for i in range(3))
    res["contrasts_abandon_r1"] = contrast_abandon(U, Qs["A"], rng)
    name = dict(V="Phase V MedARC", A="Phase VI A (interactive)", B="Phase VI B (no summary)", C="Phase VI C (placebo summary)")
    L = ["=== ARR2-E4 MedARC failure trace (controlled parser; final = A_paper_no_reinit) ==="]
    L += ["", "T1 agent-level trajectory patterns R0R1R2 (share of agent-runs)",
          f"    {'arm':30s} " + " ".join(f"{p:>6}" for p in PATTERNS)]
    for arm in U:
        L.append(f"    {name[arm]:30s} " + " ".join(f"{res[arm]['T1_patterns'][p]*100:>5.1f}%" for p in PATTERNS))
    L += ["", "T1 transitions: abandon P(G→W), adopt P(W→G)"]
    for arm in U:
        for s in ("01", "12"):
            L.append(f"    {name[arm]:30s} {s[0]}→{s[1]}: abandon {fmt(res[arm][f'T1_{s}']['abandon'])}; adopt {fmt(res[arm][f'T1_{s}']['adopt'])}")
    L += ["", "T2 #gold holders R0 (rows) × R2 (cols): share of q×run units / P(final correct)"]
    for arm in U:
        L.append(f"    {name[arm]}")
        sh, pc = np.array(res[arm]["T2"]["share"]), np.array(res[arm]["T2"]["p_final_correct"])
        for h0 in range(4):
            L.append(f"      R0={h0}: " + "  ".join(f"R2={h2}: {sh[h0,h2]*100:5.1f}%/" + ("  -  " if np.isnan(pc[h0,h2]) else f"{pc[h0,h2]:.2f}") for h2 in range(4)))
    L += ["", "T2 counts (q×run units): R0 holders → units, lost (R2 holders = 0)"]
    for arm in U:
        n = np.array(res[arm]["T2"]["n"])
        L.append(f"    {name[arm]:30s} " + "  ".join(f"R0={h}: {n[h].sum():.0f} units, {n[h, 0]:.0f} lost" for h in (1, 2, 3))
                 + f"  | avail {n[1:].sum():.0f}, lost {n[1:, 0].sum():.0f} ({n[1:, 0].sum() / n[1:].sum() * 100:.1f}%)")
    lone_units = [u for u in U["A"] if lone(u) is not None]          # R0 identical across Phase VI arms
    per_run = [sum(u["run"] == r for u in lone_units) for r in (1, 2, 3)]
    res["lone_correct_PhaseVI"] = dict(units=len(lone_units), unique_questions=len({u["q"] for u in lone_units}), per_run=per_run)
    L.append(f"    Phase VI lone-correct units: {len(lone_units)} from {len({u['q'] for u in lone_units})} unique questions; "
             f"per run {per_run}")
    L += ["", "T3 lone-correct units (exactly one gold holder at R0)"]
    for arm in U:
        L.append(f"    {name[arm]}: lone-correct = {res[arm]['T3_share_lone_of_avail']*100:.1f}% of avail units")
        for cfg, row in res[arm]["T3"].items():
            extra = "" if cfg.startswith("gold") else f"; holder gold at R2 {fmt(row['holder_gold_r2'])}; selection miss {fmt(row['selection_miss'])}"
            L.append(f"      {cfg:16s} n={row['n_units']:4d}  abandon@r1 {fmt(row['abandon_r1'])}; realized {fmt(row['realized'])}; lost {fmt(row['lost'])}{extra}")
    L += ["", "T3 contrasts, P(lone holder abandons gold at r=1) (same R0 in all Phase VI arms; cluster bootstrap)"]
    for k, v in res["contrasts_abandon_r1"].items():
        L.append(f"    {k:22s} {v[0]*100:+.1f} pp [{v[1]*100:+.1f}, {v[2]*100:+.1f}]")
    L += ["", "T4 gate mapping"]
    for arm in U:
        t = res[arm]["T4"]
        L.append(f"    {name[arm]:30s} G2 avail {fmt(t['G2_avail'])} | G4 preserved|avail {fmt(t['G4_preserved_given_avail'])} | "
                 f"G5 adopt W→G {fmt(t['G5_adopt_WtoG'])}, abandon G→W {fmt(t['G5_abandon_GtoW'])} | "
                 f"G6 sel. miss|preserved {fmt(t['G6_selection_miss_given_preserved'])} | final {fmt(t['final_acc'])}")
    CL, nel = cases(U)
    res["case_eligible"] = nel
    open(f"{OUT}/E4.txt", "w").write("\n".join(L) + "\n")
    open(f"{OUT}/E4_CASES.md", "w").write("\n".join(CL) + "\n")
    json.dump(res, open(f"{OUT}/E4.json", "w"), indent=1, default=float)
    print("\n".join(L)); print(f"cases eligible: {nel}")


if __name__ == "__main__":
    main()
