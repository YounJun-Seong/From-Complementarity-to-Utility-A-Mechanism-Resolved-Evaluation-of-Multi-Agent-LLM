"""Phase II empirical-oracle analysis (PHASE2_FREEZE.md §3.4–3.5). No LLM calls.

The analysis takes per-question arrays from three sources:
  S   (runs x Q)      correctness of the strong single agent
  SC  pools           (Q x n) sampled answers (T=0.7) and the gold letter
  M   (runs x Q)      correctness of M1
plus mean tokens per question for each action and the pre-decision features.

  analyze(...)        -> dict with H, H_plc, G1–G4, secondary AUCs, type table, decision
  python studies/2_complementarity_router/code/oracle_analysis.py selftest   synthetic checks with known answers
"""
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

B_BOOT, SEED = 10000, 0
DRAWS = 2000
LAMBDAS = (0, 0.5, 1, 2, 5)          # pp per 10k tokens (secondary diagnostic)
ACTIONS = ("S", "SC", "M")           # cost order, used for tie-breaking (cheaper wins)


# ---------------------------------------------------------------- estimators
def p_sc(votes, gold, k, rng):
    """P(majority of k draws from the pool is correct) per question; ties -> earliest drawn sample.
    Draws are without replacement when the pool has >= k samples, with replacement otherwise."""
    votes = np.asarray(votes)
    Q, n = votes.shape
    out = np.empty(Q)
    for i in range(Q):
        if n >= k:
            idx = np.argsort(rng.random((DRAWS, n)), axis=1)[:, :k]
        else:
            idx = rng.integers(0, n, (DRAWS, k))
        drawn = votes[i][idx]                                  # DRAWS x k
        labs, codes = np.unique(drawn, return_inverse=True)
        codes = codes.reshape(drawn.shape)
        onehot = codes[..., None] == np.arange(len(labs))      # DRAWS x k x L
        cnt = onehot.sum(1)
        first = np.where(onehot.any(1), onehot.argmax(1), k)   # earliest position of each label
        best = cnt == cnt.max(1, keepdims=True)
        win = np.argmin(np.where(best, first, k + 1), axis=1)
        out[i] = np.mean(labs[win] == gold[i])
    return out


def k_star(m_tokens, sc_tokens_per_sample, kmax):
    """Integer k whose mean SC-k tokens/question (k x per-sample tokens) is closest to M1's."""
    ks = np.arange(1, kmax + 1)
    return int(ks[np.argmin(np.abs(ks * sc_tokens_per_sample - m_tokens))])


def types(pS, pSC, pM, t=0.5):
    """Partition A–E of §3.4 (selection-half estimates)."""
    T = np.full(len(pS), "E", dtype="<U1")
    good_m = pM >= t
    T[good_m & (pS >= t)] = "A"
    T[good_m & (pS < t) & (pSC >= t)] = "B"
    T[good_m & (pS < t) & (pSC < t)] = "C"
    T[~good_m & (np.maximum(pS, pSC) >= t)] = "D"
    return T


def pick(est):
    """Per-question argmax over the columns of est (cost order); ties -> cheaper (lower column)."""
    est = np.asarray(est)
    return np.argmax(est >= est.max(axis=1, keepdims=True) - 1e-12, axis=1)


def boot_ci(d, rng):
    d = np.asarray(d, float)
    n = len(d)
    bs = d[rng.integers(0, n, (B_BOOT, n))].mean(axis=1)
    return float(d.mean()), *map(float, np.percentile(bs, [2.5, 97.5]))


def oof_auc(X, y, rng_seed=SEED):
    """5-fold out-of-fold AUC of a standardized logistic regression; CI by bootstrap over questions."""
    y = np.asarray(y, int)
    if y.min() == y.max() or min(np.bincount(y)) < 5:
        return dict(auc=float("nan"), lo=float("nan"), hi=float("nan"), n_pos=int(y.sum()))
    oof = np.empty(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=rng_seed).split(X, y):
        mdl = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        oof[te] = mdl.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    rng = np.random.default_rng(rng_seed)
    bs = []
    for _ in range(B_BOOT // 10):          # 1,000 resamples are enough for an AUC interval
        b = rng.integers(0, len(y), len(y))
        if y[b].min() != y[b].max():
            bs.append(roc_auc_score(y[b], oof[b]))
    return dict(auc=float(roc_auc_score(y, oof)), lo=float(np.percentile(bs, 2.5)),
                hi=float(np.percentile(bs, 97.5)), n_pos=int(y.sum()))


# ---------------------------------------------------------------- main analysis
def analyze(S, sc_votes, gold, M, tok, X, k=None, placebo_votes=None, sel_runs=(0, 1), eval_run=2,
            pool_split=40, seed=SEED):
    """S, M: (runs x Q) 0/1. sc_votes: (Q x n) answers, columns ordered run1, run2, ... (pool A = first
    `pool_split` columns, pool E = the rest). placebo_votes: (Q x n_P) or None (DEV). tok: dict of mean
    tokens/question for 'S', 'M', and per-sample 'SC'. X: (Q x f) pre-decision features."""
    rng = np.random.default_rng(seed)
    S, M, gold = np.asarray(S, float), np.asarray(M, float), np.asarray(gold)
    sc_votes = np.asarray(sc_votes)
    k = k or k_star(tok["M"], tok["SC"], kmax=sc_votes.shape[1])
    sel = list(sel_runs)

    pS_sel, pM_sel = S[sel].mean(0), M[sel].mean(0)
    pSC_sel = p_sc(sc_votes[:, :pool_split], gold, k, rng)
    eS, eM = S[eval_run], M[eval_run]
    eSC = p_sc(sc_votes[:, pool_split:], gold, k, rng)

    # cross-fitted empirical oracle: choose on the selection half, score on the evaluation half
    a_cs = pick(np.c_[pS_sel, pSC_sel, pM_sel])
    a_s = pick(np.c_[pS_sel, pSC_sel])
    ev = np.c_[eS, eSC, eM]
    h_i = ev[np.arange(len(a_cs)), a_cs] - ev[np.arange(len(a_s)), a_s]
    cost = np.array([tok["S"], tok["S"] + k * tok["SC"], tok["S"] + tok["M"]])
    res = dict(k_star=k, Q=len(gold),
               acc_oracle_CS=float(ev[np.arange(len(a_cs)), a_cs].mean()),
               acc_oracle_S=float(ev[np.arange(len(a_s)), a_s].mean()),
               tok_oracle_CS=float(cost[a_cs].mean()), tok_oracle_S=float(cost[a_s].mean()),
               share_M_chosen=float((a_cs == 2).mean()))
    res["H"] = boot_ci(h_i, rng)

    if placebo_votes is not None:
        P = np.asarray(placebo_votes)
        half = P.shape[1] // 2
        pP_sel, eP = p_sc(P[:, :half], gold, k, rng), p_sc(P[:, half:], gold, k, rng)
        a_cp = pick(np.c_[pS_sel, pSC_sel, pP_sel])
        evp = np.c_[eS, eSC, eP]
        hp_i = evp[np.arange(len(a_cp)), a_cp] - ev[np.arange(len(a_s)), a_s]
        res["H_plc"] = boot_ci(hp_i, rng)
        res["H_minus_H_plc"] = boot_ci(h_i - hp_i, rng)
    else:
        res["H_plc"] = res["H_minus_H_plc"] = None

    T = types(pS_sel, pSC_sel, pM_sel)
    res["types"] = {t: float((T == t).mean()) for t in "ABCDE"}
    c = T == "C"
    res["n_C"] = int(c.sum())
    res["G3_contrast"] = boot_ci(eM[c] - eSC[c], rng) if c.sum() >= 2 else None

    y = (pM_sel - np.maximum(pS_sel, pSC_sel) > 0).astype(int)
    res["G4"] = oof_auc(X, y, seed)
    t_cheap = np.where(pS_sel >= pSC_sel, cost[0], cost[1])
    gain = pM_sel - np.maximum(pS_sel, pSC_sel)
    res["G4_utility"] = {lam: oof_auc(X, (gain - lam / 100 * (cost[2] - t_cheap) / 1e4 > 0).astype(int), seed)
                         for lam in LAMBDAS}
    res.update(decide(res))
    return res


def decide(r):
    H, d, g3, g4 = r["H"], r["H_minus_H_plc"], r["G3_contrast"], r["G4"]
    G1 = H[0] >= 0.020 and H[1] > 0
    G2 = d is not None and d[0] >= 0.015 and d[1] > 0
    G3 = r["n_C"] >= 15 and g3 is not None and g3[1] > 0
    G4 = g4["auc"] >= 0.65 and g4["lo"] >= 0.55
    if not (G1 and G2):
        verdict = "NO-GO"
    elif G3 and G4:
        verdict = "GO"
    else:
        verdict = "INCONCLUSIVE"
    return dict(pass_G1=G1, pass_G2=G2, pass_G3=G3, pass_G4=G4, decision=verdict)


# ---------------------------------------------------------------- synthetic self-test
def _simulate(Q, scenario, rng):
    """Synthetic data with a known answer. Letters A–E, gold 'A'."""
    gold = np.array(["A"] * Q)
    diff = rng.random(Q)                                      # latent difficulty
    x_sig = rng.normal(size=Q)                                # observable feature
    p_samp = np.clip(0.9 - 0.8 * diff, 0.05, 0.95)            # per-sample correctness
    def draw(p, n):
        ok = rng.random((Q, n)) < p[:, None]
        wrong = rng.choice(list("BCDE"), (Q, n))
        return np.where(ok, "A", wrong)
    S = (rng.random((3, Q)) < p_samp).astype(float)
    sc = draw(p_samp, 60)
    if scenario == "M_is_sampler":          # collaboration = another sampler at equal compute
        Mv = draw(p_samp, 3 * 20)
        M = np.stack([(np.array([np.unique(r, return_counts=True)[0][np.argmax(np.unique(r, return_counts=True)[1])]
                                 for r in Mv[:, 20 * j:20 * (j + 1)]]) == "A") for j in range(3)]).astype(float)
    elif scenario == "planted_C":           # collaboration rescues hard questions flagged by x_sig
        rescue = (diff > 0.6) & (x_sig > 0.3)
        pm = np.where(rescue, 0.97, p_samp)
        M = (rng.random((3, Q)) < pm).astype(float)
    elif scenario == "M_equals_S":
        M = S.copy()
    X = np.c_[x_sig, rng.normal(size=(Q, 3))]
    P = draw(p_samp, 60)
    tok = dict(S=1400, SC=1150, M=22000)
    return dict(S=S, sc_votes=sc, gold=gold, M=M, tok=tok, X=X, placebo_votes=P)


def selftest():
    rng = np.random.default_rng(123)
    expect = {"M_is_sampler": "NO-GO", "planted_C": "GO", "M_equals_S": "NO-GO"}
    ok = True
    for sc, want in expect.items():
        r = analyze(**_simulate(600, sc, rng))
        got = r["decision"]
        print(f"{sc:14s} k*={r['k_star']:2d}  H {r['H'][0]:+.3f}  H−Hplc {r['H_minus_H_plc'][0]:+.3f}  "
              f"n_C {r['n_C']:3d}  AUC {r['G4']['auc']:.2f}  -> {got} (expected {want})")
        ok &= got == want
    # unit checks
    assert types(np.array([1, 0, 0, 0, 1, 0.]), np.array([0, 1, 0, 1, 0, 0.]), np.array([1, 1, 1, 0, 0, 0.])).tolist() \
        == list("ABCDDE")
    assert pick(np.array([[.5, .5, .5], [0, 1, 1], [0, 0, 1]])).tolist() == [0, 1, 2]
    r1 = p_sc(np.array([["A"] * 10, ["B"] * 10]), np.array(["A", "A"]), 5, np.random.default_rng(0))
    assert r1.tolist() == [1.0, 0.0]
    assert k_star(22000, 1150, 60) == 19
    print("unit checks passed")
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        sys.exit(0 if selftest() else 1)
    print(__doc__)
