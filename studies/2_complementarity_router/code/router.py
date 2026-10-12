"""Phase II router (PHASE2_FREEZE.md §5–§6), built on PILOT only. HELD-OUT is never read here.

Router: for each action a ∈ {STOP, SAMPLE, COLLABORATE}, an interpretable model of P(correct | a, x) on
the §3.4 pre-decision features; policy a(x) = argmax_a P̂_a(x) − λ·cost_a (accuracy units per token).
Labels use all PILOT runs: S = mean over 3 runs, SAMPLE = p_SC(k*) from the 60-sample SC pool,
COLLABORATE = mean M1 over 3 runs. Fractional labels enter as sample weights of a binary fit.
Costs (tokens/question, PILOT means): the 4-sample probe is charged to every routed method;
STOP = S + probe, SAMPLE = S + k*·sample (probe samples are part of SC-k*), COLLAB = S + probe + M1.

Selection rules (fixed before running this script; all on PILOT, 5-fold cross-fitting, seed 0):
  λ   — the grid value whose cross-fitted mean cost is closest to 0.5 × always-on M1 cost (§5);
  model — logistic regression unless a depth-≤3 tree has > 1 pp higher cross-fitted expected accuracy at
          its own λ; ties / smaller gains keep the logistic regression;
  τ   — difficulty routing: escalate when probe max vote share < τ; τ is the grid value whose PILOT mean
          cost for variant (b) (escalate to M1) is closest to Ours' cross-fitted mean cost.
Output: studies/2_complementarity_router/results/router/router_candidate.json (+ PILOT cross-fitted summary, descriptive only).

  python studies/2_complementarity_router/code/router.py
"""
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.dirname(__file__))
import features as F            # noqa: E402
import oracle_analysis as OA    # noqa: E402
import pilot_analysis as P      # noqa: E402

OUT = f"{P.ROOT}/studies/2_complementarity_router/results/router"
LAMBDA_GRID = np.round(np.concatenate([np.arange(0, 2, 0.05), np.arange(2, 20.5, 0.5)]), 3)  # pp / 10k tokens
TAU_GRID = np.round(np.arange(0.25, 1.01, 0.25 / 4), 4)          # probe max share takes values in {.25,.5,.75,1}
ACTIONS = ("STOP", "SAMPLE", "COLLABORATE")


def make_model(kind):
    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    return DecisionTreeClassifier(max_depth=3, min_samples_leaf=15, random_state=0)


def fit_frac(kind, X, y):
    """Binary fit with fractional labels: each row twice, weights y and 1 − y."""
    Xd = np.r_[X, X]
    yd = np.r_[np.ones(len(y)), np.zeros(len(y))]
    w = np.r_[y, 1 - y]
    keep = w > 0
    m = make_model(kind)
    if kind == "logreg":
        m.fit(Xd[keep], yd[keep], logisticregression__sample_weight=w[keep])
    else:
        m.fit(Xd[keep], yd[keep], sample_weight=w[keep])
    return m


def predict(models, X):
    return np.c_[[m.predict_proba(X)[:, 1] for m in models]].T          # Q x 3


def policy(Phat, cost, lam):
    return np.argmax(Phat - lam / 100 * cost[None, :] / 1e4, axis=1)


def crossfit(kind, X, Y):
    Phat = np.empty_like(Y)
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
        Phat[te] = predict([fit_frac(kind, X[tr], Y[tr, a]) for a in range(3)], X[te])
    return Phat


def vc_labels(Y, cost, lam):
    l = lam / 100 / 1e4
    cheap = np.where(Y[:, 1] - l * cost[1] > Y[:, 0] - l * cost[0], 1, 0)
    yC = ((Y[:, 2] - l * cost[2]) - Y[np.arange(len(Y)), cheap] + l * cost[cheap] > 0).astype(int)
    yA = ((Y[:, 1] - l * cost[1]) - (Y[:, 0] - l * cost[0]) > 0).astype(int)
    return yC, yA


def crossfit_vc(X, Y, cost, lam):
    """§6-conformant design: classifiers for utility-positive escalation, labels depend on λ.
    q_C = P[(y_M − max(y_S, y_SC)) − λΔcost_C > 0],  q_A = P[(y_SC − y_S) − λΔcost_A > 0];
    policy: COLLABORATE if q_C > .5, else SAMPLE if q_A > .5, else STOP."""
    yC, yA = vc_labels(Y, cost, lam)
    a = np.zeros(len(Y), int)
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
        q = []
        for y in (yC, yA):
            if y[tr].min() == y[tr].max():
                q.append(np.full(len(te), float(y[tr][0])))
            else:
                q.append(make_model("logreg").fit(X[tr], y[tr]).predict_proba(X[te])[:, 1])
        a[te] = np.where(q[0] > .5, 2, np.where(q[1] > .5, 1, 0))
    return a


def main():
    os.makedirs(OUT, exist_ok=True)
    tag = "Qwen2.5-14B-Instruct"                     # router is built for the decision model
    R = json.load(open(f"{P.BASE}/RESULTS.json"))[tag]
    k = R["k_star"]
    idx = json.load(open(f"{P.ROOT}/studies/2_complementarity_router/results/splits/pilot.json"))["idx"]
    qs = [r["question"] for r in P.jl(f"{P.ROOT}/data/splits/MedQA_p2pilot/test.jsonl")]
    S, SC, M, cost_rows = P.load(tag)
    gold = np.array([r["gold"] for r in S[0]])
    votes = [np.array([r["votes"] for r in run]) for run in SC]
    X = F.build(S[0], votes[0][:, :4], idx, qs)
    yS = np.array([[r["E3"] == r["gold"] for r in run] for run in S], float).mean(0)
    yM = np.array([[r["pred_answer"] == r["gold_answer"] for r in run] for run in M], float).mean(0)
    ySC = OA.p_sc(np.concatenate(votes[:3], axis=1), gold, k, np.random.default_rng(0))
    Y = np.c_[yS, ySC, yM]
    t = R["tok"]
    cost = np.array([t["S"] + 4 * t["SC"], t["S"] + k * t["SC"], t["S"] + 4 * t["SC"] + t["M"]])
    always_m = t["M"]
    target = 0.5 * always_m

    def choose(kind):
        Phat = crossfit(kind, X, Y)
        mc = np.array([cost[policy(Phat, cost, l)].mean() for l in LAMBDA_GRID])
        lam = float(LAMBDA_GRID[np.argmin(np.abs(mc - target))])
        a = policy(Phat, cost, lam)
        return dict(kind=kind, lam=lam, acc=float(Y[np.arange(len(a)), a].mean()), cost=float(cost[a].mean()),
                    share={ACTIONS[j]: float((a == j).mean()) for j in range(3)}, a=a)

    lr, tr = choose("logreg"), choose("tree")
    # §6-conformant V_C design (primary): λ by the same cost rule
    mcv = []
    for l_ in LAMBDA_GRID:
        mcv.append(cost[crossfit_vc(X, Y, cost, l_)].mean())
    lam_v = float(LAMBDA_GRID[np.argmin(np.abs(np.array(mcv) - target))])
    av = crossfit_vc(X, Y, cost, lam_v)
    vc = dict(kind="vc-logreg", lam=lam_v, acc=float(Y[np.arange(len(av)), av].mean()), cost=float(cost[av].mean()),
              share={ACTIONS[j]: float((av == j).mean()) for j in range(3)}, a=av)
    vc_samp = float(np.where(av == 2, Y[:, 1], Y[np.arange(len(av)), av].T).mean())
    print(f"V_C design: λ={lam_v}  exp. acc {vc['acc']:.3f}  cost {vc['cost']:.0f}  shares {vc['share']}  "
          f"| same routing with COLLAB->SC: {vc_samp:.3f}")
    best = tr if tr["acc"] - lr["acc"] > 0.01 else lr
    a = best["a"]
    # Ours-Sampling on PILOT (descriptive): COLLABORATE -> SC-k*, cost of SC-k* plus nothing else
    acc_samp = float(np.where(a == 2, Y[:, 1], Y[np.arange(len(a)), a]).mean())
    cost_samp = float(np.where(a == 2, cost[1], cost[a]).mean())

    # difficulty routing τ (variant b: escalate to M1), matched to Ours' cross-fitted cost
    pmax = X[:, F.NAMES.index("probe_max_share")]
    def diff_cost(tau, to):
        esc = pmax < tau
        return np.where(esc, cost[to], cost[0]).mean()
    tau = float(TAU_GRID[np.argmin([abs(diff_cost(tt, 2) - best["cost"]) for tt in TAU_GRID])])

    final = [fit_frac(best["kind"], X, Y[:, j]) for j in range(3)]
    params = []
    for m in final:
        if best["kind"] == "logreg":
            sc_, lr_ = m.named_steps["standardscaler"], m.named_steps["logisticregression"]
            params.append(dict(mean=sc_.mean_.tolist(), scale=sc_.scale_.tolist(),
                               coef=lr_.coef_[0].tolist(), intercept=float(lr_.intercept_[0])))
        else:
            tr_ = m.tree_
            params.append(dict(children_left=tr_.children_left.tolist(), children_right=tr_.children_right.tolist(),
                               feature=tr_.feature.tolist(), threshold=tr_.threshold.tolist(),
                               value=(tr_.value[:, 0, 1] / tr_.value[:, 0].sum(1)).tolist()))
    frozen = dict(model=best["kind"], features=F.NAMES, actions=ACTIONS, lam_pp_per_10k=best["lam"], k_star=k,
                  cost_tokens=dict(zip(ACTIONS, cost.tolist())), always_on_M1_tokens=always_m,
                  params=dict(zip(ACTIONS, params)), difficulty_tau=tau,
                  random_routing_shares=best["share"], built_on="PILOT (300), Qwen2.5-14B-Instruct, all runs")
    json.dump(frozen, open(f"{OUT}/router_candidate.json", "w"), indent=1)
    lines = [f"cross-fitted on PILOT (descriptive; HELD-OUT untouched). target cost 0.5 × M1 = {target:.0f} tok/q",
             *(f"{c['kind']:6s} λ={c['lam']:5.2f}  exp. acc {c['acc']:.3f}  cost {c['cost']:.0f}  shares "
               + " ".join(f"{k_}:{v:.2f}" for k_, v in c["share"].items()) for c in (lr, tr)),
             f"chosen model: {best['kind']}",
             f"Ours-Sampling (same routing, COLLAB -> SC-{k}): exp. acc {acc_samp:.3f}  cost {cost_samp:.0f}",
             f"always-on M1: exp. acc {yM.mean():.3f}  cost {always_m:.0f};  S: {yS.mean():.3f};  SC-{k}: {ySC.mean():.3f}",
             f"difficulty routing τ = {tau}"]
    # ---- Amendment 2: the frozen router is the §6 V_C design (b), λ by the pre-declared cost rule
    pmax_cost = lambda tt: np.where(pmax < tt, cost[2], cost[0]).mean()
    tau_v = float(TAU_GRID[np.argmin([abs(pmax_cost(tt) - vc["cost"]) for tt in TAU_GRID])])
    yC, yA = vc_labels(Y, cost, lam_v)
    fp = {}
    for name, y in (("q_C", yC), ("q_A", yA)):
        m = make_model("logreg").fit(X, y)
        sc_, lr_ = m.named_steps["standardscaler"], m.named_steps["logisticregression"]
        fp[name] = dict(mean=sc_.mean_.tolist(), scale=sc_.scale_.tolist(), coef=lr_.coef_[0].tolist(),
                        intercept=float(lr_.intercept_[0]), n_pos=int(y.sum()))
    frozen_b = dict(design="V_C direct (§6), logistic regression ×2", features=F.NAMES, actions=ACTIONS,
                    rule="COLLABORATE if q_C > .5, else SAMPLE if q_A > .5, else STOP",
                    lam_pp_per_10k=lam_v,
                    lam_rule="grid value whose PILOT cross-fitted mean cost is closest to 0.5 × always-on M1; "
                             f"target {target:.0f}, cost at λ=0 {mcv[0]:.0f} (maximum over the grid)",
                    k_star=k, probe_samples=4, pilot_cost_tokens=dict(zip(ACTIONS, cost.tolist())),
                    always_on_M1_tokens_pilot=always_m, params=fp, difficulty_tau=tau_v,
                    random_routing_shares=vc["share"], pilot_crossfit=dict(acc=vc["acc"], cost=vc["cost"],
                                                                          acc_collab_to_sc=vc_samp),
                    built_on="PILOT (300), Qwen2.5-14B-Instruct, all runs; HELD-OUT never read")
    json.dump(frozen_b, open(f"{OUT}/router_frozen.json", "w"), indent=1)
    lines += [f"FROZEN (Amendment 2): V_C design λ={lam_v}, cross-fit exp. acc {vc['acc']:.3f}, cost {vc['cost']:.0f}, "
              f"COLLAB->SC {vc_samp:.3f}; positives q_C {fp['q_C']['n_pos']}, q_A {fp['q_A']['n_pos']}; τ_b = {tau_v}"]
    open(f"{OUT}/PILOT_CROSSFIT.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
