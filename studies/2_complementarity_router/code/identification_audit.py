"""Identification-capacity audit of the frozen Phase II router (post-hoc diagnostic on the already-opened HELD-OUT;
no threshold is selected here and nothing is reported as new prospective performance).

Score  : q_C(x) of the frozen V_C router (router_frozen.json), computed from pre-decision features only.
Labels : primary   y = 1[M1 correct and SC-22 wrong]           (collaboration beneficial vs compute-matched sampling)
         secondary y2 = 1[M1 correct and both S and SC-22 wrong] (beneficial vs every cheap action)
Reports: AUROC / AUPRC (bootstrap 95% CI over questions, 10,000, seed 0) against prevalence; top-k recall at
         collaboration budgets k ∈ {1, 5, 10, 20, 30}%; coverage–cost curve (route the top-k% by q_C to M1, every
         other question keeps its frozen STOP/SAMPLE action); calibration (q_C deciles vs observed y2-like rate
         P(M1 beats the better cheap action), mean q_C vs base rate, Brier).

  python studies/2_complementarity_router/code/identification_audit.py  ->  studies/2_complementarity_router/results/heldout/IDENTIFICATION_AUDIT.{txt,json}
"""
import json
import os
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
import features as F            # noqa: E402
import heldout_analysis as H    # noqa: E402
import router_policy as RP      # noqa: E402

B, SEED = 10000, 0
BUDGETS = (0.01, 0.05, 0.10, 0.20, 0.30)


def main():
    rng = np.random.default_rng(SEED)
    frozen = RP.load()
    S, SC, HET, M, cost, gold = H.load()
    idx = json.load(open(f"{H.ROOT}/studies/2_complementarity_router/results/splits/heldout.json"))["idx"]
    qs = [r["question"] for r in H.jl(f"{H.ROOT}/data/splits/MedQA_p2heldout/test.jsonl")]
    X = F.build(S, np.array([r["votes"] for r in SC])[:, :4], idx, qs)
    a, qC, qA = RP.act(frozen, X)
    c = dict(S=np.array([r["E3"] == r["gold"] for r in S], float), SC=np.array([r["pred"] == r["gold"] for r in SC], float),
             M=np.array([r["pred_answer"] == r["gold_answer"] for r in M], float))
    t = dict(S=np.array([r["S_prompt_tokens"] + r["S_completion_tokens"] for r in S], float),
             SC=np.array([r["prompt_tokens"] + r["completion_tokens"] for r in SC], float),
             M=np.array([r["prompt_tokens"] + r["completion_tokens"] for r in cost], float))
    probe = t["SC"] * 4 / 22
    y = ((c["M"] == 1) & (c["SC"] == 0)).astype(int)
    y2 = ((c["M"] == 1) & (c["S"] == 0) & (c["SC"] == 0)).astype(int)
    Q = len(y)

    def auc_ci(lab):
        if lab.min() == lab.max():
            return None
        bs = []
        for _ in range(B // 10):
            b = rng.integers(0, Q, Q)
            if lab[b].min() != lab[b].max():
                bs.append((roc_auc_score(lab[b], qC[b]), average_precision_score(lab[b], qC[b])))
        bs = np.array(bs)
        return dict(n_pos=int(lab.sum()), prevalence=float(lab.mean()),
                    AUROC=[float(roc_auc_score(lab, qC)), *map(float, np.percentile(bs[:, 0], [2.5, 97.5]))],
                    AUPRC=[float(average_precision_score(lab, qC)), *map(float, np.percentile(bs[:, 1], [2.5, 97.5]))])
    res = dict(n=Q, frozen_collab_selected=int((a == RP.COLLABORATE).sum()),
               qC_summary=dict(mean=float(qC.mean()), median=float(np.median(qC)), max=float(qC.max()),
                               share_above_0_5=float((qC > 0.5).mean())),
               primary=auc_ci(y), secondary=auc_ci(y2))

    order = np.argsort(-qC, kind="stable")
    base_c = np.select([a == RP.STOP, a == RP.SAMPLE], [c["S"], c["SC"]], c["M"])
    base_t = np.select([a == RP.STOP, a == RP.SAMPLE], [t["S"] + probe, t["S"] + t["SC"]], t["S"] + probe + t["M"])
    curve = []
    for k in (0.0,) + BUDGETS:
        top = np.zeros(Q, bool); top[order[:int(round(k * Q))]] = True
        acc = np.where(top, c["M"], base_c); tok = np.where(top, t["S"] + probe + t["M"], base_t)
        rec = float(y[top].sum() / max(y.sum(), 1))
        rand = float(k)                                    # expected recall of a random selector at budget k
        curve.append(dict(budget=k, n_selected=int(top.sum()), recall_primary=rec, random_recall=rand,
                          precision_primary=float(y[top].mean()) if top.any() else None,
                          accuracy=float(acc.mean()), tokens=float(tok.mean())))
    res["coverage_cost"] = curve

    gain = c["M"] - np.maximum(c["S"], c["SC"])            # observed (single-run) analogue of the q_C target at λ=0
    yb = (gain > 0).astype(int)
    bins = np.quantile(qC, np.linspace(0, 1, 11)); bins[-1] += 1e-9
    cal = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (qC >= lo) & (qC < hi)
        if m.any():
            cal.append(dict(qC_mean=float(qC[m].mean()), observed=float(yb[m].mean()), n=int(m.sum())))
    res["calibration"] = dict(target="1[M1 correct and both cheap actions wrong]", mean_qC=float(qC.mean()),
                              base_rate=float(yb.mean()), brier=float(np.mean((qC - yb) ** 2)), deciles=cal)

    L = [f"Identification audit (post-hoc, HELD-OUT n={Q}); frozen router chose COLLABORATE on {res['frozen_collab_selected']}",
         f"q_C: mean {res['qC_summary']['mean']:.3f}, median {res['qC_summary']['median']:.3f}, max {res['qC_summary']['max']:.3f}, "
         f"share > 0.5 {res['qC_summary']['share_above_0_5']:.3f}"]
    for name, r in (("primary  y = M1 right & SC-22 wrong", res["primary"]), ("secondary y2 = M1 right & S, SC wrong", res["secondary"])):
        L.append(f"{name}: n_pos {r['n_pos']} (prevalence {r['prevalence']:.3f}); AUROC {r['AUROC'][0]:.3f} "
                 f"[{r['AUROC'][1]:.3f}, {r['AUROC'][2]:.3f}]; AUPRC {r['AUPRC'][0]:.3f} [{r['AUPRC'][1]:.3f}, {r['AUPRC'][2]:.3f}]")
    L.append("coverage–cost (route top-k% by q_C to M1; others keep frozen action):")
    for r in curve:
        L.append(f"  k={r['budget']:.2f}  n={r['n_selected']:3d}  recall {r['recall_primary']:.3f} (random {r['random_recall']:.2f})  "
                 f"precision {r['precision_primary'] if r['precision_primary'] is None else round(r['precision_primary'], 3)}  "
                 f"acc {r['accuracy']:.3f}  tok/q {r['tokens']:.0f}")
    L.append(f"calibration ({res['calibration']['target']}): mean q_C {res['calibration']['mean_qC']:.3f} vs base rate "
             f"{res['calibration']['base_rate']:.3f}; Brier {res['calibration']['brier']:.3f}")
    L += [f"  decile q_C {d['qC_mean']:.3f} -> observed {d['observed']:.3f} (n={d['n']})" for d in cal]
    out = f"{H.ROOT}/studies/2_complementarity_router/results/heldout"
    open(f"{out}/IDENTIFICATION_AUDIT.txt", "w").write("\n".join(L) + "\n")
    json.dump(res, open(f"{out}/IDENTIFICATION_AUDIT.json", "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
