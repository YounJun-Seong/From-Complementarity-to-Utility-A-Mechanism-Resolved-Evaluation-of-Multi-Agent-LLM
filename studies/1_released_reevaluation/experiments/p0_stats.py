"""P0-6: paired statistics for every headline comparison on full MedQA.
For each pair (a, b): accuracies, Δ = b - a, paired-bootstrap 95% CI, exact McNemar p,
discordant counts, and TOST p (paired t, as in the paper) over a grid of margins.

  python experiments/p0_stats.py
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.join(os.path.dirname(__file__), "..")
RNG = np.random.default_rng(0)
DELTAS = (0.01, 0.02, 0.03, 0.04, 0.05)


def jsonl_correct(path):
    return np.array([int(json.loads(l)["pred"] == json.loads(l)["gold"]) for l in open(path)])


def medagents_rerun(return_rows=False):
    """M1 correctness vector (n=1273) or None if incomplete. run.py stores no index, so the
    index is shard start (shard s covers [80s, 80s+80)) + line number, and every record's
    question text is checked against the dataset."""
    QA = [json.loads(l) for l in open(f"{ROOT}/external/MedAgents/datasets/MedQA/test.jsonl")]
    by = {}
    for d in glob.glob(f"{ROOT}/runs/m1_medagents/shards/*/chatgpt-syn_verif"):
        start = 80 * int(os.path.basename(os.path.dirname(d)))
        for k, l in enumerate(open(d)):
            r = json.loads(l)
            i = start + k
            q = QA[i]["question"]
            assert r["question"].rstrip("?") == q.rstrip("?") or r["question"] == q + "?", f"misaligned {d}:{k}"
            by[i] = r
    if len(by) != 1273:
        return None
    rows = [by[i] for i in range(1273)]
    if return_rows:
        return rows
    return np.array([int(r["pred_answer"] == r["gold_answer"]) for r in rows])


def systems():
    """Epoch systems only (PREREGISTRATION.md). Supplement numbers are never paired here."""
    S = {}
    for f in sorted(glob.glob(f"{ROOT}/runs/p0_sweep/*.jsonl")):
        role, budget = os.path.basename(f)[:-6].split("_")
        S[f"base_cot[{role},{budget}]"] = jsonl_correct(f)
    ex = f"{ROOT}/runs/p0_extraction.jsonl"
    if os.path.exists(ex):
        E = [json.loads(l) for l in open(ex)]
        for c in ("E1", "E2", "E3", "E3f"):
            S[f"S zs-CoT [{c}]"] = np.array([int(r[c] == r["gold"]) for r in E])
    m = medagents_rerun()
    if m is not None:
        S["M1 MedAgents"] = m
    return {k: v for k, v in S.items() if len(v) == 1273}


def mcnemar_exact(a, b):
    n01, n10 = int(((a == 0) & (b == 1)).sum()), int(((a == 1) & (b == 0)).sum())
    p = stats.binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return n01, n10, p


def tost_t(d, delta):
    n, m, se = len(d), d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    return max(stats.t.sf((m + delta) / se, n - 1), stats.t.cdf((m - delta) / se, n - 1))


def compare(S, a, b, B=10000):
    x, y = S[a], S[b]
    d = y - x
    bs = np.array([d[RNG.integers(0, len(d), len(d))].mean() for _ in range(B)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    n01, n10, p = mcnemar_exact(x, y)
    row = {"a": a, "b": b, "acc_a": x.mean(), "acc_b": y.mean(), "delta": d.mean(),
           "ci95_lo": lo, "ci95_hi": hi, "b_only": n01, "a_only": n10, "mcnemar_p": p}
    row.update({f"tost_{dl}": tost_t(d, dl) for dl in DELTAS})
    row.update(both_right=int(((x == 1) & (y == 1)).sum()), both_wrong=int(((x == 0) & (y == 0)).sum()))
    return row


PRIMARY_DELTA = 0.03
B1, B2, B3, B4 = ("base_cot[empty,300]", "base_cot[expert,300]", "base_cot[empty,700]", "base_cot[expert,700]")
M1, S_ = "M1 MedAgents", "S zs-CoT [E3]"
PRIMARY = [(B1, B3), (B3, M1), (S_, M1)]                       # frozen, PREREGISTRATION.md
EQUIVALENCE = {(B3, M1), (S_, M1)}                             # TOST only for these
SECONDARY = [(B1, B2), (B3, B4), (B2, B4), (B1, M1), (B4, M1),
             ("S zs-CoT [E3f]", M1), ("S zs-CoT [E1]", M1), ("S zs-CoT [E2]", M1)]


def show(r, primary):
    tost = " ".join(("*" if dl == PRIMARY_DELTA else "") + f"δ{dl}:{r[f'tost_{dl}']:.3f}" for dl in DELTAS)
    print(f"{r['b']} − {r['a']}\n    Δ={r['delta']:+.4f} 95%CI[{r['ci95_lo']:+.4f},{r['ci95_hi']:+.4f}]  "
          f"McNemar p={r['mcnemar_p']:.2g}")
    print(f"    transitions (a→b): right→right {r['both_right']}, right→wrong {r['a_only']}, "
          f"wrong→right {r['b_only']}, wrong→wrong {r['both_wrong']}")
    if primary:
        print(f"    TOST p  {tost}   (* primary margin)")


def main():
    S = systems()
    print("accuracy (n=1273, P0 epoch):")
    for k, v in S.items():
        print(f"  {k:24s} {v.mean():.3f}")
    rows = []
    for tag, pairs in (("PRIMARY", PRIMARY), ("SECONDARY", SECONDARY)):
        print(f"\n=== {tag} ===")
        for a, b in pairs:
            if a in S and b in S:
                r = compare(S, a, b); r["tier"] = tag.lower(); rows.append(r); show(r, (a, b) in EQUIVALENCE)
            else:
                print(f"{b} − {a}: pending ({', '.join(x for x in (a, b) if x not in S)} missing)")
    pd.DataFrame(rows).to_csv(f"{ROOT}/runs/p0_stats.csv", index=False)


if __name__ == "__main__":
    main()
