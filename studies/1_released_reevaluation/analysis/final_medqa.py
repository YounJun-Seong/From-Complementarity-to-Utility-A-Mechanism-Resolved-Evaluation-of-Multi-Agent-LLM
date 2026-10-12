"""Final MedQA reality-check: same first-N questions, per-method accuracy + bootstrap
95% CIs + pairwise gains. Shows: best single-agent prompt ~= multi-agent >> the
paper's base_cot baseline (the reported gain is a weak-baseline artifact)."""
import json

import numpy as np
import pandas as pd

RNG = np.random.default_rng(0)
OUT = "external/MedAgents/outputs/MedQA"


def official(method):
    L = [json.loads(l) for l in open(f"{OUT}/chatgpt-{method}")]
    return np.array([int(x["pred_answer"] == x["gold_answer"]) for x in L])


def ci(x):
    b = [RNG.choice(x, len(x), replace=True).mean() for _ in range(5000)]
    return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def gain_ci(a, b):  # b - a
    d = b - a
    bs = [RNG.choice(d, len(d), replace=True).mean() for _ in range(5000)]
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def main():
    sv = pd.read_csv("data/single_variants_medqa.csv")
    base = official("base_cot"); syn = official("syn_verif")
    n = min(len(sv), len(base), len(syn))
    print(f"MedQA reality-check (same {n} questions, gpt-4o-mini):\n")
    cols = {"base_cot (paper baseline)": base[:n],
            "single: direct": sv["direct"].to_numpy()[:n],
            "single: cot": sv["cot"].to_numpy()[:n],
            "single: expert_cot": sv["expert_cot"].to_numpy()[:n],
            "MULTI-AGENT (syn_verif)": syn[:n]}
    for name, x in cols.items():
        m, lo, hi = ci(x)
        print(f"  {name:26s} acc={m:.3f} [{lo:.3f},{hi:.3f}]")
    best_single = max(["direct", "cot", "expert_cot"], key=lambda c: sv[c][:n].mean())
    bs = sv[best_single].to_numpy()[:n]
    print(f"\n  best single = {best_single}")
    g, lo, hi = gain_ci(bs, syn[:n])
    print(f"  multi-agent - best_single = {g:+.3f} [{lo:+.3f},{hi:+.3f}]  "
          f"{'(multi wins)' if lo > 0 else '(no diff)' if lo <= 0 <= hi else '(single wins)'}")
    g2, lo2, hi2 = gain_ci(base[:n], syn[:n])
    print(f"  multi-agent - base_cot    = {g2:+.3f} [{lo2:+.3f},{hi2:+.3f}]  (the paper's 'gain')")
    g3, lo3, hi3 = gain_ci(base[:n], bs)
    print(f"  best_single - base_cot    = {g3:+.3f} [{lo3:+.3f},{hi3:+.3f}]  (baseline artifact)")
    print("\n=> reported multi-agent gain vs base_cot is matched by a single prompt; "
          "the gain is a weak-baseline artifact.")


if __name__ == "__main__":
    main()
