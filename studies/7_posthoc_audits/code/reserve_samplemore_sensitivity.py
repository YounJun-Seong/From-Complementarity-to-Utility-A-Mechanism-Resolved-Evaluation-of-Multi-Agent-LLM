"""Post-hoc compute sensitivity for the sealed RESERVE disagreement study (paper Appendix A, Compute Accounting).
SampleMore drew k_add = 4 samples in one request, so the generation prompt was charged once. Here SampleMore is
re-scored with only its first k ∈ {2, 3} additional samples (probe answers + first k votes), against unchanged TDC.
No LLM calls.

  python reserve_samplemore_sensitivity.py  ->  ../results/RESERVE_SAMPLEMORE_SENSITIVITY.txt
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..", "..")
sys.path.insert(0, os.path.join(HERE, "..", "..", "3_disagreement_intervention", "code"))
import aggregate as AG   # noqa: E402

R = f"{ROOT}/studies/3_disagreement_intervention/results"
B, SEED = 10000, 0


def jl(p):
    return [json.loads(l) for l in open(p)]


def main():
    sm = {r["qid"]: r for r in jl(f"{R}/reserve/main/sample_more.jsonl")}
    td = {r["qid"]: r for r in jl(f"{R}/reserve/main/tdc.jsonl")}
    pr = {r["qid"]: r for r in jl(f"{R}/reserve/main/probe.jsonl")}
    lab = {int(k): v for k, v in json.load(open(f"{R}/splits/reserve_labels.json")).items()}
    q = sorted(sm)
    acc = lambda f: np.array([f(i) == lab[i] for i in q], float)
    a_td = acc(lambda i: td[i]["final_answer"])
    L = [f"RESERVE triggered n={len(q)}; TDC {a_td.mean():.3f}; SampleMore (k_add=4, frozen) {acc(lambda i: sm[i]['final_answer']).mean():.3f}"]
    rng = np.random.default_rng(SEED)
    for k in (3, 2):
        a = acc(lambda i: AG.majority(pr[i]["answers"] + sm[i]["votes"][:k]))
        d = a_td - a
        bs = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        L.append(f"SampleMore k_add={k}: acc {a.mean():.3f}; TDC − SampleMore {d.mean()*100:+.1f} pp [{lo*100:+.1f}, {hi*100:+.1f}]")
    open(f"{ROOT}/studies/7_posthoc_audits/results/RESERVE_SAMPLEMORE_SENSITIVITY.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
