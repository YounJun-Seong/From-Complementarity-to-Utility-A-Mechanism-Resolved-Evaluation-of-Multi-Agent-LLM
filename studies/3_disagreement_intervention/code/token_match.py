"""§4: fix k_add and k_het from the DEV token probe (token counts only; no labels are read).

  python token_match.py --dir studies/3_disagreement_intervention/results/dev/token_probe  ->  studies/3_disagreement_intervention/results/K_FROZEN.json
"""
import argparse
import json
import os

import numpy as np

from common import ROOT, jl


def tot(t):
    return t["input_tokens"] + t["output_tokens"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    P = [p for p in jl(f"{a.dir}/probe.jsonl") if p["error"] is None and p["probe"]]
    T = jl(f"{a.dir}/tdc.jsonl")
    H = [h for h in jl(f"{a.dir}/het.jsonl") if h["error"] is None]
    T_TDC = float(np.mean([tot(t["tokens"]) for t in T]))
    t_s = float(np.mean([tot(p["probe_tokens"]) / 4 for p in P]))
    t_r = float(np.mean([tot(h["recruit_tokens"]) for h in H]))
    t_h = float(np.mean([tot(h["sample_tokens"]) / len(h["votes"]) for h in H]))
    ks = np.arange(1, 201)
    k_add = int(ks[np.argmin(np.abs(ks * t_s - T_TDC))])
    k_het = int(ks[np.argmin(np.abs(t_r + ks * t_h - T_TDC))])
    out = dict(n_triggered=len(T), mean_tdc_incremental_tokens=T_TDC, mean_sample_tokens=t_s,
               mean_hetero_recruitment_tokens=t_r, mean_hetero_sample_tokens=t_h,
               predicted_samplemore_tokens=k_add * t_s, predicted_samplemore_ratio=T_TDC / (k_add * t_s),
               predicted_heteromore_tokens=t_r + k_het * t_h, predicted_heteromore_ratio=T_TDC / (t_r + k_het * t_h),
               n_token_probe=len(T), n_probe_questions=len(P), T_TDC=T_TDC, t_s=t_s, t_r=t_r, t_h=t_h,
               k_add=k_add, k_het=k_het, predicted_ratio_add=T_TDC / (k_add * t_s),
               predicted_ratio_het=T_TDC / (t_r + k_het * t_h),
               n_single_answer=sum(bool(t.get("single_answer")) for t in T), source=os.path.relpath(a.dir, ROOT))
    json.dump(out, open(f"{ROOT}/studies/3_disagreement_intervention/results/K_FROZEN.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
