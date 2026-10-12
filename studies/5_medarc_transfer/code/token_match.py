"""PHASE5_FREEZE.md §6: k* from the token probe (MedARC run 1 and one round-0 sample per probe question). No labels."""
import json
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
D = f"{ROOT}/studies/5_medarc_transfer/results/token_probe"


def tok(r):
    return sum(c["input_tokens"] + c["output_tokens"] for c in r["calls"])


def main():
    M = [json.loads(l) for l in open(f"{D}/medarc_run1.jsonl")]
    S = [json.loads(l) for l in open(f"{D}/sc1_run1.jsonl")]
    assert all(r["error"] is None for r in M + S)
    T_M, t_s = float(np.mean([tok(r) for r in M])), float(np.mean([tok(r) for r in S]))
    ks = np.arange(1, 201)
    k = int(ks[np.argmin(np.abs(ks * t_s - T_M))])
    out = dict(n_probe=len(M), T_M=T_M, t_s=t_s, k_star=k, predicted_sc_tokens=k * t_s, predicted_ratio=T_M / (k * t_s),
               note="t_s = one independent round-0 sample (one request, n=1); the SC arm issues k* separate n=1 requests (Amendment 2)")
    json.dump(out, open(f"{ROOT}/studies/5_medarc_transfer/results/K_FROZEN.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
