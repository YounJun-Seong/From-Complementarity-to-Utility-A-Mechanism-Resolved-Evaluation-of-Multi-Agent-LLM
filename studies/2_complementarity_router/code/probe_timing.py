"""Timing probe read-out (PHASE2_FREEZE.md §3.3, §3.6). Reads only timestamps and token counts; never
the gold label or any prediction. Projects full-PILOT GPU-hours per backbone linearly from 20 to 300
questions (conservative: the probe runs fewer parallel shards than the full pilot) and applies the
48 GPU-hour rule.

  python studies/2_complementarity_router/code/probe_timing.py
"""
import glob
import json
import os

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
N_PROBE, N_PILOT, LIMIT_H = 20, 300, 48.0


def tokens(rows, a="prompt_tokens", b="completion_tokens"):
    return sum(r.get(a, 0) + r.get(b, 0) for r in rows)


def main():
    for d in sorted(glob.glob(f"{ROOT}/studies/2_complementarity_router/results/probe/*/")):
        t = dict(l.split("\t")[:2] for l in open(f"{d}/timing.tsv").read().splitlines())
        wall = (int(t["jobs_done"]) - int(t["server_ready"])) / 3600
        S = [json.loads(l) for f in glob.glob(f"{d}/S/run*.jsonl") for l in open(f)]
        SC = [json.loads(l) for f in glob.glob(f"{d}/SC/run*.jsonl") for l in open(f)]
        M = [json.loads(l) for f in glob.glob(f"{d}/M1/run*/shard*/cost.jsonl") for l in open(f)]
        tok = dict(S=tokens(S, "S_prompt_tokens", "S_completion_tokens") / N_PROBE / 3,
                   SC_sample=tokens(SC) / N_PROBE / len(glob.glob(f"{d}/SC/run*.jsonl")) / 20,
                   M=tokens(M) / N_PROBE / 3)
        proj = wall * N_PILOT / N_PROBE
        branch = "cut P to 40 samples (P_RUNS=5)" if proj > LIMIT_H else "full design (P = 60 samples)"
        print(f"{os.path.basename(d.rstrip('/')):22s} probe wall {wall*60:5.1f} min  -> projected PILOT {proj:5.1f} GPU-h  "
              f"| tokens/q S {tok['S']:.0f}, SC/sample {tok['SC_sample']:.0f}, M1 {tok['M']:.0f}  => {branch}")


if __name__ == "__main__":
    main()
