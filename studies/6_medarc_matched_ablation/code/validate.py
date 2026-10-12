"""Label-free validation (PHASE6_FREEZE.md §5): completeness, errors, round-0 byte identity across A/B/C, B has no
summary, C/A summary sources, call structure, SC request counts, truncation, tokens."""
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
D = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/studies/6_medarc_matched_ablation/results/main"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 472


def jl(p):
    return sorted([json.loads(l) for l in open(p)], key=lambda x: x["idx"])


ok = True
arms = {m: [jl(f"{D}/{m}_run{r}.jsonl") for r in (1, 2, 3)] for m in "ABC"}
for m, runs in arms.items():
    for r, run in enumerate(runs, 1):
        err = sum(x["error"] is not None for x in run)
        struct = {(sum(c["kind"] == "agent" for c in x["calls"]), sum(c["kind"] == "summary" for c in x["calls"])) for x in run if not x["error"]}
        trunc = np.mean([c["finish"] == "length" for x in run for c in x["calls"]])
        print(f"{m} run{r}: n={len(run)} errors={err} (agent,summary)={struct} trunc={trunc:.4f}")
        ok &= len(run) == N and err == 0 and struct == ({(9, 0)} if m == "B" else {(9, 6)})
for r in range(3):
    same = all(arms["A"][r][q]["agent_contexts"][i][:2] == arms[m][r][q]["agent_contexts"][i][:2]
               for m in "BC" for q in range(N) for i in range(3))
    nosum = all("Summary of other agents" not in msg["content"] for x in arms["B"][r] for ctx in x["agent_contexts"] for msg in ctx)
    srcA = all(s["source_idx"] == x["idx"] for x in arms["A"][r] for s in x["summaries"])
    srcC = all(s["source_idx"] == arms["A"][r][(q + 1) % N]["idx"] for q, x in enumerate(arms["C"][r]) for s in x["summaries"])
    print(f"run{r+1}: round-0 identical A=B=C {same}; B no summary {nosum}; A source=self {srcA}; C source=next {srcC}")
    ok &= same and nosum and srcA and srcC
for name, k in (("cot", None), ("sc", 25)):
    runs = [jl(f"{D}/{name}_run{r}.jsonl") for r in ((1, 2, 3) if name == "cot" else range(1, 7))]
    for r, run in enumerate(runs, 1):
        err = sum(x["error"] is not None for x in run)
        extra = f" requests/q={ {len(x['calls']) for x in run} }" if k else ""
        print(f"{name} run{r}: n={len(run)} errors={err}{extra}")
        ok &= len(run) == N and err == 0 and (not k or {len(x["calls"]) for x in run} == {k})
tok = lambda x: sum(c["input_tokens"] + c["output_tokens"] for c in x["calls"])
print("tokens/q: " + ", ".join(f"{m} {np.mean([tok(x) for run in arms[m] for x in run]):.0f}" for m in "ABC")
      + f", SC-25 {np.mean([tok(x) for r in range(1, 7) for x in jl(f'{D}/sc_run{r}.jsonl')]):.0f}")
print("VALID" if ok else "INVALID")
