"""Phase V execution validation (no labels): completion, errors, SC = exactly k* independent requests,
MedARC 15-call structure, token accounting, truncation, controlled-parser unparsed rates, realized token ratio."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aggregate as AG   # noqa: E402
import parsers as P      # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
D = f"{ROOT}/studies/5_medarc_transfer/results/main"


def jl(p):
    return [json.loads(l) for l in open(p)]


def tok(r):
    return sum(c["input_tokens"] + c["output_tokens"] for c in r["calls"])


def main():
    K = json.load(open(f"{ROOT}/studies/5_medarc_transfer/results/K_FROZEN.json"))["k_star"]
    files = [f"medarc_run{r}" for r in (1, 2, 3)] + [f"zeroshot_run{r}" for r in (1, 2, 3)] + \
            [f"cot_run{r}" for r in (1, 2, 3)] + [f"sc_run{r}" for r in range(1, 7)]
    ok, A = True, {}
    for f in files:
        rows = jl(f"{D}/{f}.jsonl"); A[f] = rows
        complete = sorted(r["idx"] for r in rows) == list(range(498))
        err = sum(r["error"] is not None for r in rows)
        line = f"{f:14s} n={len(rows)} complete={complete} errors={err}"
        if f.startswith("sc"):
            req = {len(r["calls"]) for r in rows}; ntx = {len(r["texts"]) for r in rows}
            line += f" requests/q={req} samples/q={ntx} trunc={np.mean([x == 'length' for r in rows for x in r['finish']]):.4f}"
            ok &= req == {K} and ntx == {K}
        if f.startswith("medarc"):
            struct = {(sum(c['kind'] == 'agent' for c in r['calls']), sum(c['kind'] == 'summary' for c in r['calls'])) for r in rows}
            line += f" (agent,summary)/q={struct} trunc={np.mean([c['finish'] == 'length' for r in rows for c in r['calls']]):.4f}"
            line += f" unparsed_turns={np.mean([P.answer(t) is None for r in rows for a in AG.turns(r) for t in a]):.4f}"
            line += f" conf_extracted={np.mean([P.confidence(t) > 0 for r in rows for a in AG.turns(r) for t in a]):.4f}"
            ok &= struct == {(9, 6)}
        ok &= complete and err == 0
        print(line)
    tm = np.mean([tok(r) for f in files if f.startswith("medarc") for r in A[f]])
    ts = np.mean([tok(r) for f in files if f.startswith("sc") for r in A[f]])
    print(f"tokens/q: MedARC {tm:.0f}, SC-{K} {ts:.0f}; realized ratio T_SC/T_M = {ts / tm:.3f} (predicted {1 / 0.991:.3f})")
    print("VALID" if ok else "INVALID")


if __name__ == "__main__":
    main()
