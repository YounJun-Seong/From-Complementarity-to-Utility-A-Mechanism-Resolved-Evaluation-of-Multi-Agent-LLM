"""Phase VI arms (PHASE6_FREEZE.md §2). Gold is never read.

  python run_ablation.py --mode round0|A|B|C|cot|sc --run R [--n N] [--data inputs.jsonl] --out f.jsonl
Seeds: run·10⁶ + idx·100 + call number; round-0 calls are 0–2 in every MedARC arm (identical cache keys).
B keeps A's call numbering (a skipped summary still consumes its number) so agent-call seeds align.
C summarises arm A's latest turns of the next question (idx+1 mod N, same run, same round and agent positions).
"""
import argparse
import json
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "5_medarc_transfer", "code"))
import released as R                    # noqa: E402
from client import chat                 # noqa: E402
import run_arms as RA                   # noqa: E402

ROOT = os.path.join(HERE, "..", "..", "..")
T, MAXTOK = RA.T_GEN, RA.MAX_TOK
NO_COMM = ("Please re-analyze the question. Provide your updated answer to the question: {question}\n"
           "Present your reasoning before final answer.")


def round0(d, run):
    seed0 = run * 10**6 + d["idx"] * 100
    out = []
    for i in range(R.AGENTS):
        c = chat([{"role": "user", "content": R.initial_prompt(d["contexts"], d["question"])}], T, MAXTOK, seed0 + i)
        out.append(dict(text=c["texts"][0], input_tokens=c["input_tokens"], output_tokens=c["output_tokens"]))
    return dict(round0=out)


def medarc(d, run, mode, source=None):
    """mode A: identical to phase5 run_arms.medarc; B: no-summary template; C: placebo summary from `source`
    (arm-A record of another question)."""
    seed0, call = run * 10**6 + d["idx"] * 100, 0
    ctxs = [[{"role": "user", "content": R.initial_prompt(d["contexts"], d["question"])}] for _ in range(R.AGENTS)]
    calls, summaries = [], []
    for rnd in range(R.ROUNDS):
        for i, ctx in enumerate(ctxs):
            if rnd != 0:
                if mode == "B":
                    call += 1                                   # keep A's numbering
                    ctx.append({"role": "user", "content": NO_COMM.format(question=d["question"])})
                    summaries.append(dict(round=rnd, agent=i, text=None, source_idx=None))
                else:
                    if mode == "A":
                        others, src = ctxs[:i] + ctxs[i + 1:], d["idx"]
                    else:
                        sctx = source["agent_contexts"]
                        others, src = sctx[:i] + sctx[i + 1:], source["idx"]
                    resp = R.other_responses(others, 2 * rnd - 1)
                    s = chat(R.summary_messages(resp), T, MAXTOK, seed0 + call); call += 1
                    calls.append(dict(kind="summary", round=rnd, agent=i, input_tokens=s["input_tokens"],
                                      output_tokens=s["output_tokens"], finish=s["finish"][0]))
                    summaries.append(dict(round=rnd, agent=i, text=s["texts"][0], source_idx=src))
                    ctx.append(R.debate_message(s["texts"][0], d["question"]))
            c = chat(ctx, T, MAXTOK, seed0 + call); call += 1
            calls.append(dict(kind="agent", round=rnd, agent=i, input_tokens=c["input_tokens"],
                              output_tokens=c["output_tokens"], finish=c["finish"][0]))
            ctx.append({"role": "assistant", "content": c["texts"][0]})
    return dict(agent_contexts=ctxs, summaries=summaries, calls=calls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["round0", "A", "B", "C", "cot", "sc"])
    ap.add_argument("--run", type=int, required=True)
    ap.add_argument("--n", type=int, default=-1)
    ap.add_argument("--data", default=f"{ROOT}/studies/6_medarc_matched_ablation/results/data/inputs.jsonl")
    ap.add_argument("--a_file", default=None, help="arm-A output of the same run (mode C)")
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    D = [json.loads(l) for l in open(a.data)]
    D = D if a.n < 0 else D[:a.n]
    srcmap = None
    if a.mode == "C":
        A = {r["idx"]: r for r in map(json.loads, open(a.a_file))}
        srcmap = {d["idx"]: A[D[(j + 1) % len(D)]["idx"]] for j, d in enumerate(D)}
    fn = dict(round0=lambda d: round0(d, a.run), A=lambda d: medarc(d, a.run, "A"), B=lambda d: medarc(d, a.run, "B"),
              C=lambda d: medarc(d, a.run, "C", srcmap[d["idx"]]),
              cot=lambda d: RA.single(d, a.run, R.cot_prompt, 0.0), sc=lambda d: RA.sc(d, a.run, a.k))[a.mode]

    def job(d):
        try:
            return dict(idx=d["idx"], pubid=d["pubid"], arm=a.mode, run=a.run, error=None, **fn(d))
        except Exception:
            return dict(idx=d["idx"], pubid=d["pubid"], arm=a.mode, run=a.run, error=traceback.format_exc())
    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(job, D))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tok = sum(c["input_tokens"] + c["output_tokens"] for r in rows if not r["error"] for c in r.get("calls", [])) / max(1, len(rows))
    print(f"{a.mode} run{a.run}: n={len(rows)} errors={sum(r['error'] is not None for r in rows)} tokens/q={tok:.0f}")


if __name__ == "__main__":
    main()
