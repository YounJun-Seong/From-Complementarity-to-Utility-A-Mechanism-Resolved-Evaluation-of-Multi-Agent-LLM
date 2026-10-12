"""Phase V arms (PHASE5_FREEZE.md §2). Gold is never read.

  python run_arms.py --arm medarc|zeroshot|cot|sc --split eval|probe --run R [--k K] --out f.jsonl
Seeds: run·10⁶ + idx·100 + call number (calls numbered in the released order).
"""
import argparse
import json
import os
import traceback
from concurrent.futures import ThreadPoolExecutor

import released as R
from client import chat

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
T_GEN, MAX_TOK = 1.0, 1024


def medarc(d, run):
    """Port of gen_pubmedqa_deepseek.py's per-question loop (homogeneous; summarizer = same model)."""
    seed0, call = run * 10**6 + d["idx"] * 100, 0
    agent_contexts = [[{"role": "user", "content": R.initial_prompt(d["contexts"], d["question"])}] for _ in range(R.AGENTS)]
    calls, summaries = [], []
    for rnd in range(R.ROUNDS):
        for i, agent_context in enumerate(agent_contexts):
            if rnd != 0:
                others = agent_contexts[:i] + agent_contexts[i + 1:]
                resp = R.other_responses(others, 2 * rnd - 1)
                if resp:
                    s = chat(R.summary_messages(resp), T_GEN, MAX_TOK, seed0 + call); call += 1
                    calls.append(dict(kind="summary", round=rnd, agent=i, input_tokens=s["input_tokens"],
                                      output_tokens=s["output_tokens"], finish=s["finish"][0]))
                    summary = s["texts"][0]
                else:
                    summary = "No other opinions available."
                summaries.append(dict(round=rnd, agent=i, text=summary))
                agent_context.append(R.debate_message(summary, d["question"]))
            c = chat(agent_context, T_GEN, MAX_TOK, seed0 + call); call += 1
            calls.append(dict(kind="agent", round=rnd, agent=i, input_tokens=c["input_tokens"],
                              output_tokens=c["output_tokens"], finish=c["finish"][0]))
            agent_context.append({"role": "assistant", "content": c["texts"][0]})
    return dict(agent_contexts=agent_contexts, summaries=summaries, calls=calls)


def single(d, run, prompt_fn, temperature):
    c = chat([{"role": "user", "content": prompt_fn(d["contexts"], d["question"])}], temperature, MAX_TOK, run * 10**6 + d["idx"] * 100)
    return dict(text=c["texts"][0], finish=c["finish"][0], calls=[dict(input_tokens=c["input_tokens"], output_tokens=c["output_tokens"])])


def sc(d, run, k):
    """k independent round-0 samples as k separate requests (n=1), so every sample pays its own prompt tokens,
    matching t_s and MedARC's per-call accounting (Amendment 2). Seeds: run·10⁶ + idx·100 + j."""
    texts, finish, calls = [], [], []
    for j in range(k):
        c = chat([{"role": "user", "content": R.initial_prompt(d["contexts"], d["question"])}], T_GEN, MAX_TOK,
                 run * 10**6 + d["idx"] * 100 + j)
        texts.append(c["texts"][0]); finish.append(c["finish"][0])
        calls.append(dict(input_tokens=c["input_tokens"], output_tokens=c["output_tokens"]))
    return dict(texts=texts, finish=finish, calls=calls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["medarc", "zeroshot", "cot", "sc"])
    ap.add_argument("--split", required=True, choices=["eval", "probe"])
    ap.add_argument("--run", type=int, required=True)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--n", type=int, default=-1)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    D = [json.loads(l) for l in open(f"{ROOT}/studies/5_medarc_transfer/results/data/{a.split}_inputs.jsonl")]
    D = D if a.n < 0 else D[:a.n]
    fn = dict(medarc=lambda d: medarc(d, a.run), zeroshot=lambda d: single(d, a.run, R.zeroshot_prompt, T_GEN),
              cot=lambda d: single(d, a.run, R.cot_prompt, 0.0), sc=lambda d: sc(d, a.run, a.k))[a.arm]

    def job(d):
        try:
            return dict(idx=d["idx"], pubid=d["pubid"], arm=a.arm, run=a.run, error=None, **fn(d))
        except Exception:
            return dict(idx=d["idx"], pubid=d["pubid"], arm=a.arm, run=a.run, error=traceback.format_exc())
    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(job, D))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tok = sum(c["input_tokens"] + c["output_tokens"] for r in rows if not r["error"] for c in r["calls"]) / max(1, len(rows))
    print(f"{a.arm} {a.split} run{a.run}: n={len(rows)} errors={sum(r['error'] is not None for r in rows)} tokens/q={tok:.0f}")


if __name__ == "__main__":
    main()
