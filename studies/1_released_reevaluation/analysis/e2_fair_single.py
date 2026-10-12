"""E2 — fair, PRE-REGISTERED standard single-agent prompts (no per-dataset tuning),
to rebut the 'cherry-picked single prompt' critique. Canonical prompts only:
  - direct            : zero-shot direct answer
  - zeroshot_cot      : "Let's think step by step" then extract (Kojima et al. 2022)
  - fiveshot          : 5-shot in-context (demos from held-out questions)
Same robust answer extraction for all. Best-of-standard is compared to the paper's
base_cot and the multi-agent system. Prompts fixed BEFORE seeing results.
"""
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from llm import chat

MODEL = "gpt-4o-mini-2024-07-18"
QA = []                                   # populated in main() from --data
DEMOS = []                                # held-out 5-shot demos (set in main)


def opts(q):
    return "\n".join(f"{k}. {v}" for k, v in q["options"].items())


def jchoice(t):
    try:
        return str(json.loads(t).get("choice", "")).strip()[:1].upper()
    except Exception:
        m = re.search(r'"choice"\s*:\s*"?([ABCD])', t)
        return m.group(1) if m else ""


def direct(q):
    t = chat([{"role": "system", "content": 'You are an expert physician. Respond ONLY as JSON: {"choice":"A|B|C|D"}'},
              {"role": "user", "content": f"{q['question']}\n{opts(q)}"}],
             json_mode=True, model=MODEL, max_tokens=20)
    return jchoice(t)


def zeroshot_cot(q):
    # step 1: free CoT (Kojima "Let's think step by step")
    r = chat([{"role": "system", "content": "You are an expert physician."},
              {"role": "user", "content": f"{q['question']}\n{opts(q)}\n\nLet's think step by step."}],
             json_mode=False, model=MODEL, max_tokens=700)
    # step 2: extract final answer
    t = chat([{"role": "system", "content": 'Given the reasoning, output the final answer. Respond ONLY as JSON: {"choice":"A|B|C|D"}'},
              {"role": "user", "content": f"Reasoning:\n{r[:2500]}\n\nOptions:\n{opts(q)}"}],
             json_mode=True, model=MODEL, max_tokens=20)
    return jchoice(t)


def fiveshot(q):
    demo = ""
    for d in DEMOS:
        demo += f"{d['question']}\n{opts(d)}\nAnswer: {d['answer_idx']}\n\n"
    t = chat([{"role": "system", "content": 'You are an expert physician. Respond ONLY as JSON: {"choice":"A|B|C|D"}'},
              {"role": "user", "content": demo + f"{q['question']}\n{opts(q)}"}],
             json_mode=True, model=MODEL, max_tokens=20)
    return jchoice(t)


PROMPTS = {"direct": direct, "zeroshot_cot": zeroshot_cot, "fiveshot": fiveshot}


def main():
    global QA, DEMOS, MODEL
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--data", default="external/MedAgents/datasets/MedQA/test.jsonl")
    ap.add_argument("--out", default="data/e2_fair_single.csv")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--prompts", default="direct,zeroshot_cot,fiveshot",
                    help="comma list; use 'zeroshot_cot' alone for full-test runs (no demo leak)")
    ap.add_argument("--workers", type=int, default=8,
                    help="concurrency; lower (e.g. 2) for low-TPM models like gpt-4.1")
    args = ap.parse_args()
    MODEL = args.model
    QA = [json.loads(l) for l in open(args.data)]
    DEMOS = QA[500:505]                    # held-out 5-shot demos (no overlap with first 200)
    Q = QA[:args.n]
    want = [p for p in args.prompts.split(",") if p in PROMPTS]
    df = pd.DataFrame({"gt": [q["answer_idx"] for q in Q]})
    for name in want:
        fn = PROMPTS[name]
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            preds = list(ex.map(fn, Q))
        df[name] = [int(p == q["answer_idx"]) for p, q in zip(preds, Q)]
        print(f"  {name:14s} acc={df[name].mean():.3f}", flush=True)
    df.to_csv(args.out, index=False)
    best = max(want, key=lambda c: df[c].mean())
    print(f"E2_DONE best standard single = {best} ({df[best].mean():.3f}) "
          f"| base_cot 0.667, multi-agent 0.807")


if __name__ == "__main__":
    main()
