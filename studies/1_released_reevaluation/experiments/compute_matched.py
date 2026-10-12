"""Layer 2D — compute-matched single-agent methods (local backend). MedQA 5-option.

  sc-<k>   : k reasoning samples of the S prompt at T=0.7 (one request, n=k), each answer extracted
             with the paper extractor E3 (T=0), majority vote (ties -> earliest sample's answer)
  reflect  : S reasoning (T=0, 700) -> self-critique (T=0, 700) -> revised reasoning (T=0, 700) -> E3

  python experiments/compute_matched.py --method sc-20 --n 200 --run_label 2D:<model>:sc-20:run1 --out f.jsonl
"""
import argparse
import collections
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.dirname(__file__))
from llm import chat_meta, sample_n_meta     # noqa: E402
from p0_extraction import MODEL, DATA, opts, jchoice   # noqa: E402

SYS = "You are an expert physician."
E3_SYS = 'Given the reasoning, output the final answer. Respond ONLY as JSON: {"choice":"A|B|C|D"}'
CRITIQUE = ("Review your reasoning above step by step. Identify any factual errors, missed findings, "
            "or flawed logic. Do not give a final answer yet.")
REVISE = ("Taking your review into account, reason through the question again and state your final answer.")


def e3(reasoning, q, tag):
    r = chat_meta([{"role": "system", "content": E3_SYS},
                   {"role": "user", "content": f"Reasoning:\n{reasoning[:2500]}\n\nOptions:\n{opts(q)}"}],
                  json_mode=True, model=MODEL, max_tokens=20, nonce=tag)
    return jchoice(r["response"]), r.get("usage") or {}


def usage_sum(us):
    return sum(u.get("prompt_tokens", 0) for u in us), sum(u.get("completion_tokens", 0) for u in us)


def run_sc(q, i, k, label, seed):
    msgs = [{"role": "system", "content": SYS},
            {"role": "user", "content": f"{q['question']}\n{opts(q)}\n\nLet's think step by step."}]
    texts, gen = sample_n_meta(msgs, k, model=MODEL, temperature=0.7, max_tokens=700, seed=seed)
    votes, us = [], []
    for j, t in enumerate(texts):
        a, u = e3(t, q, f"{label}:{i}:e3:{j}")
        votes.append(a); us.append(u)
    c = collections.Counter(v for v in votes if v)
    if c:
        top = c.most_common(1)[0][1]
        pred = next(v for v in votes if v and c[v] == top)
    else:
        pred = ""
    pi, po = usage_sum(us)
    return {"idx": i, "gold": q["answer_idx"], "pred": pred, "votes": votes,
            "vote_share": (c[pred] / len(votes)) if pred else 0.0, "calls": 1 + k,
            "prompt_tokens": gen.get("prompt_tokens", 0) + pi, "completion_tokens": gen.get("completion_tokens", 0) + po}


def run_reflect(q, i, label):
    msgs = [{"role": "system", "content": SYS},
            {"role": "user", "content": f"{q['question']}\n{opts(q)}\n\nLet's think step by step."}]
    us = []
    r1 = chat_meta(msgs, json_mode=False, model=MODEL, max_tokens=700, nonce=f"{label}:{i}:1"); us.append(r1.get("usage") or {})
    msgs += [{"role": "assistant", "content": r1["response"]}, {"role": "user", "content": CRITIQUE}]
    r2 = chat_meta(msgs, json_mode=False, model=MODEL, max_tokens=700, nonce=f"{label}:{i}:2"); us.append(r2.get("usage") or {})
    msgs += [{"role": "assistant", "content": r2["response"]}, {"role": "user", "content": REVISE}]
    r3 = chat_meta(msgs, json_mode=False, model=MODEL, max_tokens=700, nonce=f"{label}:{i}:3"); us.append(r3.get("usage") or {})
    pred, u = e3(r3["response"], q, f"{label}:{i}:e3"); us.append(u)
    first, u0 = e3(r1["response"], q, f"{label}:{i}:e3draft"); us.append(u0)
    pi, po = usage_sum(us)
    return {"idx": i, "gold": q["answer_idx"], "pred": pred, "draft_pred": first, "calls": 5,
            "finish": [r1.get("finish_reason"), r2.get("finish_reason"), r3.get("finish_reason")],
            "prompt_tokens": pi, "completion_tokens": po}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)          # sc-<k> | reflect
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--run_label", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    QA = [json.loads(l) for l in open(a.data)][:a.n]
    if a.method.startswith("sc-"):
        k = int(a.method[3:])
        fn = lambda iq: run_sc(iq[1], iq[0], k, a.run_label, a.seed * 100000 + iq[0])
    else:
        fn = lambda iq: run_reflect(iq[1], iq[0], a.run_label)
    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(fn, enumerate(QA)))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    acc = sum(r["pred"] == r["gold"] for r in rows) / len(rows)
    tok = sum(r["prompt_tokens"] + r["completion_tokens"] for r in rows) / len(rows)
    print(f"{a.method} {a.run_label}: acc {acc:.3f}, tokens/q {tok:.0f}, n={len(rows)}")


if __name__ == "__main__":
    main()
