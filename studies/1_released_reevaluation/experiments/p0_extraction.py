"""P0-5: extraction audit for the paper's primary single agent (zero-shot CoT with a
second LLM extraction call, analysis/e2_fair_single.py::zeroshot_cot).

Regenerates the step-1 reasoning (identical prompt / 700-token cap) and scores it with
  E1  deterministic regex over the conclusion (explicit > bold choice > option text)
  E2  strict structured output (JSON schema, enum A-E) on the same reasoning
  E3  the paper's extractor, verbatim ({"choice":"A|B|C|D"}, reasoning[:2500])
  E3f the paper's extractor with the option list fixed to A-E and no 2500-char cut
and reports answer-flip rates of each LLM extractor against E1's explicit answer.

  python experiments/p0_extraction.py
"""
import argparse
import hashlib
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from llm import chat_meta                                    # noqa: E402

MODEL = "gpt-4o-mini-2024-07-18"
DATA = os.path.join(ROOT, "external/MedAgents/datasets/MedQA/test.jsonl")
SCHEMA = {"type": "object", "properties": {"choice": {"type": "string", "enum": list("ABCDE")}},
          "required": ["choice"], "additionalProperties": False}
# E1 looks only at the conclusion (last TAIL chars) and takes the LAST hit of the first tier
# that matches: explicit "answer is X"/"Answer: X", then a bolded "**X." choice, then
# (no letter at all) the option whose text is mentioned last.
TAIL = 800
EXPLICIT = r"answer\s*(?:is|:)\s*[:\-]?\s*\**\s*\(?([A-E])\b(?![a-z'])"
BOLD = r"\*\*\s*\(?([A-E])[.):]\s|\(([A-E])\)\s*\*\*"


def opts(q):
    return "\n".join(f"{k}. {v}" for k, v in q["options"].items())


def regex_answer(text, q):
    tail = text[-TAIL:]
    m = re.findall(EXPLICIT, tail, flags=re.I)
    if m:
        return m[-1].upper(), "explicit"
    m = [a or b for a, b in re.findall(BOLD, tail)]
    if m:
        return m[-1], "bold"
    low = tail.lower()
    pos = {k: low.rfind(v.lower().rstrip(".")) for k, v in q["options"].items()}
    k, p = max(pos.items(), key=lambda kv: kv[1])
    return (k, "option_text") if p >= 0 else ("", "none")


def jchoice(t):  # verbatim from analysis/e2_fair_single.py
    try:
        return str(json.loads(t).get("choice", "")).strip()[:1].upper()
    except Exception:
        m = re.search(r'"choice"\s*:\s*"?([ABCD])', t)
        return m.group(1) if m else ""


RUN = None   # --run_label: enters every cache nonce so repeated T=0 runs are fresh API calls


def one(q):
    kw = {"nonce": f"{RUN}:{hashlib.sha1(q['question'].encode()).hexdigest()[:12]}"} if RUN else {}
    r = chat_meta([{"role": "system", "content": "You are an expert physician."},
                   {"role": "user", "content": f"{q['question']}\n{opts(q)}\n\nLet's think step by step."}],
                  json_mode=False, model=MODEL, max_tokens=700, **kw)
    reasoning = r["response"]
    e3r = chat_meta([{"role": "system", "content": 'Given the reasoning, output the final answer. Respond ONLY as JSON: {"choice":"A|B|C|D"}'},
                    {"role": "user", "content": f"Reasoning:\n{reasoning[:2500]}\n\nOptions:\n{opts(q)}"}],
                   json_mode=True, model=MODEL, max_tokens=20, **kw)
    e3 = e3r["response"]
    e3f = chat_meta([{"role": "system", "content": 'Given the reasoning, output the final answer. Respond ONLY as JSON: {"choice":"A|B|C|D|E"}'},
                     {"role": "user", "content": f"Reasoning:\n{reasoning}\n\nOptions:\n{opts(q)}"}],
                    json_mode=True, model=MODEL, max_tokens=20, **kw)["response"]
    e2 = chat_meta([{"role": "system", "content": "Given the reasoning, output the final answer it arrives at."},
                    {"role": "user", "content": f"Reasoning:\n{reasoning}\n\nOptions:\n{opts(q)}"}],
                   json_mode=True, model=MODEL, max_tokens=20, schema=SCHEMA, **kw)["response"]
    u1, u3 = r.get("usage") or {}, e3r.get("usage") or {}   # S = reasoning call + paper extractor (E3)
    return {"gold": q["answer_idx"], "finish_reason": r["finish_reason"],
            "S_prompt_tokens": u1.get("prompt_tokens", 0) + u3.get("prompt_tokens", 0),
            "S_completion_tokens": u1.get("completion_tokens", 0) + u3.get("completion_tokens", 0),
            "reasoning_chars": len(reasoning), **dict(zip(("E1", "E1_src"), regex_answer(reasoning, q))),
            "E2": jchoice(e2), "E3": jchoice(e3), "E3f": jchoice(e3f), "reasoning": reasoning}


def flips(df, col):
    k = df["E1"] != ""
    d = df[k]
    changed = d[col] != d["E1"]
    w2c = changed & (d["E1"] != d["gold"]) & (d[col] == d["gold"])
    c2w = changed & (d["E1"] == d["gold"]) & (d[col] != d["gold"])
    return changed.mean(), w2c.mean(), c2w.mean(), k.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=-1)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", default=os.path.join(ROOT, "runs/p0_extraction.jsonl"))
    ap.add_argument("--data", default=DATA, help="MedQA jsonl (5-option default; MedLA uses 4-option)")
    ap.add_argument("--run_label", default=None, help="cache namespace for repeated runs (e.g. s4-run2)")
    a = ap.parse_args()
    global RUN
    RUN = a.run_label
    QA = [json.loads(l) for l in open(a.data)]
    QA = QA if a.n < 0 else QA[:a.n]
    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(one, QA))
    with open(a.out, "w") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({"idx": i, **r}) + "\n")
    df = pd.DataFrame(rows)
    n = len(df)
    print(f"zero-shot CoT reasoning, n={n}: finish=length {np.mean(df['finish_reason'] == 'length'):.3f}, "
          f"E1 regex finds an explicit answer in {np.mean(df['E1'] != ''):.3f}, "
          f"reasoning >2500 chars {np.mean(df['reasoning_chars'] > 2500):.3f}")
    print("E1 source:", df["E1_src"].value_counts().to_dict())
    print(f"\n{'extractor':6s} {'acc':>6s} {'predE':>6s}")
    for c in ("E1", "E2", "E3", "E3f"):
        print(f"{c:6s} {np.mean(df[c] == df['gold']):6.3f} {np.mean(df[c] == 'E'):6.3f}")
    print(f"\nflips vs E1's explicit answer (on the {int((df['E1'] != '').sum())} questions where E1 parses):")
    print(f"{'extractor':9s} {'changed':>8s} {'wrong->right':>13s} {'right->wrong':>13s}")
    for c in ("E2", "E3", "E3f"):
        ch, w2c, c2w, _ = flips(df, c)
        print(f"{c:9s} {ch:8.3%} {w2c:13.3%} {c2w:13.3%}")
    ref = os.path.join(ROOT, "data/e2_full_single.csv")
    if os.path.exists(ref) and n == 1273 and a.data == DATA:
        old = pd.read_csv(ref)["zeroshot_cot"].to_numpy()
        new = (df["E3"] == df["gold"]).to_numpy().astype(int)
        print(f"\nE3 rerun vs supplement zeroshot_cot correctness: agree {np.mean(old == new):.3f} "
              f"(acc {old.mean():.3f} -> {new.mean():.3f})")


if __name__ == "__main__":
    main()
