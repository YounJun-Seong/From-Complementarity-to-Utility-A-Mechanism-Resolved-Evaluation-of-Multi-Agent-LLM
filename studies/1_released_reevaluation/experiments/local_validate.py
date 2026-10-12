"""Local-backend validation (run while a vLLM server is up). Checks, on the cache records of the
validation runs (identified by their nonce / run labels), that the local backend behaves like
the API backend in everything except the model:

  1. every call resolved and was served by the local model (incl. all MedAgents sub-agents)
  2. max_tokens honoured: completion_tokens <= max_tokens; finish_reason=length only at the cap
  3. finish_reason and token usage recorded for every call
  4. no chat-template / EOS leakage in outputs (<|im_end|>, <|im_start|>, <|endoftext|>)
  5. system role reaches the model (vLLM /tokenize with a system+user message)
  6. temperature 0 determinism (same S questions re-run under a second label)
  7. extractors parse (S: E1 / E3; MedAgents: non-empty pred_answer)

  python experiments/local_validate.py --model Qwen/Qwen2.5-7B-Instruct --url http://127.0.0.1:8011
"""
import argparse
import glob
import json
import os

import requests

ROOT = os.path.join(os.path.dirname(__file__), "..")
LEAK = ("<|im_end|>", "<|im_start|>", "<|endoftext|>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--url", required=True)
    ap.add_argument("--s_runs", default="runs/local_val/S_a.jsonl,runs/local_val/S_b.jsonl")
    ap.add_argument("--m1_out", default="runs/local_val/m1/chatgpt-syn_verif")
    a = ap.parse_args()
    ok = True

    recs = []
    for f in glob.glob(f"{ROOT}/data/llm_cache/*.json"):
        s = open(f).read()
        if f'"resolved_model": "{a.model}"' in s or f'"model": "{a.model}"' in s:
            recs.append(json.loads(s))
    print(f"cache records for {a.model}: {len(recs)}")

    bad_model = [r for r in recs if r.get("resolved_model") != a.model or (r.get("served_model") or a.model) != a.model]
    req = sorted({r.get("requested_model") for r in recs})
    print(f"[1] requested models {req} -> all resolved/served as local: {not bad_model} ({len(bad_model)} bad)")
    ok &= not bad_model

    over = [r for r in recs if r["request"].get("max_tokens") and r["usage"]["completion_tokens"] > r["request"]["max_tokens"]]
    len_wrong = [r for r in recs if r.get("finish_reason") == "length" and r["request"].get("max_tokens")
                 and r["usage"]["completion_tokens"] < r["request"]["max_tokens"]]
    n_len = sum(r.get("finish_reason") == "length" for r in recs)
    print(f"[2] completion > max_tokens: {len(over)}; length-stop below cap: {len(len_wrong)}; length stops: {n_len}")
    ok &= not over and not len_wrong

    missing = [r for r in recs if r.get("finish_reason") is None or not r.get("usage")]
    print(f"[3] records missing finish_reason/usage: {len(missing)}")
    ok &= not missing

    leak = [r for r in recs if any(t in (r.get("response") or "") for t in LEAK)]
    print(f"[4] outputs with template/EOS tokens: {len(leak)}")
    ok &= not leak

    msg = [{"role": "system", "content": "SYSTEM-MARKER-7731"}, {"role": "user", "content": "hi"}]
    t = requests.post(f"{a.url}/tokenize", json={"model": a.model, "messages": msg}, timeout=30).json()
    detok = requests.post(f"{a.url}/detokenize", json={"model": a.model, "tokens": t["tokens"]}, timeout=30).json()
    sys_ok = "SYSTEM-MARKER-7731" in detok.get("prompt", "")
    print(f"[5] system role in rendered prompt: {sys_ok}  | {detok.get('prompt', '')[:160]!r}")
    ok &= sys_ok

    fa, fb = [f"{ROOT}/{p}" for p in a.s_runs.split(",")]
    A = [json.loads(l) for l in open(fa)]
    B = [json.loads(l) for l in open(fb)]
    same_text = sum(x["reasoning"] == y["reasoning"] for x, y in zip(A, B))
    same_ans = sum(x["E3"] == y["E3"] for x, y in zip(A, B))
    print(f"[6] T=0 re-run: identical reasoning {same_text}/{len(A)}, identical E3 answer {same_ans}/{len(A)}")

    e1 = sum(r["E1"] != "" for r in A); e3 = sum(r["E3"] != "" for r in A)
    M = [json.loads(l) for l in open(f"{ROOT}/{a.m1_out}")]
    m_ok = sum(bool(r.get("pred_answer")) for r in M)
    print(f"[7] S: E1 parsed {e1}/{len(A)}, E3 parsed {e3}/{len(A)}, acc(E3) {sum(r['E3'] == r['gold'] for r in A)}/{len(A)}; "
          f"M1: pred_answer non-empty {m_ok}/{len(M)}, acc {sum(r['pred_answer'] == r['gold_answer'] for r in M)}/{len(M)}")
    ok &= e3 >= 0.9 * len(A) and m_ok >= 0.9 * len(M)
    print("VALIDATION", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
