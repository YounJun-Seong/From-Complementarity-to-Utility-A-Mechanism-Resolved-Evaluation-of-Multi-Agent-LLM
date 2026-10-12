"""HET-k (PHASE2_FREEZE.md Amendment 1): role-heterogeneous, non-interactive sampling.

Per question: MedAgents' released expert recruitment (5 question domains + 2 option domains, T=0), then
samples of the S user prompt at T=0.7 / 700 tok under MedAgents' released analyzer system prompt for each
recruited domain, roles round-robin in recruitment order, `total` samples per run whatever the number of recruited roles; E3 per sample; majority vote (ties -> earliest).
No sample sees another's output.

  python studies/2_complementarity_router/code/hetero_sc.py --data <jsonl> --n 300 --total 21 --seed 201 --run_label L --out f.jsonl
"""
import argparse
import collections
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..", "..", "..")
sys.path.insert(0, os.path.join(ROOT, "common"))
sys.path.insert(0, os.path.join(ROOT, "studies", "1_released_reevaluation", "experiments"))
sys.path.insert(0, os.path.join(ROOT, "external", "MedAgents"))
from llm import chat_meta, sample_n_meta                       # noqa: E402
from p0_extraction import MODEL, opts                          # noqa: E402
from compute_matched import e3, usage_sum                      # noqa: E402
from prompt_generator import (get_question_domains_prompt, get_options_domains_prompt,   # noqa: E402
                              get_question_analysis_prompt, NUM_QD, NUM_OD)


def recruit(q, label, i):
    """MedAgents' expert recruitment, released prompts, parsed as in MedAgents utils.fully_decode."""
    us, doms = [], []
    for fn, args, k in ((get_question_domains_prompt, (q["question"],), NUM_QD),
                        (get_options_domains_prompt, (q["question"], opts(q)), NUM_OD)):
        system, prompt = fn(*args)
        r = chat_meta([{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                      json_mode=False, model=MODEL, temperature=0, max_tokens=50, nonce=f"{label}:{i}:recruit:{k}")
        us.append(r.get("usage") or {})
        raw = r["response"] or ""
        parsed = [d.strip() for d in raw.split(":")[-1].strip().split(" | ") if d.strip()]
        doms += parsed or ["General Medicine"] * k
    return doms, us


def run_het(q, i, total, label, seed):
    doms, us = recruit(q, label, i)
    per_role = -(-total // len(doms))          # ceil: enough samples per role for `total` round-robin draws
    user = f"{q['question']}\n{opts(q)}\n\nLet's think step by step."
    by_role, gen_p, gen_c = [], 0, 0
    for j, d in enumerate(doms):
        system, _ = get_question_analysis_prompt(q["question"], d)
        texts, gen = sample_n_meta([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                   per_role, model=MODEL, temperature=0.7, max_tokens=700, seed=seed * 100000 + i * 10 + j)
        by_role.append(texts)
        gen_p += gen.get("prompt_tokens", 0); gen_c += gen.get("completion_tokens", 0)
    # round-robin order: role 0 sample 0, role 1 sample 0, ..., role 0 sample 1, ...
    order = [(s, j) for s in range(per_role) for j in range(len(doms))][:total]
    texts = [by_role[j][s] for s, j in order]
    roles = [doms[j] for s, j in order]
    votes = []
    for s, t in enumerate(texts):
        a, u = e3(t, q, f"{label}:{i}:e3:{s}")
        votes.append(a); us.append(u)
    c = collections.Counter(v for v in votes if v)
    pred = next(v for v in votes if v and c[v] == c.most_common(1)[0][1]) if c else ""
    pi, po = usage_sum(us)
    return {"idx": i, "gold": q["answer_idx"], "pred": pred, "votes": votes, "roles": roles, "domains": doms,
            "calls": 2 + len(doms) + len(texts), "samples_generated": per_role * len(doms), "prompt_tokens": gen_p + pi, "completion_tokens": gen_c + po}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--total", type=int, default=21)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--run_label", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    QA = [json.loads(l) for l in open(a.data)][:a.n]
    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(lambda iq: run_het(iq[1], iq[0], a.total, a.run_label, a.seed), enumerate(QA)))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tok = sum(r["prompt_tokens"] + r["completion_tokens"] for r in rows) / len(rows)
    print(f"HET {a.run_label}: n={len(rows)}, samples/q {len(rows[0]['votes'])}, tokens/q {tok:.0f}")   # no accuracy printed


if __name__ == "__main__":
    main()
