"""§3.5 HeteroMore: MedAgents' released recruitment (5 + 2 domains, T=0, parsed as M1 parses it; tokens
charged), then k_het samples round-robin over the recruited roles (analyzer system prompt + solve user
prompt, T=0.7, 700 tok, seed 53e6 + qid·10 + role), E3 each, majority over probe then HET answers.

  python run_hetero_more.py --split dev --probe probe.jsonl --k K --label L --out het.jsonl [--max_triggered 20]
"""
import argparse
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor

from common import MODEL, ROOT, prompt, opts, e3, usage, add, load_split, jl, write_jl, sample_n_meta, chat_meta
import aggregate as AG
from run_tdc import TRIGGERED

sys.path.insert(0, os.path.join(ROOT, "external", "MedAgents"))
from prompt_generator import (get_question_domains_prompt, get_options_domains_prompt,   # noqa: E402
                              get_question_analysis_prompt, NUM_QD, NUM_OD)


def recruit(q, tag):
    us, doms = [], []
    for fn, args, k in ((get_question_domains_prompt, (q["question"],), NUM_QD),
                        (get_options_domains_prompt, (q["question"], opts(q)), NUM_OD)):
        system, pr = fn(*args)
        r = chat_meta([{"role": "system", "content": system}, {"role": "user", "content": pr}],
                      json_mode=False, model=MODEL, temperature=0, max_tokens=50, nonce=f"{tag}:recruit:{k}")
        us.append(usage(r.get("usage")))
        parsed = [d.strip() for d in (r["response"] or "").split(":")[-1].strip().split(" | ") if d.strip()]
        doms += parsed or ["General Medicine"] * k
    return doms, us


def hetero_more(q, p, k, label):
    tag = f"{label}:{q['qid']}"
    try:
        doms, rus = recruit(q, tag)
        per_role = -(-k // len(doms))
        solve_user = prompt("solve", question=q["question"], options=opts(q))[1]
        by_role, gus = [], []
        for j, d in enumerate(doms):
            if j >= k:                       # roles beyond k receive no sample (round-robin starts at role 0)
                by_role.append([]); continue
            system, _ = get_question_analysis_prompt(q["question"], d)
            texts, gen = sample_n_meta([{"role": "system", "content": system}, solve_user], per_role, model=MODEL,
                                       temperature=0.7, max_tokens=700, seed=53_000_000 + q["qid"] * 10 + j)
            by_role.append(texts); gus.append(usage(gen))
        order = [(s, j) for s in range(per_role) for j in range(len(doms)) if s < len(by_role[j])][:k]
        votes, eus = [], []
        for n_, (s, j) in enumerate(order):
            a, u = e3(by_role[j][s], q, f"{tag}:e3:{n_}")
            votes.append(a); eus.append(u)
        return dict(qid=q["qid"], state=p["state"], k_het=k, domains=doms, roles=[doms[j] for _, j in order],
                    votes=votes, final_answer=AG.majority(p["answers"] + votes),
                    recruit_tokens=add(*rus), sample_tokens=add(*gus, *eus), tokens=add(*rus, *gus, *eus), error=None)
    except Exception:
        return dict(qid=q["qid"], state=p["state"], final_answer="", error=traceback.format_exc(),
                    tokens=dict(input_tokens=0, output_tokens=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--max_triggered", type=int, default=-1)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Q = {q["qid"]: q for q in load_split(a.split)}
    P = [p for p in jl(a.probe) if p["state"] in TRIGGERED]
    if a.max_triggered > 0:
        P = P[:a.max_triggered]
    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(lambda p: hetero_more(Q[p["qid"]], p, a.k, a.label), P))
    write_jl(a.out, rows)
    print(f"HeteroMore {a.label}: n={len(rows)}, k={a.k}, errors={sum(r['error'] is not None for r in rows)}")


if __name__ == "__main__":
    main()
