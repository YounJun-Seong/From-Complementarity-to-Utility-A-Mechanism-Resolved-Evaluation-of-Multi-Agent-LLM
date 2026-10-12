"""§3.4 SampleMore: k_add more samples (one request, n=k_add, T=0.7, seed 52e6+qid), E3 each,
majority over probe answers then additional answers (ties -> earliest).

  python run_sample_more.py --split dev --probe probe.jsonl --k K --label L --out sm.jsonl
"""
import argparse
import traceback
from concurrent.futures import ThreadPoolExecutor

from common import MODEL, prompt, opts, e3, usage, add, load_split, jl, write_jl, sample_n_meta
import aggregate as AG
from run_tdc import TRIGGERED


def sample_more(q, p, k, label):
    try:
        texts, gen = sample_n_meta(prompt("solve", question=q["question"], options=opts(q)), k, model=MODEL,
                                   temperature=0.7, max_tokens=700, seed=52_000_000 + q["qid"])
        votes, us = [], [usage(gen)]
        for j, t in enumerate(texts):
            a, u = e3(t, q, f"{label}:{q['qid']}:e3:{j}")
            votes.append(a); us.append(u)
        return dict(qid=q["qid"], state=p["state"], k_additional=k, votes=votes,
                    final_answer=AG.majority(p["answers"] + votes), tokens=add(*us), error=None)
    except Exception:
        return dict(qid=q["qid"], state=p["state"], final_answer="", error=traceback.format_exc(),
                    tokens=dict(input_tokens=0, output_tokens=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Q = {q["qid"]: q for q in load_split(a.split)}
    P = [p for p in jl(a.probe) if p["state"] in TRIGGERED]
    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(lambda p: sample_more(Q[p["qid"]], p, a.k, a.label), P))
    write_jl(a.out, rows)
    print(f"SampleMore {a.label}: n={len(rows)}, k={a.k}, errors={sum(r['error'] is not None for r in rows)}")


if __name__ == "__main__":
    main()
