"""§3.1–3.2 probe: 4 samples (one request, n=4, T=0.7, 700 tok, seed 51e6+qid), E3 each, disagreement state.

  python run_probe.py --split dev --start 0 --n 20 --label L --out studies/3_disagreement_intervention/results/.../probe.jsonl
"""
import argparse
import traceback
from concurrent.futures import ThreadPoolExecutor

from common import MODEL, prompt, opts, e3, usage, add, load_split, write_jl, sample_n_meta
import aggregate as AG


def probe(q, label):
    try:
        texts, gen = sample_n_meta(prompt("solve", question=q["question"], options=opts(q)), 4, model=MODEL,
                                   temperature=0.7, max_tokens=700, seed=51_000_000 + q["qid"])
        gen = usage(gen)
        samples, e3u = [], []
        for j, t in enumerate(texts):
            a, u = e3(t, q, f"{label}:{q['qid']}:e3:{j}")
            e3u.append(u)
            samples.append(dict(sample_id=j, answer=a, reasoning=t, e3_tokens=u))
        answers = [s["answer"] for s in samples]
        return dict(qid=q["qid"], probe=samples, answers=answers, state=AG.state(answers),
                    probe_gen_tokens=gen, probe_tokens=add(gen, *e3u), error=None)
    except Exception:
        return dict(qid=q["qid"], probe=[], answers=["", "", "", ""], state="failure", error=traceback.format_exc())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=-1)
    ap.add_argument("--label", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Q = load_split(a.split)[a.start:] if a.n < 0 else load_split(a.split)[a.start:a.start + a.n]
    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(lambda q: probe(q, a.label), Q))
    write_jl(a.out, rows)
    print(f"probe {a.label}: n={len(rows)}, errors={sum(r['error'] is not None for r in rows)}")


if __name__ == "__main__":
    main()
