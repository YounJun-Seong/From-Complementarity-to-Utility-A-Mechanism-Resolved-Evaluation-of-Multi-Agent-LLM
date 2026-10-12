"""§8 run checks (no labels): expected qids, no errors, required fields; prints implementation diagnostics.

  python validate_runs.py --dir studies/3_disagreement_intervention/results/<split>/... [--m1]
"""
import argparse
import glob
import json
import os
from collections import Counter

from common import jl
from run_tdc import TRIGGERED


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--m1", action="store_true")
    a = ap.parse_args()
    ok = True
    P = jl(f"{a.dir}/probe.jsonl")
    trig = [p["qid"] for p in P if p["state"] in TRIGGERED]
    perr = sum(p["error"] is not None for p in P)
    parse = sum(bool(x) for p in P for x in p["answers"]) / (4 * len(P))
    print(f"probe: n={len(P)} errors={perr} parse_success={parse:.3f} states={dict(Counter(p['state'] for p in P))}")
    ok &= perr == 0
    for arm in ("tdc", "sample_more", "het"):
        f = f"{a.dir}/{arm}.jsonl"
        if not os.path.exists(f):
            continue
        R = jl(f)
        qids = [r["qid"] for r in R]
        err = sum(r["error"] is not None for r in R)
        exp = trig if len(R) == len(trig) or arm == "sample_more" else trig[:len(R)]   # token probe: first 20 triggered
        same = qids == exp
        print(f"{arm}: n={len(R)} expected={len(exp)} qids_match={same} errors={err} "
              f"empty_final={sum(not r['final_answer'] for r in R)}")
        ok &= same and err == 0
        if arm == "tdc":
            D = [r for r in R if not r.get("single_answer") and r["error"] is None]
            if D:
                print(f"  extraction parsed_all={sum(r['disagreement']['parsed_all'] for r in D)}/{len(D)} "
                      f"core_fallback={sum(r['disagreement']['core_fallback'] for r in D)} "
                      f"claim_missing={sum(r['disagreement']['claim_x_missing'] or r['disagreement']['claim_y_missing'] for r in D)} "
                      f"revision_e3_failed={sum(r['revision_x']['e3_failed'] + r['revision_y']['e3_failed'] for r in D)}/{2*len(D)} "
                      f"single_answer={sum(bool(r.get('single_answer')) for r in R)}")
                print(f"  flow joint={dict(Counter(r['flow']['joint'] for r in D))}")
                print(f"  revision finish=length: {sum((r['revision_x']['finish_reason'] == 'length') + (r['revision_y']['finish_reason'] == 'length') for r in D)}/{2*len(D)}")
    if a.m1:
        qids = json.load(open(f"{a.dir}/m1_input/qids.json"))
        from string import punctuation
        inp = [json.loads(l) for l in open(f"{a.dir}/m1_input/test.jsonl")]
        assert len({r["answer_idx"] for r in inp}) == 1 and len({r["answer"] for r in inp}) == 1
        out = {}
        for f in glob.glob(f"{a.dir}/M1/shard*/chatgpt-syn_verif"):
            s_ = int(os.path.basename(os.path.dirname(f))[5:])
            for k, l in enumerate(open(f)):
                out[s_ * 20 + k] = json.loads(l)
        # MedAgents run.py appends '?' when a question lacks final punctuation (released behaviour)
        norm = lambda q: q if q[-1] in punctuation else q + "?"
        order = all(out[i]["question"] == norm(inp[i]["question"]) for i in range(len(inp)) if i in out)
        crashes = sum(r.get("error") is not None for r in out.values())
        print(f"M1: rows={len(out)} expected={len(qids)} order_ok={order} crashes={crashes} placeholder_constant=True")
        ok &= len(out) == len(qids) and order
    print("VALID" if ok else "INVALID")


if __name__ == "__main__":
    main()
