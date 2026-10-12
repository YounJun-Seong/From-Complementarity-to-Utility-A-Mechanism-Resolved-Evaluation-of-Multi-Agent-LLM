"""Phase IV implementation invariants on run outputs (no labels). Rebuilds every revision request from the
stored probe/extraction exactly as run_revisions.py does and compares it with the cached request that was
actually sent (cache/phase4), so leakage between conditions would show up as a mismatch.

  python studies/4_controlled_revision/code/check_invariants.py --dir studies/4_controlled_revision/results/dev_smoke --split dev
"""
import argparse
import glob
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "3_disagreement_intervention", "code"))
import run_revisions as RR     # noqa: E402
import common as C3            # noqa: E402
sys.path.insert(0, os.path.join(C3.ROOT, "common"))
import llm                     # noqa: E402


def cached(msgs, nonce):
    payload = {"model": os.environ.get("TMC_MODEL_OVERRIDE", C3.MODEL), "temperature": 0, "messages": msgs,
               "json_mode": False, "max_tokens": 700, "nonce": nonce}
    p = os.path.join(os.environ["TMC_CACHE"], llm._key(payload) + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--label", required=True)
    a = ap.parse_args()
    Q = {q["qid"]: q for q in C3.load_split(a.split)}
    P = {p["qid"]: p for p in C3.jl(f"{a.dir}/probe.jsonl")}
    R = C3.jl(f"{a.dir}/revisions.jsonl")
    E = {r["qid"]: r["extraction"] for r in R}
    fails, n = [], 0
    for r in R:
        q, p, e = Q[r["qid"]], P[r["qid"]], r["extraction"]
        other = E[r["irrelevant_source_qid"]]
        assert r["error"] is None, r["error"]
        order = sorted(E)
        if order[(order.index(r["qid"]) + 1) % len(order)] != r["irrelevant_source_qid"]:
            fails.append((r["qid"], "irrelevant source is not the next triggered qid"))
        for side, sid, own, opp, opp_irr in (("x", e["rep_x"], e["X"], e["claim_y"], other["claim_y"]),
                                             ("y", e["rep_y"], e["Y"], e["claim_x"], other["claim_x"])):
            rep = next(s for s in p["probe"] if s["sample_id"] == sid)
            if rep["answer"] != own:
                fails.append((r["qid"], side, "representative answer mismatch"))
            base = dict(question=q["question"], options=C3.opts(q), own_reasoning_full=rep["reasoning"], own=own)
            expect = dict(control=RR.prompt4("revise_control", **base), blinded=RR.prompt4("revise_blinded", **base),
                          opponent=RR.prompt4("targeted_revision", core_disagreement=e["core"], opposing_claim=opp, **base),
                          irrelevant=RR.prompt4("targeted_revision", core_disagreement=other["core"], opposing_claim=opp_irr, **base))
            for cond, msgs in expect.items():
                n += 1
                rec = cached(msgs, f"{a.label}:{r['qid']}:{cond}:{side}")
                if rec is None:
                    fails.append((r["qid"], side, cond, "sent request differs from the frozen construction")); continue
                if rec["response"] != r["revisions"][cond][side]["text"]:
                    fails.append((r["qid"], side, cond, "stored text != cached response"))
                body = json.dumps(msgs)
                if cond in ("control", "blinded") and (e["core"] in body or opp in body):
                    fails.append((r["qid"], side, cond, "opponent information leaked"))
                if cond == "opponent" and other["core"] in body and other["qid"] != r["qid"]:
                    fails.append((r["qid"], side, cond, "irrelevant content in opponent"))
                if cond == "irrelevant" and (e["core"] in body and e["core"] != other["core"]):
                    fails.append((r["qid"], side, cond, "own CORE in irrelevant"))
                if msgs[2]["content"] != rep["reasoning"] or msgs[2]["role"] != "assistant":
                    fails.append((r["qid"], side, cond, "context is not the original representative turn"))
                rv = r["revisions"][cond][side]
                if rv["before"] != own or not rv["after"] or rv["tokens"]["input_tokens"] <= 0:
                    fails.append((r["qid"], side, cond, "before/after/token fields"))
                if rv["finish_reason"] == "length":
                    fails.append((r["qid"], side, cond, "truncated"))
    kw = dict(question="Q?", options="A. a", own_reasoning_full="R", own="A", core_disagreement="c", opposing_claim="o")
    eq = RR.prompt4("targeted_revision", **kw) == C3.prompt("targeted_revision", **kw)
    print(f"revision requests checked: {n}; Phase III prompt equivalence: {eq}; failures: {len(fails)}")
    for f in fails[:20]:
        print("  ", f)
    print("INVARIANTS PASS" if eq and not fails else "INVARIANTS FAIL")


if __name__ == "__main__":
    main()
