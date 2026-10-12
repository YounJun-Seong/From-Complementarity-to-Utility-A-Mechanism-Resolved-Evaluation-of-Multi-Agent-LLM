"""Phase IV §2: disagreement extraction (Phase III, unchanged) and four independent revision conditions per
representative. Every call is a separate stateless request built from the identical original context.

Irrelevant condition: CORE and opposing claim of the next triggered question (qid order, cyclic); rep(X)
receives that question's claim for its X-side opponent (its CLAIM_Y), rep(Y) its CLAIM_X.

  python run_revisions.py --split final --probe probe.jsonl --label L --out revisions.jsonl
"""
import argparse
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "3_disagreement_intervention", "code"))
import common as C3                       # noqa: E402
import aggregate as AG                    # noqa: E402
from run_tdc import field, TRIGGERED      # noqa: E402

P4 = os.path.join(HERE, "prompts")
CONDITIONS = ("control", "blinded", "opponent", "irrelevant")


def prompt4(name, **kw):
    """Render a frozen prompt (Phase IV prompts in phase4/prompts; unchanged Phase III prompts from
    phase3/prompts) exactly as Phase III common.prompt does. No shared state is modified (thread-safe)."""
    path = f"{P4}/{name}.txt" if os.path.exists(f"{P4}/{name}.txt") else f"{C3.HERE}/prompts/{name}.txt"
    msgs, role, buf = [], None, []
    for line in open(path).read().split("\n"):
        if line.startswith("### "):
            if role:
                msgs.append({"role": role, "content": "\n".join(buf).strip("\n")})
            role, buf = line[4:].strip(), []
        else:
            buf.append(line)
    msgs.append({"role": role, "content": "\n".join(buf).strip("\n")})
    for m in msgs:
        for k, v in kw.items():
            m["content"] = m["content"].replace("{" + k + "}", str(v))
    return msgs


def extract(q, p, label):
    x, y = AG.top2(p["answers"])
    rx = next(s for s in p["probe"] if s["answer"] == x)
    ry = next(s for s in p["probe"] if s["answer"] == y)
    ex = C3.chat_meta(prompt4("extract_disagreement", question=q["question"], options=C3.opts(q), X=x, Y=y,
                              reasoning_X_first_2500_chars=rx["reasoning"][:2500],
                              reasoning_Y_first_2500_chars=ry["reasoning"][:2500]),
                      json_mode=False, model=C3.MODEL, temperature=0, max_tokens=300, nonce=f"{label}:{q['qid']}:extract")
    raw = ex["response"] or ""
    cx, cy, core = field(raw, f"CLAIM_{x}"), field(raw, f"CLAIM_{y}"), field(raw, "CORE_DISAGREEMENT")
    return dict(qid=q["qid"], X=x, Y=y, rep_x=rx["sample_id"], rep_y=ry["sample_id"], raw=raw,
                parsed_all=all((cx, cy, core)), core_fallback=core is None,
                claim_x=cx or "(not extracted)", claim_y=cy or "(not extracted)", core=core or raw.strip()[:1000],
                tokens=C3.usage(ex.get("usage")))


def revise(q, p, e, other, label):
    tag = f"{label}:{q['qid']}"
    reps = dict(x=(next(s for s in p["probe"] if s["sample_id"] == e["rep_x"]), e["X"], e["claim_y"], other["claim_y"]),
                y=(next(s for s in p["probe"] if s["sample_id"] == e["rep_y"]), e["Y"], e["claim_x"], other["claim_x"]))
    out = {}
    for cond in CONDITIONS:
        out[cond] = {}
        for side, (rep, own, opp, opp_irrel) in reps.items():
            base = dict(question=q["question"], options=C3.opts(q), own_reasoning_full=rep["reasoning"], own=own)
            if cond == "control":
                msgs = prompt4("revise_control", **base)
            elif cond == "blinded":
                msgs = prompt4("revise_blinded", **base)
            elif cond == "opponent":
                msgs = prompt4("targeted_revision", core_disagreement=e["core"], opposing_claim=opp, **base)
            else:
                msgs = prompt4("targeted_revision", core_disagreement=other["core"], opposing_claim=opp_irrel, **base)
            r = C3.chat_meta(msgs, json_mode=False, model=C3.MODEL, temperature=0, max_tokens=700,
                             nonce=f"{tag}:{cond}:{side}")
            a, u = C3.e3(r["response"], q, f"{tag}:{cond}:{side}:e3", tail=True)
            out[cond][side] = dict(before=own, after=a or own, e3_failed=not a, text=r["response"],
                                   finish_reason=r.get("finish_reason"), tokens=C3.add(C3.usage(r.get("usage")), u))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Q = {q["qid"]: q for q in C3.load_split(a.split)}
    P = sorted([p for p in C3.jl(a.probe) if p["state"] in TRIGGERED and AG.top2(p["answers"])[1] is not None],
               key=lambda p: p["qid"])
    with ThreadPoolExecutor(a.workers) as ex:
        E = list(ex.map(lambda p: extract(Q[p["qid"]], p, a.label), P))
    def job(i):
        p, e, other = P[i], E[i], E[(i + 1) % len(E)]
        try:
            return dict(qid=p["qid"], state=p["state"], extraction=e, irrelevant_source_qid=other["qid"],
                        revisions=revise(Q[p["qid"]], p, e, other, a.label), error=None)
        except Exception:
            return dict(qid=p["qid"], state=p["state"], extraction=e, error=traceback.format_exc())
    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(job, range(len(P))))
    C3.write_jl(a.out, rows)
    print(f"revisions {a.label}: n={len(rows)} (triggered with two parsed answers), "
          f"errors={sum(r['error'] is not None for r in rows)}")


if __name__ == "__main__":
    main()
