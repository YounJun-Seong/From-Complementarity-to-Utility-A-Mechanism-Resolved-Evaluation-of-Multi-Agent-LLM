"""§3.3 TDC on triggered probe rows: top-2, representatives, disagreement extraction (T=0, 300 tok),
one targeted revision per representative (T=0, 700 tok, E3 on the last 2,500 chars), frozen aggregation.

  python run_tdc.py --split dev --probe probe.jsonl --label L --out tdc.jsonl [--max_triggered 20]
"""
import argparse
import re
import traceback
from concurrent.futures import ThreadPoolExecutor

from common import MODEL, prompt, opts, e3, usage, add, load_split, jl, write_jl, chat_meta
import aggregate as AG

TRIGGERED = ("weak", "split", "fragmented")


def field(text, name):
    for line in (text or "").splitlines():
        m = re.match(rf"^\W*{name}\W*:\s*(.*?)\s*\**\s*$", line, flags=re.I)
        if m and m.group(1).strip():
            return m.group(1).strip()
    return None


def tdc(q, p, label):
    tag = f"{label}:{q['qid']}"
    answers = p["answers"]
    x, y = AG.top2(answers)
    out = dict(qid=q["qid"], state=p["state"], X=x, Y=y, error=None)
    if y is None:                                   # one distinct parsed answer: no opposing answer, no calls
        out.update(final_answer=x, single_answer=True, tokens=dict(input_tokens=0, output_tokens=0))
        return out
    try:
        rx = next(s for s in p["probe"] if s["answer"] == x)
        ry = next(s for s in p["probe"] if s["answer"] == y)
        ex = chat_meta(prompt("extract_disagreement", question=q["question"], options=opts(q), X=x, Y=y,
                              reasoning_X_first_2500_chars=rx["reasoning"][:2500],
                              reasoning_Y_first_2500_chars=ry["reasoning"][:2500]),
                       json_mode=False, model=MODEL, temperature=0, max_tokens=300, nonce=f"{tag}:extract")
        raw = ex["response"] or ""
        cx, cy, core = field(raw, f"CLAIM_{x}"), field(raw, f"CLAIM_{y}"), field(raw, "CORE_DISAGREEMENT")
        diag = dict(parsed_all=all((cx, cy, core)), core_fallback=core is None,
                    claim_x_missing=cx is None, claim_y_missing=cy is None)
        core = core or raw.strip()[:1000]
        cx, cy = cx or "(not extracted)", cy or "(not extracted)"
        toks = [usage(ex.get("usage"))]
        rev = {}
        for side, rep, own, opp in (("x", rx, x, cy), ("y", ry, y, cx)):
            r = chat_meta(prompt("targeted_revision", question=q["question"], options=opts(q),
                                 own_reasoning_full=rep["reasoning"], own=own, core_disagreement=core,
                                 opposing_claim=opp),
                          json_mode=False, model=MODEL, temperature=0, max_tokens=700, nonce=f"{tag}:revise:{side}")
            toks.append(usage(r.get("usage")))
            a, u = e3(r["response"], q, f"{tag}:revise:{side}:e3", tail=True)
            toks.append(u)
            rev[side] = dict(representative=rep["sample_id"], before=own, after=a or own, e3_failed=not a,
                             text=r["response"], finish_reason=r.get("finish_reason"))
        fx, fy, joint = AG.flow(x, y, rev["x"]["after"], rev["y"]["after"])
        out.update(disagreement=dict(claim_x=cx, claim_y=cy, core=core, raw=raw, **diag),
                   revision_x=rev["x"], revision_y=rev["y"], flow=dict(x=fx, y=fy, joint=joint),
                   final_answer=AG.tdc_final(answers, rev["x"]["after"], rev["y"]["after"]),
                   single_answer=False, tokens=add(*toks))
    except Exception:
        out.update(final_answer="", error=traceback.format_exc(), tokens=dict(input_tokens=0, output_tokens=0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--probe", required=True)
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
        rows = list(ex.map(lambda p: tdc(Q[p["qid"]], p, a.label), P))
    write_jl(a.out, rows)
    print(f"TDC {a.label}: n={len(rows)}, errors={sum(r['error'] is not None for r in rows)}")


if __name__ == "__main__":
    main()
