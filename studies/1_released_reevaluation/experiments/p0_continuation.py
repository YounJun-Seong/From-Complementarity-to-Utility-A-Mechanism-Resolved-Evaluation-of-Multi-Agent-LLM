"""P0-4: counterfactual continuation. Keep the exact short-budget reasoning that failed to
complete, and only let the model finish it:

  A  fresh generation at the long budget              (from the p0 sweep)
  B  same context + partial output, "continue and give the final answer" (+400 tok)
  C  same context + partial output, "give only the final answer"          (+30 tok)

If B/C recover about as much as A, the short-budget failure is output truncation,
not missing reasoning.

  python experiments/p0_continuation.py --role empty --short 300 --long 700
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "external", "MedAgents"))
from llm import chat_meta                                    # noqa: E402
from prompt_generator import get_cot_prompt                  # noqa: E402
from data_utils import cleansing_final_output                # noqa: E402
from p0_sweep import MODEL, ROLES, DATA                      # noqa: E402
from p0_truncation import category, load                    # noqa: E402
from string import punctuation                               # noqa: E402

FOLLOW = {"B": ("Continue from exactly where you stopped and provide the final answer "
                "in the required format.", 400),
          "C": ("Give only the final answer, in the format 'Answer: X'.", 30)}


def continue_one(q, r, role, cond):
    question = q["question"] if q["question"][-1] in punctuation else q["question"] + "?"
    msgs = [{"role": "system", "content": ROLES[role]}] if ROLES[role] else []
    msgs += [{"role": "user", "content": get_cot_prompt(question, q["options"])},
             {"role": "assistant", "content": r["output"]},
             {"role": "user", "content": FOLLOW[cond][0]}]
    rec = chat_meta(msgs, json_mode=False, model=MODEL, temperature=0, max_tokens=FOLLOW[cond][1])
    pred, _ = cleansing_final_output(rec["response"])
    return {"idx": r["idx"], "cond": cond, "pred": pred, "finish_reason": rec["finish_reason"],
            "completion_tokens": (rec.get("usage") or {}).get("completion_tokens"),
            "output": rec["response"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", default="empty")
    ap.add_argument("--short", type=int, default=300)
    ap.add_argument("--long", type=int, default=700)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--dir", default=os.path.join(ROOT, "runs/p0_sweep"))
    ap.add_argument("--out", default=os.path.join(ROOT, "runs/p0_continuation"))
    a = ap.parse_args()
    QA = [json.loads(l) for l in open(DATA)]
    S = load(f"{a.dir}/{a.role}_{a.short}.jsonl")
    L = {r["idx"]: r for r in load(f"{a.dir}/{a.role}_{a.long}.jsonl")}
    todo = [r for r in S if category(r) != "clean"]
    res = {}
    for cond in FOLLOW:
        with ThreadPoolExecutor(a.workers) as ex:
            res[cond] = list(ex.map(lambda r: continue_one(QA[r["idx"]], r, a.role, cond), todo))
    os.makedirs(a.out, exist_ok=True)
    with open(f"{a.out}/{a.role}_{a.short}.jsonl", "w") as f:
        for cond in FOLLOW:
            for x in res[cond]:
                f.write(json.dumps(x) + "\n")

    gold = {r["idx"]: r["gold"] for r in S}
    cat = np.array([category(r) for r in todo])
    acc = {"short (as is)": np.array([r["pred"] == r["gold"] for r in todo]),
           f"A fresh @{a.long}": np.array([L[r["idx"]]["pred"] == r["gold"] for r in todo]),
           "B continue": np.array([x["pred"] == gold[x["idx"]] for x in res["B"]]),
           "C answer-only": np.array([x["pred"] == gold[x["idx"]] for x in res["C"]])}
    print(f"role={a.role} short={a.short}: {len(todo)} affected questions")
    groups = [("ALL affected", np.ones(len(todo), bool))] + \
             [(c, cat == c) for c in ("length", "no_answer", "malformed", "extract_fail") if (cat == c).any()]
    print(f"{'condition':16s}" + "".join(f"{g:>16s}" for g, _ in groups))
    print(f"{'n':16s}" + "".join(f"{k.sum():16d}" for _, k in groups))
    for name, v in acc.items():
        print(f"{name:16s}" + "".join(f"{v[k].mean():16.3f}" for _, k in groups))
    same_A = np.mean([x["pred"] == L[x["idx"]]["pred"] for x in res["B"]])
    print(f"\nB continuation agrees with fresh long-budget answer on {same_A:.3f} of affected questions")


if __name__ == "__main__":
    main()
