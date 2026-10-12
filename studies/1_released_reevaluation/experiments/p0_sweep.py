"""P0-1/P0-2: MedAgents' own single-agent CoT baseline (base_cot) on full MedQA,
swept over system role x generation budget. Prompt and answer extraction are
imported unchanged from the patched MedAgents release; only system_role and
max_tokens vary. Records finish_reason for the truncation analysis.

  python experiments/p0_sweep.py --roles empty,expert --budgets 300,700
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from string import punctuation

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "external", "MedAgents"))
from llm import chat_meta                                    # noqa: E402
from prompt_generator import get_cot_prompt                  # noqa: E402
from data_utils import cleansing_final_output                # noqa: E402

MODEL = "gpt-4o-mini-2024-07-18"
ROLES = {"empty": "",                                        # base_cot / base_cot_budget
         "expert": "You are an expert physician. Let's think step by step."}  # _role / _strong
DATA = os.path.join(ROOT, "external/MedAgents/datasets/MedQA/test.jsonl")
RUN = None


def run_one(q, idx, role, budget):
    question = q["question"] if q["question"][-1] in punctuation else q["question"] + "?"
    msgs = []
    if ROLES[role]:
        msgs.append({"role": "system", "content": ROLES[role]})
    msgs.append({"role": "user", "content": get_cot_prompt(question, q["options"])})
    kw = {"nonce": f"{RUN}:{idx}"} if RUN else {}   # per-run namespace for repeated T=0 runs
    rec = chat_meta(msgs, json_mode=False, model=MODEL, temperature=0, max_tokens=budget, **kw)
    pred, _ = cleansing_final_output(rec["response"])
    u = rec.get("usage") or {}
    return {"idx": idx, "gold": q["answer_idx"], "pred": pred,
            "finish_reason": rec["finish_reason"],
            "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
            "output": rec["response"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roles", default="empty,expert")
    ap.add_argument("--budgets", default="128,256,300,512,700,1024,1536")
    ap.add_argument("--n", type=int, default=-1)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", default=os.path.join(ROOT, "runs/p0_sweep"))
    ap.add_argument("--run_label", default=None, help="cache namespace for repeated runs")
    a = ap.parse_args()
    global RUN
    RUN = a.run_label
    QA = [json.loads(l) for l in open(DATA)]
    QA = QA if a.n < 0 else QA[:a.n]
    os.makedirs(a.out, exist_ok=True)
    for role in a.roles.split(","):
        for budget in map(int, a.budgets.split(",")):
            with ThreadPoolExecutor(a.workers) as ex:
                rows = list(ex.map(lambda iq: run_one(iq[1], iq[0], role, budget), enumerate(QA)))
            with open(f"{a.out}/{role}_{budget}.jsonl", "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            acc = sum(r["pred"] == r["gold"] for r in rows) / len(rows)
            trunc = sum(r["finish_reason"] == "length" for r in rows) / len(rows)
            ctok = sum(r["completion_tokens"] or 0 for r in rows) / len(rows)
            print(f"{role:6s} {budget:5d}  acc={acc:.3f}  finish=length {trunc:.3f}  "
                  f"out_tok/q={ctok:.0f}  n={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
