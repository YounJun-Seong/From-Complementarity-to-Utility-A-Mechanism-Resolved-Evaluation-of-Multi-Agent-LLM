"""Judge validity control (added 10-11 after J1 descriptives, see POSTHOC_EXPERIMENTS.md): J1 prompt with the summary
taken from a DIFFERENT question (VI_A, idx+1 cyclic, same run/round/agent/j). A valid judge should answer
'unstated' / evidence 'none'. Also re-judges the same items with the true summary (cached from J1). Gold not read.

  TMC_BASE_URL=... python experiments/posthoc_judge_control.py  ->  studies/7_posthoc_audits/results/judge_control.json
"""
import collections
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import posthoc_summary_judge as J   # noqa: E402

items = [it for it in J.j1_items() if it["set"] == "VI_A"]
by = {(it["run"], it["idx"], it["round"], it["agent"], it["j"]): it for it in items}
N = max(it["idx"] for it in items) + 1
rng = random.Random(0)
pick = rng.sample(items, 500)
ctrl = [dict(it, summary=by[(it["run"], (it["idx"] + 1) % N, it["round"], it["agent"], it["j"])]["summary"]) for it in pick]
with ThreadPoolExecutor(64) as ex:
    real, mis = list(ex.map(J.run_j1, pick)), list(ex.map(J.run_j1, ctrl))
res = {}
for name, rows in (("true_summary", real), ("mismatched_summary", mis)):
    res[name] = dict(answer_in_summary=dict(collections.Counter(r["answer_in_summary"] for r in rows)),
                     label_equals_source=sum(r["answer_in_summary"] == r["src_label"] for r in rows) / len(rows),
                     evidence=dict(collections.Counter(r["evidence_preserved"] for r in rows)),
                     misrepresented=sum(bool(r["misrepresented"]) for r in rows) / len(rows))
# chance level for label match under mismatch: P(source label == label of the other question's response)
res["n"] = len(pick)
json.dump(res, open(f"{J.OUT}/judge_control.json", "w"), indent=1)
print(json.dumps(res, indent=1))
