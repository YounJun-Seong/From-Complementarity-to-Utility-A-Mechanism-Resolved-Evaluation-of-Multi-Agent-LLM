"""Full MAS input (§9): triggered questions with a constant placeholder label (answer_idx "A",
answer "PLACEHOLDER"); asserts the placeholder is constant. Row order = order in the probe file.

  python make_m1_input.py --split pilot --probe probe.jsonl --out_dir <dataset dir>
"""
import argparse
import json
import os

from common import load_split, jl
from run_tdc import TRIGGERED


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    Q = {q["qid"]: q for q in load_split(a.split)}
    rows = [dict(question=Q[p["qid"]]["question"], options=Q[p["qid"]]["options"], qid=p["qid"],
                 answer_idx="A", answer="PLACEHOLDER") for p in jl(a.probe) if p["state"] in TRIGGERED]
    assert len({r["answer_idx"] for r in rows}) == 1 and len({r["answer"] for r in rows}) == 1
    os.makedirs(a.out_dir, exist_ok=True)
    with open(f"{a.out_dir}/test.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    json.dump([r["qid"] for r in rows], open(f"{a.out_dir}/qids.json", "w"))
    print(len(rows))


if __name__ == "__main__":
    main()
