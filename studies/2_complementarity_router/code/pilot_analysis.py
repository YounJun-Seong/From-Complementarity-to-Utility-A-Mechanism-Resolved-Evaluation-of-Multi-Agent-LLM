"""Sealed Go/No-Go analysis of the Phase II PILOT (PHASE2_FREEZE.md §3.4–3.6). No LLM calls.
Written and committed before any PILOT output existed. The decision uses Qwen2.5-14B only; Llama-3.1-8B
is reported with the same quantities.

Pools: SC runs 1–2 = A (selection, 40 samples), run 3 = E (evaluation, 20), runs 4–6 = P (placebo, 60,
split 30 / 30). S and M1: runs 1–2 = selection, run 3 = evaluation.

  python studies/2_complementarity_router/code/pilot_analysis.py   ->  studies/2_complementarity_router/results/pilot/RESULTS.{txt,json}
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import features as F            # noqa: E402
import oracle_analysis as OA    # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
BASE = f"{ROOT}/studies/2_complementarity_router/results/pilot"
MODELS = [("Qwen2.5-14B-Instruct", True), ("Llama-3.1-8B-Instruct", False)]   # (tag, decision model)
N = 300


def jl(p):
    return [json.loads(l) for l in open(p)]


def load(tag):
    d = f"{BASE}/{tag}"
    S = [jl(f"{d}/S/run{r}.jsonl") for r in (1, 2, 3)]
    SC = [jl(f"{d}/SC/run{r}.jsonl") for r in range(1, 7) if os.path.exists(f"{d}/SC/run{r}.jsonl")]
    M = []
    for r in (1, 2, 3):
        by = {}
        for sd in glob.glob(f"{d}/M1/run{r}/shard*"):
            s = int(os.path.basename(sd)[5:])
            for k, l in enumerate(open(f"{sd}/chatgpt-syn_verif")):
                by[s * 20 + k] = json.loads(l)
        M.append([by[i] for i in range(N)])
    cost = [json.loads(l) for f in glob.glob(f"{d}/M1/run*/shard*/cost.jsonl") for l in open(f)]
    return S, SC, M, cost


def main():
    idx = json.load(open(f"{ROOT}/studies/2_complementarity_router/results/splits/pilot.json"))["idx"]
    qs = [r["question"] for r in jl(f"{ROOT}/data/splits/MedQA_p2pilot/test.jsonl")]
    out, lines = {}, []
    for tag, decides in MODELS:
        S, SC, M, cost = load(tag)
        assert all(len(run) == N for run in S + SC + M), f"{tag}: incomplete runs"
        gold = np.array([r["gold"] for r in S[0]])
        assert all(r["gold"] == g for run in SC for r, g in zip(run, gold))
        assert all(r["gold_answer"] == g for run in M for r, g in zip(run, gold))
        Sm = np.array([[r["E3"] == r["gold"] for r in run] for run in S], float)
        Mm = np.array([[r["pred_answer"] == r["gold_answer"] for r in run] for run in M], float)
        votes = [np.array([r["votes"] for r in run]) for run in SC]
        tok = dict(S=np.mean([r["S_prompt_tokens"] + r["S_completion_tokens"] for run in S for r in run]),
                   SC=np.mean([(r["prompt_tokens"] + r["completion_tokens"]) / 20 for run in SC[:3] for r in run]),
                   M=np.mean([c["prompt_tokens"] + c["completion_tokens"] for c in cost]))   # one record per question-run
        X = F.build(S[0], votes[0][:, :4], idx, qs)
        res = OA.analyze(Sm, np.concatenate(votes[:3], axis=1), gold, Mm, tok, X,
                         placebo_votes=np.concatenate(votes[3:], axis=1), pool_split=40)
        res["placebo_samples"] = 20 * (len(votes) - 3)
        out[tag] = dict(tok=tok, decision_model=decides, **res)

        f = lambda c: f"{c[0]*100:+.1f} pp [{c[1]*100:+.1f}, {c[2]*100:+.1f}]" if c else "n/a"
        lines += [f"=== {tag} {'(DECISION MODEL)' if decides else '(reported, no decision role)'} ===",
                  f"k* = {res['k_star']}  (tokens/q: S {tok['S']:.0f}, SC per sample {tok['SC']:.0f}, M1 {tok['M']:.0f}); "
                  f"placebo pool {res['placebo_samples']} samples",
                  f"types (selection half): " + "  ".join(f"{t} {v*100:.1f}%" for t, v in res["types"].items()),
                  f"empirical-oracle acc  {{S,SC,M}} {res['acc_oracle_CS']:.3f} ({res['tok_oracle_CS']:.0f} tok/q, M chosen "
                  f"{res['share_M_chosen']*100:.1f}%)   {{S,SC}} {res['acc_oracle_S']:.3f} ({res['tok_oracle_S']:.0f} tok/q)",
                  f"G1  H                 {f(res['H'])}   -> {'PASS' if res['pass_G1'] else 'FAIL'}",
                  f"G2  H − H_plc         {f(res['H_minus_H_plc'])}   (H_plc {f(res['H_plc'])})   -> {'PASS' if res['pass_G2'] else 'FAIL'}",
                  f"G3  type C: n_C = {res['n_C']}, M1 run3 − SC pool E  {f(res['G3_contrast'])}   -> {'PASS' if res['pass_G3'] else 'FAIL'}",
                  f"G4  OOF AUC {res['G4']['auc']:.3f} [{res['G4']['lo']:.3f}, {res['G4']['hi']:.3f}] (positives {res['G4']['n_pos']})"
                  f"   -> {'PASS' if res['pass_G4'] else 'FAIL'}",
                  "secondary utility AUC by λ (pp/10k tok): " + "  ".join(
                      f"{lam}: {v['auc']:.3f}" for lam, v in res["G4_utility"].items()),
                  f"mapping (§3.5): {res['decision']}", ""]
    q = out["Qwen2.5-14B-Instruct"]["decision"]
    lines.append(f"PHASE II GO/NO-GO (Qwen2.5-14B): {q}")
    os.makedirs(BASE, exist_ok=True)
    open(f"{BASE}/RESULTS.txt", "w").write("\n".join(lines) + "\n")
    json.dump(out, open(f"{BASE}/RESULTS.json", "w"), indent=1, default=float)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
