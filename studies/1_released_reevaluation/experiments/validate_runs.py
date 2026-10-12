"""Completion check for every result file (no LLM calls). A run counts as complete only if it has
the expected number of questions (unique indices), no harness error, no api_failed flag, and a valid
final prediction for every question. Crashes that the pre-registration scores as wrong (framework
IndexError etc.) are reported separately and are not a completion failure.

  python experiments/validate_runs.py
"""
import glob
import json
import os

ROOT = os.path.join(os.path.dirname(__file__), "..")


def jl(p):
    return [json.loads(l) for l in open(p)]


def check(label, rows, n, pred_key, valid, idx_key="idx", framework_crash=None):
    idx = [r[idx_key] for r in rows] if idx_key else list(range(len(rows)))
    probs = []
    if len(set(idx)) != n or len(rows) != n:
        probs.append(f"rows {len(rows)} / unique {len(set(idx))} (expected {n})")
    api = sum(1 for r in rows if r.get("api_failed"))
    if api:
        probs.append(f"api_failed {api}")
    harness = [r for r in rows if r.get("error") and not (framework_crash and framework_crash(r))]
    if harness:
        probs.append(f"harness errors {len(harness)}")
    fw = sum(1 for r in rows if r.get("error") and framework_crash and framework_crash(r))
    bad = sum(1 for r in rows if not r.get("error") and r.get(pred_key) not in valid)
    status = "OK " if not probs else "BAD"
    extra = f"; framework crashes scored wrong: {fw}" if fw else ""
    extra += f"; invalid/unparsed predictions (scored wrong): {bad}" if bad else ""
    print(f"[{status}] {label:48s} {'; '.join(probs) if probs else 'complete'}{extra}")
    return not probs


def main():
    A4, A5 = set("ABCD"), set("ABCDE")
    ok = True
    # MedLA (GPT-4o-mini) and Layer 3 (Llama)
    for base in ("runs/medla_full", "runs/layer3"):
        for s in ("L0", "L1"):
            for f in sorted(glob.glob(f"{ROOT}/{base}/{s}/run*.jsonl")):
                ok &= check(os.path.relpath(f, ROOT), jl(f), 1273, "final_decision", A4)
        for f in sorted(glob.glob(f"{ROOT}/{base}/S4/run*.jsonl")):
            ok &= check(os.path.relpath(f, ROOT), jl(f), 1273, "E3", A4, idx_key=None)
    # P0 single runs
    for f in sorted(glob.glob(f"{ROOT}/runs/p0_sweep/*.jsonl")):
        ok &= check(os.path.relpath(f, ROOT), jl(f), 1273, "pred", A5 | {""})
    # Layer 2A M1 (framework crashes scored wrong)
    fw = lambda r: r.get("error", "").startswith(("IndexError", "KeyError", "ValueError", "AttributeError"))
    for d in sorted(glob.glob(f"{ROOT}/runs/layer2/2[AC]/*/M1/run*")):
        rows = [json.loads(l) for f in sorted(glob.glob(f"{d}/shard*/chatgpt-syn_verif")) for l in open(f)]
        ok &= check(os.path.relpath(d, ROOT), rows, 200, "pred_answer", A5 | {""}, idx_key=None, framework_crash=fw)
    # MDAgents F-conditions (framework crashes scored wrong)
    for d in sorted(glob.glob(f"{ROOT}/runs/mda/F*/run*")):
        if "buggy" in d or "archive" in d:
            continue
        rows = [json.loads(l) for f in glob.glob(f"{d}/shard*.jsonl") for l in open(f)]
        fwm = lambda r: not r.get("error", "").startswith("RateLimitError")
        ok &= check(os.path.relpath(d, ROOT), rows, 200, "response", None, framework_crash=fwm) if False else \
            check(os.path.relpath(d, ROOT), [dict(r, api_failed=(r.get("error") or "").startswith("RateLimitError")) for r in rows],
                  200, "difficulty", {"basic", "intermediate", "advanced", None}, framework_crash=fwm)
    print("\nALL COMPLETE" if ok else "\nSOME RUNS INCOMPLETE (see BAD lines)")


if __name__ == "__main__":
    main()
