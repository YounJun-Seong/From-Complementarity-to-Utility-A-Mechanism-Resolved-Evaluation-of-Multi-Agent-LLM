"""Numbers quoted in the paper that are simple derivations of shipped outputs and are not printed by any sealed or
post-hoc analysis script. Descriptive only; no LLM calls, no new definitions of any gate quantity.

  python paper_derived_numbers.py  ->  ../results/PAPER_DERIVED_NUMBERS.txt
"""
import csv
import json
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..", "..")
S = f"{ROOT}/studies"
sys.path.insert(0, f"{S}/3_disagreement_intervention/code")
import aggregate as AG3   # noqa: E402


def jl(p):
    return [json.loads(l) for l in open(p)]


def main():
    L = []
    # Phase II probe votes (App. C.4, uncertainty selector)
    L.append("[App. C.4] Phase II 4-sample probe (first four SC run-1 votes): probes containing an empty or out-of-option vote")
    for split in ("pilot", "heldout"):
        rows = jl(f"{S}/2_complementarity_router/results/{split}/Qwen2.5-14B-Instruct/SC/run1.jsonl")
        bad = lambda v: not v or v not in "ABCDE"
        empty = sum(any(not v for v in r["votes"][:4]) for r in rows)
        anyb = sum(any(bad(v) for v in r["votes"][:4]) for r in rows)
        L.append(f"    {split:8s} n={len(rows)}: empty vote in {empty} probes; empty or out-of-option vote in {anyb} probes "
                 f"({sum(bad(v) for r in rows for v in r['votes'][:4])} votes)")

    # Phase III disagreement study (§4.4, App. D)
    R3 = f"{S}/3_disagreement_intervention/results"
    for split in ("pilot", "reserve"):
        lab = {int(k): v for k, v in json.load(open(f"{R3}/splits/{split}_labels.json")).items()}
        probe = jl(f"{R3}/{split}/main/probe.jsonl")
        trig = loss = absent = 0
        for r in probe:
            a = [p["answer"] or "" for p in r["probe"]]
            if AG3.state(a) in ("unanimous", "failure"):
                continue
            trig += 1
            g = lab[r["qid"]]
            absent += g not in a
            loss += g in a and g not in AG3.top2(a)
        res = json.load(open(f"{R3}/{split}/RESULTS.json"))
        d, t = res["diag"], res["table1"]
        L += ["", f"[§4.4, App. D] Phase III {split.upper()}: triggered {trig}/{len(probe)} ({trig / len(probe) * 100:.1f}%)",
              f"    gold in probe but outside top-2 (lost by top-two selection): {loss}; gold absent from all 4 probe answers: {absent}",
              f"    disagreement extraction: parsed share {d['extraction_parsed']:.3f} over {trig - d['single_answer']} applicable "
              f"(triggered {trig} − single-answer {d['single_answer']}); revision E3 failures {d['revision_e3_failed']:.4f}",
              f"    Full MAS / SampleMore tokens = {t['Full MAS']['tokens']:.0f} / {t['SampleMore']['tokens']:.0f} = "
              f"{t['Full MAS']['tokens'] / t['SampleMore']['tokens']:.2f}x"]
        if split == "reserve":
            sm = jl(f"{R3}/reserve/main/sample_more.jsonl")
            by = {r["qid"]: r for r in probe}
            gen = float(np.mean([by[r["qid"]]["probe_gen_tokens"]["input_tokens"] for r in sm]))
            k = sm[0]["k_additional"]
            for g_, note in ((gen, "measured: mean generation-prompt input tokens of the probe request on triggered questions"),
                             (101.0, "value quoted in POSTHOC_TRANSITION.md (no script)")):
                alt = t["SampleMore"]["tokens"] + (k - 1) * g_
                L.append(f"    TDC / SampleMore with the generation prompt charged per sample (gen ≈ {g_:.0f} tok, {note}): "
                         f"{t['TDC']['tokens']:.0f} / {alt:.0f} = {t['TDC']['tokens'] / alt:.3f}  (frozen accounting "
                         f"{t['TDC']['tokens'] / t['SampleMore']['tokens']:.3f})")

    # Phase V MedARC (§4.5, App. F)
    r5 = json.load(open(f"{S}/5_medarc_transfer/results/RESULTS.json"))
    tk, g4 = r5["descriptive"]["tokens_per_question"], r5["G4"]
    L += ["", "[§3.2, App. F.2, F.6] Phase V MedARC",
          f"    tokens/question MedARC {tk['medarc']:.0f}, SC-25 {tk['SC_k']:.0f}; SC/MedARC = {tk['SC_k'] / tk['medarc']:.3f}",
          f"    κ_loss / κ_gain = {g4['kappa_loss'][0]:.4f} / {g4['kappa_gain'][0]:.4f} = {g4['kappa_loss'][0] / g4['kappa_gain'][0]:.2f}x"]

    # Summary judge control and human check (App. F.8)
    jc = json.load(open(f"{S}/7_posthoc_audits/results/judge_control.json"))
    mm = jc["mismatched_summary"]
    L += ["", "[App. F.8] Judge swapped-summary control (n = %d)" % jc["n"],
          f"    label = source: true {jc['true_summary']['label_equals_source']:.3f}, swapped {mm['label_equals_source']:.3f}; "
          f"swapped evidence 'none' {mm['evidence']['none']}/{jc['n']} = {mm['evidence']['none'] / jc['n'] * 100:.1f}%"]
    rows = list(csv.DictReader(open(f"{S}/7_posthoc_audits/results/human_check/sheet_labeled.csv", encoding="utf-8-sig")))
    ev = Counter(r["evidence_preserved"].strip().lower() for r in rows)
    mis = [r["item"] for r in rows if r["misrepresented"].strip().lower() in ("true", "yes", "1")]
    L += [f"[App. F.8] Human check: n = {len(rows)}; evidence_preserved " + ", ".join(f"{k} {v}" for k, v in sorted(ev.items()))
          + f"; misrepresented = true on items {mis}"]

    out = f"{S}/7_posthoc_audits/results/PAPER_DERIVED_NUMBERS.txt"
    open(out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
