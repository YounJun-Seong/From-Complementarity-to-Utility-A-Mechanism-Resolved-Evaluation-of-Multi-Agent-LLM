# 7_posthoc_audits

**Paper:** Appendix A (compute accounting, candidate budget), C.4 (selector extension), F.6–F.8 (summary fidelity, matched failure traces)

Post-hoc analyses, all descriptive: complementary-candidate conversion (`TRANSITION.txt`), Phase II SC token-accounting sensitivity, RESERVE SampleMore sensitivity, summary fidelity / uptake with a Qwen3-32B-FP8 judge (`judge_j1.jsonl`, `judge_j2.jsonl`, `judge_control.json`), matched lone-correct abandonment, and the 80-item blind human check. Definitions: `POSTHOC_TRANSITION.md`, `POSTHOC_EXPERIMENTS.md`, `POSTHOC_ARR2.md`.

ARR2 (revision round 2, `results/arr2/`): E1 compute-corrected C\* (`E1.txt`), E2 C\* across candidate budget k (`E2.txt`), E3 router selector extension (`E3.txt`), E4 MedARC failure traces (`E4.txt`, `E4_CASES.md`). No LLM calls.

**Reproduce** (from this folder, with `PYTHONPATH=../../common`): `python code/posthoc_transition.py`; `python code/reserve_samplemore_sensitivity.py`; `python code/posthoc_summary_analysis.py` (and `... human`); `python code/posthoc_arr2_medarc.py`; `python code/posthoc_arr2_phase2.py` (≈15 min on one CPU core). Judge generation: `code/posthoc_summary_judge.py j1|j2|sheet` (needs a vLLM server).

All checks at once: `bash ../../tools/verify_reproduction.sh` from the repository root.
