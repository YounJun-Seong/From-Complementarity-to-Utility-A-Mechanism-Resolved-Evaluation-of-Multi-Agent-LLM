# 6_medarc_matched_ablation

**Paper:** Appendix F.6–F.7

Matched interaction ablation on 472 fresh PubMedQA PQA-L questions: (A) summary-mediated interaction, (B) no communication (same template without the summary), (C) placebo summaries from another question; identical round-0 turns across arms; SC-25. Prespecified primary: Acc(A) − Acc(B).

**Reproduce** (from this folder, with `PYTHONPATH=../../common`): `python code/sealed_analysis.py`

All checks at once: `bash ../../tools/verify_reproduction.sh` from the repository root.
