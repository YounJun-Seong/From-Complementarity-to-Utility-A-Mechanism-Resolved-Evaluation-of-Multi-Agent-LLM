# 5_medarc_transfer

**Paper:** §4.5, Figure 2(b), Table 2, Appendix F (G1, G2, G4–G6)

Frozen framework applied to a controlled instantiation of MedARC on its PubMedQA set (n = 498), Qwen2.5-14B-Instruct, SC-25 compute control. Amendments 1–2 (confidence extraction, SC request batching) are recorded at the end of `PROTOCOL.md`; `MEDARC_AUDIT.md` documents the released-code audit.

**Reproduce** (from this folder, with `PYTHONPATH=../../common`): `python code/sealed_analysis.py`

All checks at once: `bash ../../tools/verify_reproduction.sh` from the repository root.
