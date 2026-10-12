# 2_complementarity_router

**Paper:** §4.2–4.3, Appendix C (G2, G3)

Cross-fitted empirical oracle (H, placebo-corrected C*), heterogeneous non-interactive control (HET), and the frozen pre-decision router evaluated once on HELD-OUT. `results/router/FREEZE_MANIFEST.sha256` lists the hashes of the frozen router files.

**Reproduce** (from this folder, with `PYTHONPATH=../../common`): `python code/pilot_analysis.py`, `code/het_compare.py`, `code/het_analysis.py`, `code/heldout_analysis.py`, `code/identification_audit.py`

All checks at once: `bash ../../tools/verify_reproduction.sh` from the repository root.
