# 3_disagreement_intervention

**Paper:** §4.3–4.4, Appendix D (G3, G4)

Disagreement-triggered localized interaction (TDC) vs. compute-matched SampleMore: PILOT (GO/NO-GO) and the sealed RESERVE confirmation. Gold labels are only in `results/splits/*_labels.json`.

**Reproduce** (from this folder, with `PYTHONPATH=../../common`): `python code/sealed_analysis.py --split pilot` and `--split reserve`

All checks at once: `bash ../../tools/verify_reproduction.sh` from the repository root.
