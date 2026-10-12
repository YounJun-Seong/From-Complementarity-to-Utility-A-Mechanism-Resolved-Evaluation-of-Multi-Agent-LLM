# 1_released_reevaluation

**Paper:** §4.1, Table 1, Appendix B (G1)

Released-system reevaluation of MedAgents, MDAgents and MedLA against released, strong single-agent and compute-matched baselines. This folder keeps its original internal layout (`experiments/`, `runs/`, `data/`, `analysis/`); the pre-registrations and amendments are in `protocol/`.

**Reproduce** (from this folder, with `PYTHONPATH=../../common`): `python experiments/synthesis.py` → `runs/SYNTHESIS.txt`; `python experiments/medla_full_report.py`; `python experiments/pareto.py`

All checks at once: `bash ../../tools/verify_reproduction.sh` from the repository root.
