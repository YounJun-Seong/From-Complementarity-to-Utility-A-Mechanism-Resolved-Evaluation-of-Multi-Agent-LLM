#!/bin/bash
# Re-run every analysis that produces a number or figure in the paper from the shipped model outputs
# (no LLM calls, no GPU), then check the regenerated files against tools/expected_results.sha256.
set -e
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}; export PYTHONPATH=$PWD/common${PYTHONPATH:+:$PYTHONPATH}
S=studies
run() { echo "  $*"; $PY "$@" > /dev/null; }
echo "[1] released-system reevaluation"; run $S/1_released_reevaluation/experiments/synthesis.py
$PY $S/1_released_reevaluation/experiments/medla_full_report.py > $S/1_released_reevaluation/runs/medla_full/RESULTS.txt
$PY $S/1_released_reevaluation/experiments/pareto.py > $S/1_released_reevaluation/runs/layer2/2D/PARETO.txt
echo "[2] complementarity + frozen router"
for f in pilot_analysis het_analysis het_compare heldout_analysis identification_audit; do run $S/2_complementarity_router/code/$f.py; done
echo "[3] disagreement intervention"; for s in pilot reserve; do run $S/3_disagreement_intervention/code/sealed_analysis.py --split $s; done
echo "[4] controlled revision"; run $S/4_controlled_revision/code/sealed_analysis.py; run $S/4_controlled_revision/code/cud_analysis.py
echo "[5] MedARC transfer"; run $S/5_medarc_transfer/code/sealed_analysis.py
echo "[6] MedARC matched ablation"; run $S/6_medarc_matched_ablation/code/sealed_analysis.py
echo "[7] post-hoc audits"; run $S/7_posthoc_audits/code/posthoc_transition.py
run $S/7_posthoc_audits/code/reserve_samplemore_sensitivity.py
run $S/7_posthoc_audits/code/posthoc_summary_analysis.py; run $S/7_posthoc_audits/code/posthoc_summary_analysis.py human
run $S/7_posthoc_audits/code/posthoc_arr2_medarc.py
run $S/7_posthoc_audits/code/paper_derived_numbers.py
echo "  (ARR2-E1–E3 resampling takes about 15 minutes)"; run $S/7_posthoc_audits/code/posthoc_arr2_phase2.py
echo "[figures]"; run figures/code/fig_framework.py; run figures/code/fig_cug.py; run $S/4_controlled_revision/code/fig_paper.py
echo "[check]"; sha256sum -c tools/expected_results.sha256
