#!/bin/bash
# Phase VI unattended runner (user-approved 2026-10-08). Never touches other users' processes.
# Gate: both GPUs >= 60 GB free continuously for 10 min -> integrity checks -> smoke -> validate -> (gate again) -> full
#       -> validate -> commit -> sealed analysis once -> commit. Any failure stops the chain with logs kept (no retries).
ROOT=$(cd "$(dirname "$0")/../../.." && pwd); E=$ROOT/studies/6_medarc_matched_ablation/code; PY=${PYTHON:-python}
LOG=$ROOT/logs/p6_orchestrator.log; GIT="git -c user.name=tmc -c user.email=tmc@local"
say () { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG; }
gate () {   # wait until both GPUs have >= 60 GB free for 10 consecutive minutes
  local ok=0
  while [ $ok -lt 20 ]; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' ' ')
    if [ $(echo $free | awk '{print ($1>=61440 && $2>=61440)}') = 1 ]; then ok=$((ok+1)); else ok=0; fi
    sleep 30
  done
  say "GPU gate passed (free MiB: $free)"
}
cd $ROOT
say "orchestrator start; waiting for GPUs"
gate
# integrity: freeze doc and tag unchanged, data hashes, no uncommitted code changes
git rev-parse -q --verify refs/tags/phase6-freeze >/dev/null || { say "FAIL: tag missing"; exit 1; }
git diff --quiet phase6-freeze -- PHASE6_FREEZE.md studies/6_medarc_matched_ablation/results/data studies/6_medarc_matched_ablation/results/labels.json || { say "FAIL: freeze/data differ from tag"; exit 1; }
sha256sum -c --quiet studies/6_medarc_matched_ablation/results/data/DATA.sha256 || { say "FAIL: data hash"; exit 1; }
git diff --quiet HEAD -- studies/5_medarc_transfer/code studies/6_medarc_matched_ablation/code studies/2_complementarity_router/code || { say "FAIL: uncommitted code changes"; exit 1; }
(cd $E && $PY test_arms.py) >> $LOG 2>&1 || { say "FAIL: unit checks"; exit 1; }
say "integrity OK (tag, freeze, data hashes, code clean, unit checks)"
bash $E/run_phase6.sh smoke > $ROOT/logs/p6_smoke.log 2>&1 || { say "FAIL: smoke run (see logs/p6_smoke.log)"; exit 1; }
$PY $E/validate.py $ROOT/studies/6_medarc_matched_ablation/results/smoke 5 > $ROOT/studies/6_medarc_matched_ablation/results/SMOKE_VALIDATION.txt 2>&1
grep -q "^VALID" $ROOT/studies/6_medarc_matched_ablation/results/SMOKE_VALIDATION.txt || { say "FAIL: smoke validation"; exit 1; }
$GIT add studies/6_medarc_matched_ablation/results/smoke studies/6_medarc_matched_ablation/results/SMOKE_VALIDATION.txt && $GIT commit -q -m "Phase VI smoke (5 probe questions, no labels): VALID

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
say "smoke VALID; waiting for GPUs again before the full run"
gate
bash $E/run_phase6.sh full > $ROOT/logs/p6_full.log 2>&1 || { say "FAIL: full run (see logs/p6_full.log)"; exit 1; }
say "full run done; servers down. GPU memory now: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')"
$PY $E/validate.py > $ROOT/studies/6_medarc_matched_ablation/results/VALIDATION.txt 2>&1
grep -q "^VALID" $ROOT/studies/6_medarc_matched_ablation/results/VALIDATION.txt || { say "FAIL: full validation (outputs kept, no analysis)"; exit 1; }
$GIT add studies/6_medarc_matched_ablation/results/main studies/6_medarc_matched_ablation/results/VALIDATION.txt && $GIT commit -q -m "Phase VI full outputs + label-free validation (VALID) before sealed analysis

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
$PY $E/sealed_analysis.py > $ROOT/logs/p6_analysis.log 2>&1 || { say "FAIL: sealed analysis"; exit 1; }
$GIT add studies/6_medarc_matched_ablation/results/RESULTS.txt studies/6_medarc_matched_ablation/results/RESULTS.json && $GIT commit -q -m "Phase VI sealed result (run once)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
say "DONE: results in studies/6_medarc_matched_ablation/results/RESULTS.txt"
