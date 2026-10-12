# P0 analysis plan (frozen)

Frozen 2026-09-24, before any MedAgents (M1) rerun result or any primary comparison was
computed. At freeze time only these marginal accuracies had been seen:
Empty·300 = 0.643 (length 21.9%), Empty·700 = 0.747 (length 0.2%), Expert·300 = 0.599.
No new conditions are added until P0 is reported.

## Evaluation epoch
- All P0 numbers come from one synchronized epoch: the response cache `data/llm_cache/`
  started empty on 2026-09-24 (the supplement ships no cache), so every response in it was
  generated in this epoch. `experiments/epoch_check.py` verifies this from the records.
- Model `gpt-4o-mini-2024-07-18`, temperature 0 (continuation / extraction calls too).
- Supplement numbers (0.636, 0.754, 0.755, 0.768) are **historical replication references
  only**. They are never paired with epoch results.

## Systems (all MedQA test, n = 1,273, same question order)
| ID | System | max_tokens | Source |
|---|---|---|---|
| B1 | MedAgents base_cot, empty role | 300 | `runs/p0_sweep/empty_300.jsonl` |
| B2 | base_cot, expert role | 300 | `runs/p0_sweep/expert_300.jsonl` |
| B3 | base_cot, empty role | 700 | `runs/p0_sweep/empty_700.jsonl` |
| B4 | base_cot, expert role | 700 | `runs/p0_sweep/expert_700.jsonl` |
| S  | zero-shot CoT single agent, paper's extractor (E3) | 700 | `runs/p0_extraction.jsonl` |
| M1 | MedAgents syn_verif, released setting | released | `runs/m1_medagents/shards/*` |

## Primary comparisons (exactly three)
1. **B3 − B1**: budget effect on MedAgents' own baseline.
2. **M1 − B3**: multi-agent vs budget-fixed own baseline.
3. **M1 − S**: multi-agent vs primary single agent.

Each reports: accuracies, Δ, paired-bootstrap 95% CI (10k resamples, seed 0), exact McNemar p,
and the full 2×2 paired transition table. Equivalence (2, 3): paired-t TOST, **primary margin
δ = 0.03**; δ ∈ {0.01, 0.02, 0.04, 0.05} are sensitivity only.
No multiplicity correction across the three (each answers a separate question); reported as-is.

## Secondary / sensitivity (descriptive, not used for primary claims)
- B2, B4 (role factor); budget sweep 128/256/512/1024/1536 × both roles (dose-response).
- Failure taxonomy on B1 (length / no_answer / malformed / extract_fail / clean) with
  P(correct@700 | cause@300); B1→B3 net gain split by affected vs clean.
- Continuation on B1-affected questions: A fresh@700, B continue (+400), C answer-only (+30).
- Extraction audit of S: E1 regex, E2 strict schema (enum A–E), E3 paper, E3f (A–E prompt).
  E3 stays primary because its A–D spec is prompt text only (`json_object` mode, decoder not
  constrained); if the audit shows E is suppressed, S is re-reported with E3f and flagged.
