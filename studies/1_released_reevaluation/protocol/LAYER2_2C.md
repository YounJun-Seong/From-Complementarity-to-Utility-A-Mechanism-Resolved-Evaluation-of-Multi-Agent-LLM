# Layer 2C — M1 internal generation-budget ablation (Qwen2.5 7B / 14B) — frozen

Frozen 2026-09-24 before any 2C result. Exploratory layer; not pooled with GPT-4o-mini.

**Question:** is MedAgents' deficit against a strong single agent explained by its fixed per-call
generation ceilings? Pre-inspection (zero-cost, from logs): the expert-analysis calls (max 300)
end at the cap in 100% of calls for Qwen 7B / 14B (2A) **and for GPT-4o-mini (P0, 9,051 calls)**;
advice calls (500) 13–18% (Qwen); all other call types ≤ 3%. The released code strips the last
incomplete sentence of each analysis (`remove_incomplete_sentence`), so the 300 cap is a design
choice, not an accident.

## Conditions (same orchestration, prompts, agents, model, decoding; only ceilings differ)
| call type | released max_tokens | M1-expanded |
|---|---|---|
| domain classification | 50 | 150 |
| vote (yes/no) | 30 | 75 |
| expert analysis (question / options) | 300 | 700 |
| advice | 500 | 1000 |
| synthesis / final answer / revision | 2500 (effective 2000, api_utils cap) | unchanged (never lowered) |

Rule: ~2–2.5× for short and mid calls, ×2 for 500, never below the released ceiling.
Implemented as `TMC_BUDGET_MAP="50:150,30:75,300:700,500:1000"` (MedAgents `bfacd02`).

- M1-original and S: the 2A runs for 7B / 14B (identical frozen setup, same day and backend).
- M1-expanded: new, Qwen2.5-7B / 14B, MedQA 0–199, 3 runs, per-run cache namespace.
- Report per model: accuracy, M1-expanded − S and − M1-original (question-level paired bootstrap),
  crash rate, calls / tokens per question, truncation rate per call type.
- Interpretation fixed now: if M1-expanded stays below S -> "the multi-agent deficit is not simply
  internal truncation"; if it recovers to S or above -> "the orchestration is highly sensitive to
  per-agent generation budgets".
