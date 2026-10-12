# Experiment layers (design, 2026-09-24)

Results from different model families are never pooled or averaged, and never presented as the
same condition.

## Layer 1 — Confirmatory, cross-framework (GPT-4o-mini-2024-07-18, OpenAI API)
| study | status | spec |
|---|---|---|
| MedAgents mechanism (P0) | done, frozen (`p0-freeze`) | PREREGISTRATION.md |
| MDAgents bridge F0–F3 (200 × 5) | finishing (credit stop at 15:51) | MDA_PREREGISTRATION.md |
| MedLA controlled (L0 / L1 / S4, 200 × 3 pilot) | pilot in progress | MEDLA_PREREGISTRATION.md |

"The primary conclusions are established under a common GPT-4o-mini backbone across
frameworks." MedLA under GPT-4o-mini is called a *model-matched controlled evaluation*.

## Layer 2 — Exploratory scaling / family robustness (open weights, local vLLM)
- Model scale: Qwen2.5-Instruct 1.5B / 3B / 7B / 14B. Family: Llama-3.1-8B-Instruct.
- Analysis axis: x = single-agent accuracy of that model, y = multi-agent gain
  (Acc_MAS − Acc_single, both on the same model), not parameter count.
- Precision: exploratory sweeps may use 4-bit (AWQ/GPTQ); every condition that is reported as
  a result is re-run in BF16 (at least the 2–3 representative conditions). Quantized and BF16
  results are never mixed.
- Stage 1 on 200 questions; only conditions needed for the paper go to 1,273. Seeds adaptive:
  3 by default, 5 if run-to-run variance is large, 10 only for key stochastic comparisons.

## Layer 3 — Original-framework fidelity (open weights, local vLLM)
- MedLA with its paper backbone (LLaMA3.1-8B) on questions 0–199 (L0, L1): publication
  reproducibility, reported separately from the controlled GPT-4o-mini evaluation.

## Held identical between the API and local backends
Prompts, max_tokens, temperature, stop conditions, answer extraction, question order and
evaluation code. Only the backend differs: `scripts/llm.py` sends every call to
`TMC_BASE_URL` (OpenAI-compatible vLLM server) with `TMC_MODEL_OVERRIDE` as the model when
these are set; the model name is part of the cache key, and each record logs
`requested_model` and `backend`.

## No-LLM analyses (existing logs)
Difficulty strata, disagreement entropy, error taxonomy, transition tables, paired statistics,
token / cost accounting, Pareto frontiers.
