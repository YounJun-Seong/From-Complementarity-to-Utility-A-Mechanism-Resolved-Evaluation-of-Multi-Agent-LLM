# Layer 3 — MedLA with its paper backbone (Llama-3.1-8B-Instruct) — frozen

Frozen 2026-09-25 before any Layer-3 result. Purpose: publication reproducibility of MedLA's reported
gain (paper Table 2, MedQA-US: MedLA-LLaMA3.1-8B 62.6 vs CoT-LLaMA3.1-8B 54.9), reported separately
from the model-matched GPT-4o-mini evaluation; never pooled with it.

- Model: meta-llama/Llama-3.1-8B-Instruct (snapshot 0e9e39f), BF16, vLLM 0.11.0 with
  `--generation-config vllm` (neutral defaults), local GPU. The paper used vLLM 0.7.2 on A100s;
  version/hardware differ.
- Data: MedQA-US 4-option (MIRAGE), **all 1,273 questions** (the first 200 were shown to be
  unrepresentative in the GPT-4o-mini evaluation).
- Systems (same code as the GPT-4o-mini evaluation; backend switched via TMC_BASE_URL /
  TMC_MODEL_OVERRIDE, requested vs resolved model logged):
  - L0 = MedLA multi-agent, released defaults (17 agents, 3 rounds, T=0.7, 1 few-shot, max_tokens 8000)
  - L1 = MedLA released single-agent mode (direct answer, T=0.7)
  - S4 = strong single (zero-shot CoT, 700 tok, T=0, extractor E3)
- One run each; more runs only if pre-specified later. Same analysis as the GPT-4o-mini full test
  (paired bootstrap, McNemar, TOST δ=0.03), plus crash / unparsed rates and tokens per question.
- Comparison to the paper: L0 vs 62.6 and S4 / L1 vs 54.9 are descriptive only (different CoT
  prompt, extractor, and serving stack).

## Amendment 1 (2026-09-25) — backend context-length defect, L0 re-run
The first L0 run served Llama-3.1-8B with `--max-model-len 32768`. MedLA requests `max_tokens=8000`
per call and its multi-round prompts grow long, so the server rejected 3,216 calls with HTTP 400
("max_tokens too large" / "maximum context length exceeded"). MedLA's decision code swallows the
exception and returns an empty answer, so 1,007 questions ended with an empty final decision; the
harness flagged every affected question (`api_failed`, 1,043). With GPT-4o-mini (128k context) this
cannot occur, so this is a configuration defect of our local backend, not model behaviour.
Correction: the server is restarted with `--max-model-len 131072` (the model's native context) and
L0 is re-run for the 1,043 flagged questions only (resume keeps the 230 fully valid records, whose
calls all succeeded). The original output is archived in `runs/layer3/L0_archive_ctx32k/`.
L1 and S4 had no server errors and are kept.
