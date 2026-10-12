# Layer 2A — capability vs multi-agent gain (Qwen2.5, local) — frozen

Frozen 2026-09-24 before any 2A result. Exploratory layer (see EXPERIMENT_LAYERS.md); never
pooled with the GPT-4o-mini results.

- Models: Qwen2.5-{1.5B,3B,7B,14B}-Instruct, BF16, vLLM 0.11.0, `--generation-config vllm`
  (neutral sampling defaults = OpenAI API semantics; the HF generation_config defaults
  repetition_penalty 1.05 / top_k 20 / top_p 0.8 / T 0.7 are NOT applied), max_model_len 32768.
- Methods (unchanged code, only the backend differs):
  - S  = P0 strong single (zero-shot CoT, 700 tok, T=0, paper extractor E3) — `experiments/p0_extraction.py`
  - M1 = MedAgents syn_verif, released setting — `external/MedAgents/run.py --method syn_verif`
- Data: MedQA 5-option test, questions 0–199.
- Repeats: 3 runs per model × method; each run has its own cache namespace
  (S: `--run_label`, M1: `TMC_NONCE_NS`), so repeats are fresh generations (vLLM is not
  batch-invariant at T=0: 7B validation gave identical answers on 17/20).
- Validation: `experiments/local_validate.py` PASS on Qwen2.5-7B, 20 questions.
- Outputs: `runs/layer2/2A/<model>/{S/run<r>.jsonl, M1/run<r>/shard<s>/chatgpt-syn_verif}`.

## Analysis (fixed)
Per model: Acc_S and Acc_M1 = mean over 3 runs of per-question correctness; Δ = Acc_M1 − Acc_S with
a question-level paired bootstrap 95% CI; figure x = Acc_S, y = Δ, one point per model. Also
answer stability across runs, M1 finish=length rate (MedAgents' fixed per-call token caps),
calls / tokens per question. Unparsed answers count as wrong (same extractors as Layer 1).

## Next (not part of this freeze)
2B: B1 / B3 on the same models, decided after 2A.

## Amendment 1 (2026-09-24, before any 2A accuracy was inspected; completion counts only)
MedAgents' released `cleansing_syn_report` raises `IndexError` when a synthesized report lacks the
"Total Analysis:" delimiter (seen with Qwen2.5-7B; never with GPT-4o-mini in P0). The released
`run.py` has no per-question handling, so the whole shard process died and later questions were
not run (7B: run3/shard8 stopped after 6/20). Harness fix `3da1947`: a crashing question is
recorded with `error` and an empty answer (scored wrong, i.e. end-to-end accuracy) and the shard
continues. 3B / 1.5B run with the fix from the start; 7B / 14B (started earlier) get a fill pass:
each stopped shard is resumed from its first missing question with the fixed harness (the crashing
question replays from cache and is recorded as a crash). Crash rates are reported per model.
