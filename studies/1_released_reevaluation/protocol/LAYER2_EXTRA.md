# Layer 2 extras — 2A-Llama, 2D compute-matched single, 2E option order (frozen)

Frozen 2026-09-25 before any result of these conditions. Exploratory open-weight layer (local vLLM,
BF16, `--generation-config vllm`); never pooled with GPT-4o-mini. MedQA 5-option, questions 0–199.

## 2A-Llama (family robustness)
Llama-3.1-8B-Instruct, S and M1 exactly as 2A (3 runs each). Same analysis as 2A.

## 2D compute-matched single-agent methods (`experiments/compute_matched.py`)
Models: Qwen2.5-7B, Qwen2.5-14B, Llama-3.1-8B. 3 runs each (seed = run).
- SC-k (k = 5, 10, 20): k samples of the S prompt at T=0.7, 700 tokens, one request with n=k;
  each sample's answer via the paper extractor E3 (T=0); majority vote, ties -> earliest sample.
  SC-20 ≈ M1's tokens per question (M1 ≈ 21–25k tokens/q on these models).
- Reflection: S reasoning (T=0) -> self-critique -> revised reasoning -> E3 (5 calls incl. draft E3).
Analysis: accuracy vs tokens per question for S, SC-5/10/20, Reflection and M1 (2A) per model
(accuracy–compute frontier); paired bootstrap of each method vs M1 and vs S over questions.

## 2E option-order robustness
Qwen2.5-14B only (the clean open-model condition). 3 seeded option permutations of the test set
(`experiments/make_perm_data.py`, seeds 1001–1003; gold letter moves in ~80% of questions).
S and M1, 1 run per permutation. Report: accuracy per permutation, M1 − S per permutation, answer
consistency across original + 3 permutations (by option content), and whether M1 − S changes sign.

## Calibration (no new runs)
From existing outputs: SC vote share (2D) and cross-run agreement (S, M1) as confidence proxies;
risk–coverage curves. Descriptive only.
