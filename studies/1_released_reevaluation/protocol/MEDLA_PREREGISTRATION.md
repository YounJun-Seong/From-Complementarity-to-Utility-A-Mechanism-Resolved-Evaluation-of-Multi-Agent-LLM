# MedLA controlled evaluation — design (frozen)

Frozen 2026-09-24 before any MedLA pilot result. Naming: this is a **controlled MedLA
implementation under a common base model** (faithful workflow, gpt-4o-mini), not a reproduction
of MedLA's reported numbers (the paper used LLaMA3.1-8B).

Official code: https://github.com/alexander2618/MedLA @ `5c12cfc` (AAAI 2026, arXiv 2509.23725).
Our branch `faithful` @ `2f728ed` changes only:
- model interface: `AgentHelpler.invoke` -> `scripts/llm.py::chat_meta` (same model, messages,
  temperature, `max_tokens=8000`; unique nonce per call, so sampled calls are never merged);
- harness: per-question JSONL (index, answer, calls, tokens, served models, fingerprints,
  wall time) and crash isolation (one failing question is recorded, the joblib run continues);
- three release defects that stop the code from running at all, fixed without behaviour change:
  `run_single_basiline` builds `Agent` with a stale signature (`engine=`, `llm_name=`, `port=`),
  `wandb.init(project=args.wandb)` reads an undefined argument, and a Python-3.12-only f-string.
- `langchain==0.3.25` pinned as in the release requirements; wandb runs offline.

## Data
MedQA-US 4-option test (MIRAGE / MedQA `data_clean/.../4_options/phrases_no_exclude_test.jsonl`),
1,273 questions, same questions and order as the 5-option set (see DATA_MANIFEST.md).

## Conditions (all gpt-4o-mini-2024-07-18, all on the same 4-option data)
| ID | system | configuration | role |
|---|---|---|---|
| L0 | MedLA multi-agent workflow, released code | 17 agents, 3 rounds, T=0.7, 1 few-shot per agent (unseeded `np.random`), `max_tokens=8000` (code defaults; paper states 17 sub-agents) | system under test |
| L1 | MedLA released single-agent mode (`--baseline_bool 1`) | released prompt, T=0.7, direct answer (no CoT), 1 call | released baseline (reproduction / secondary) |
| S4 | strong single: P0 `S` (zero-shot CoT, 700 tok, T=0, + paper extractor E3) | 2 calls | **primary baseline** |

4-option results are never mixed with the 5-option P0 / MDAgents results.

## Questions (fixed in advance)
1. **Primary: L0 − S4** — does a gain remain after a strong single-agent control?
   TOST δ = 0.03 primary (0.01–0.05 sensitivity).
2. Secondary: L0 − L1 — apparent gain over the released baseline (reproduction of the claim).
3. Descriptive: S4 − L1.
A compute-matched single-agent workflow (CM-Single) is **not** part of this freeze; if added
later it is a separate, pre-specified condition.

## Analysis
- Question-level paired analysis. Each system's per-question score = mean correctness over its
  runs; paired bootstrap over questions (10k, seed 0), exact McNemar on run-1 pairs and on
  majority-vote answers, TOST as in P0. Runs are never treated as independent samples.
- Also per system: accuracy mean ± SD over runs, pairwise and all-3 answer agreement,
  completion / crash rate, calls / input / output / total tokens and $ per question
  (gpt-4o-mini list price), wall time, arbitration rate (L0).
- Cross-framework figure: strong-single gap vs inference cost (MedAgents, MDAgents, MedLA).

## Stage 1 — pilot (questions 0–199)
L0 × 3 runs, L1 × 3 runs, S4 × 3 runs (S4 repeats use a per-run cache namespace, so each run is
fresh API calls despite T=0). Smoke estimates: L0 ≈ 70 calls, ≈ $0.043 per question.

## Stage 2 — full test, expansion rule fixed now
If L0's pilot completion rate is ≥ 95% (crashes < 5%), **L0 and S4 are run on all 1,273
questions regardless of the direction of the pilot result**; L1 also (it costs 1 call per
question). Number of full runs: 3 for each, unless the pilot all-3 answer agreement of L0 is
≥ 0.95, in which case 1. If completion < 95%, the failure is diagnosed and reported first.

## Optional — MedLA-original
Paper backbone LLaMA3.1-8B-Instruct via a local vLLM server (up only for the run), questions
0–199, L0 and L1, for publication reproducibility (paper: MedLA 62.6 vs CoT 54.9 on
MedQA-US). Not required for the primary question.

## Code
MedLA `faithful` @ `2f728ed`; strong single `experiments/p0_extraction.py --data <4-option>
--run_label s4-run<r>`.

## Amendment 1 (2026-09-24, before any MedLA pilot accuracy was inspected)
Budget constraint after the OpenAI credit ran out at 15:51 (L0 pilot at ~74/200 per run; only
completion counts inspected). Stage 2 is no longer pre-committed: the full-1,273 decision is
made after the 200-question pilot (e.g. one confirmatory L0 run if the pilot difference is large
and clear; more if the confidence interval is wide around the equivalence margin). Questions that
failed with `insufficient_quota` are infrastructure failures and are re-run in the same condition
after the credit is restored.

## Amendment 2 — full test (2026-09-24, after the pilot, before any full-test result)
Pilot (200 q × 3): L0 − S4 = −0.008 [−0.047, +0.030]; TOST δ=0.03 p = 0.138 (not established);
L0 crash 0, unparsed 0, all-3 agreement 0.895, run SD 0.012; cost L0/S4 = 145× tokens, 110× $.
The pilot falls under the expansion case (CI wide around the margin), so the full test is run.
- **Primary full-test comparison:** run 1 of each system on all 1,273 questions (L0 run 1 vs S4
  run 1 vs L1 run 1), question-level paired: bootstrap CI, exact McNemar, TOST (δ = 0.03 primary).
  Symmetric replication: one realization per system.
- **Sensitivity:** S4 and L1 runs 2–3 (cheap) reported separately as run-to-run variability.
  The 200-question pilot (3 runs each) is kept as evidence of system-level stochasticity.
- **Sequential replication rule (fixed now):** after L0 run 1, compute L0 − S4 (run 1 vs run 1).
  If |Δ| ≤ 0.02, **or** its 95% CI contains −0.03 or +0.03, L0 is run two more times (runs 2–3)
  and the primary analysis becomes 3 vs 3 runs (per-question means, as in the pilot). Otherwise
  (|Δ| > 0.02 and the CI lies clearly away from both boundaries) no further L0 runs.
- Runs are fresh generations (new run ids / cache namespaces); questions 0–199 are re-run, not
  copied from the pilot. Cost estimate: L0 ≈ $49 per full run; S4 + L1 × 3 ≈ $2.

## Amendment 3 (2026-09-25) — deviation: L0 run 3 not run
The sequential rule (Amendment 2) was triggered by run 1 (L0 − S4 = −0.046 [−0.066, −0.025]; CI
contains −0.03), so L0 runs 2–3 were required. Run 2 was completed (0.7541 vs run 1 0.7588; answer
agreement 0.923; L0 run2 − S4 run2 = −0.051 [−0.071, −0.031]). **After inspecting run 2, run 3 was
skipped for budget reasons (≈ $50 API).** This is a deviation from the frozen rule and is reported as
such. Primary analysis therefore = L0 mean of runs 1–2 vs S4 mean of runs 1–3 (question-level paired
bootstrap): −0.048 [−0.066, −0.031]; run-1-only and run-2-only comparisons are reported alongside.
