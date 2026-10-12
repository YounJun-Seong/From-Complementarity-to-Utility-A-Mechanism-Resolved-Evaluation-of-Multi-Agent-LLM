# Layer 2B — single-agent generation-budget sensitivity on Qwen2.5 (frozen)

Frozen 2026-09-24 before any 2B result. Exploratory layer; not pooled with GPT-4o-mini.

**Question:** does the P0 generation-budget sensitivity of MedAgents' own single-agent baseline
(B1 = empty role, 300 tokens vs B3 = empty role, 700 tokens) replicate outside GPT-4o-mini?
This is a *single-agent* budget experiment. It does not by itself explain M1's internal truncation
(26–33% of M1 calls hit MedAgents' fixed per-call caps in 2A); that needs a separate M1
internal-budget ablation (M1-original vs M1-expanded vs S on 7B/14B), decided after 2B.

- Models / backend: as 2A (Qwen2.5-{1.5B,3B,7B,14B}-Instruct, BF16, vLLM `--generation-config vllm`).
- Conditions: B1 and B3 exactly as P0 (`experiments/p0_sweep.py --roles empty --budgets 300,700`).
- Data: MedQA 5-option, questions 0–199. Repeats: 3 runs, per-run cache namespace (`--run_label`).
- Analysis: per model, B3 − B1 (question-level paired bootstrap over 3-run means), finish=length
  rate at 300 / 700, gain split by B1-truncated vs natural-stop questions (as P0 Table 2), and
  B3 / B1 placed next to 2A's S and M1 for the same model.
