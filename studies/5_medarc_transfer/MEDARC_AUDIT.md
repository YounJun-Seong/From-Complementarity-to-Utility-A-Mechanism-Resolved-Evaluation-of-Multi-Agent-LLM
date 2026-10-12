# MedARC feasibility audit (code reading only; no run)

- **Repository:** github.com/asdmiao/MedARC at commit `3b6b5a6` (2025-09-08), README "MedARC METHOD".
- **Contents:** 3 task folders (yes_or_no-QA = PubMedQA, factoid-QA = BioASQ-10b factoid,
  open-ended-QA = BioASQ-12b summary) with 2,076 lines of Python, plus `data/` (PubMedQA 498,
  BioASQ factoid 166, summary 383).
- **Not checked:** the paper itself (IJMI 2026) has not been read here. Whether the paper's main
  numbers come from the homogeneous or the heterogeneous configuration must be checked against it.

## ① Can it run on Qwen2.5-14B / local vLLM without changing the method's semantics?

**Yes, for the homogeneous configuration**, which is a native released configuration.

Each task has two multi-agent generators:

| Script | Agents | Output file | Read by the released `eval_*_agents_confidence.py`? |
|---|---|---|---|
| `gen_<task>_deepseek.py` | 3 × DeepSeek-V3 (homogeneous) | `<task>_ds_3_3.json` | **yes** |
| `gen_<task>_multi-agents.py` | GLM-4-air, GPT-4o-mini, DeepSeek-V3 (heterogeneous) | `<task>_multi-agents_3_3.json` | no; the file name is not read anywhere |

The only differences between the two scripts are the model and client lines. Running the
homogeneous script with every call sent to Qwen2.5-14B therefore needs a change of base_url and model
name only. This is a backbone substitution of the same kind as the one already applied to MedAgents
and MedLA in Part I.

**What stays different from the paper:**
- the backbone itself (Qwen2.5-14B instead of DeepSeek-V3);
- the heterogeneous variant, which would need three model families (API keys for GLM and DeepSeek,
  or a local substitute trio, which would be a documented deviation).

**Generation parameters.** No temperature, top_p or max_tokens is set anywhere, so provider defaults
apply. With vLLM `--generation-config vllm` the defaults are T = 1.0 and max_tokens = the context
remainder. These defaults have to be fixed and reported explicitly; leaving them unset would mean
unbounded outputs.

**Task fit.** There is no multiple-choice variant. The prompts are context-based ("Based on the
following reference context: …"). There are two options:
- **(a) Native:** run on the released PubMedQA test file (yes / no / maybe, 498 questions). No
  prompt change is needed, and the project has never used PubMedQA, so the data is clean.
- **(b) MedMCQA RESERVE:** this needs a new answer-format prompt (A–D, no context). That is a
  modification, so the results would describe "MedARC-style" interaction, not MedARC.

## ② Does an agent-level pre/post trajectory exist (can DI be defined natively)?

**Yes.**
- `agent_contexts[i]` stores the full per-agent conversation, and the output JSON keeps every
  assistant turn of every agent in every round. Each agent's answer and stated confidence are
  therefore observable at rounds 0, 1 and 2.
- The interaction signal is **not a single opposing claim**. From round 1 on, each agent receives an
  LLM-written summary ("key similarities and differences") of the *other two agents' latest
  responses* and is asked to re-analyse the question.

**Native G5:**
- For agent i at round r → r+1, the agent is on the *wrong side* if its answer is wrong and at least
  one other agent's round-r answer is gold. It is on the *right side* if its answer is gold and at
  least one other agent's is wrong.
- DI_native = P(W→C | wrong side) − P(C→W | right side), computed on these discordant agent-rounds.
- **The Control / Blinded ladder does not exist natively.** Adding conditions would be an added
  probe that is not part of MedARC. If done, it has to be reported as a separate probe.

## ③ Call DAG and compute (per question, 3 agents × 3 rounds, as released)

```
round 0:  A1, A2, A3 (independent, each answers + confidence)              3 calls
round 1:  for i in 1..3: summary(others' round-0 responses) → A_i revises    3 + 3 calls
round 2:  for i in 1..3: summary(others' round-1 responses) → A_i revises    3 + 3 calls
aggregation: offline (no LLM)                                         total 15 calls
```

- Agent contexts accumulate the whole history, so round 2 inputs contain round-0 and round-1 turns
  plus 2 summaries.
- **Rough token estimate for PubMedQA:** about 20–25k tokens per question.
  - The context takes about 300–400 tokens; outputs are about 500 tokens each, with no cap set.
  - The 6 summarizer calls take about 1k input each.
  - This is the same order as MedAgents (about 22–25k on Qwen-14B). It must be measured on a token
    probe before any k is matched.

## Released evaluation code: findings that matter before any comparison

These belong to the pipeline layer of the audit (Part I-style).

1. **Final-answer selection.**
   - `eval_pubmedqa_agents_confidence.py` pools *all 9* assistant turns, including the round-0
     (pre-interaction) answers.
   - It selects the turn with the **highest** stated confidence: `sorted_indices[0]`. The inline
     comment says "second highest", and the variable is named `second_highest_idx`.
   - `most_frequent()` is defined but unused, and it would return `List[-1]` anyway.
2. **Answer parsing.**
   - `parse_answer` returns "yes" if the substring "yes" appears anywhere, then "no", then "maybe".
   - "no" also matches "not" and "know".
   - If no `\boxed{}` is found, the **entire last response** is parsed. This yes-first
     substring rule could bias accuracy and must be measured (parse / fallback rate).
3. **Confidence extraction.** This is the first `confidence: <number>` regex match. If none is found,
   the confidence is 0, which removes that turn from the top choice.
4. **Data.** `data/random_extraction.py` draws a 10% sample with an unseeded `random.sample`, so the
   released test files cannot be regenerated. The files themselves are fixed, though.
5. **Factoid accuracy.** It counts as correct when gold is a substring of the prediction, which is
   lenient.

## Gate profile available natively (PubMedQA, homogeneous Qwen-14B)

| Gate | Native? | How |
|---|---|---|
| G1 | ✓ | G_app vs the released single-round baseline (`*_deepseek_danlun.py`, 1 agent × 1 round); G_base vs a strong single agent; G_int vs SC at ≈ MedARC tokens; G_het possible (independent role-free samples are the natural control; MedARC has no roles) |
| G2 | ✓ | Needs ≥ 2 runs of MedARC and of SC for cross-fitting (T = 1.0 default makes runs stochastic) |
| G3 | N/A | Always-on; there is no selector |
| G4 | ✓ | κ_gen = P(gold ∈ round-0 answers) |
| G5 | ✓ (native DI only) | Agent-round trajectories after the summary; no Control / Blinded ladder |
| G6 | ✓ | Released confidence selection vs (i) majority of round-2 answers, (ii) round-0 majority; pipeline sensitivity to items 1–2 above |

## Recommendation

MedARC is feasible **without modifying the method** if the following hold:
- the homogeneous released configuration is used;
- it runs on its own PubMedQA test file;
- T and max_tokens are fixed explicitly, because the released defaults leave them unset.

Its evaluation code contains selection and parsing issues that are themselves G1/G6 findings. Both
the released selection and pre-declared alternatives should be reported.

**Before a Phase V freeze, three decisions are needed:**
1. **Dataset:** PubMedQA native, or MedMCQA RESERVE with a modified prompt.
2. **Generation defaults:** T = 1.0, max_tokens = 1,024 or similar.
3. **Heterogeneous configuration:** skip it, or add it via API (GPT-4o-mini is available; GLM-4-air
   and DeepSeek-V3 need keys and cost).

## Paper ↔ released code ↔ Phase V instantiation (paper audit, 2026-10-05)

**Source.** Miao Y, Wen J, Luo Y, Li J. *MedARC: Adaptive multi-agent refinement and collaboration for
enhanced medical reasoning in large language models.* Int J Med Inform 206:106136 (Feb 2026),
doi:10.1016/j.ijmedinf.2025.106136, PMID 41109093.

The full text is subscription-only: Unpaywall, Semantic Scholar and Europe PMC all report it as
closed, and ScienceDirect returns 403. **Only the abstract (PubMed) has been read.** Every cell
marked *full text* below needs the PDF.

| Item | Paper | Released code (`3b6b5a6`) | Phase V instantiation (proposed) |
|---|---|---|---|
| Official code | "publicly available on GitHub (https://github.com/asdmiao/MedARC)" | this repository | same, homogeneous scripts |
| Headline backbone | "using **DeepSeek-V3 as the backbone model**" (PubMedQA 72.9% zero-shot → 77.2%) | `gen_*_deepseek.py`: 3 × deepseek-v3; `eval_*` reads `*_ds_3_3.json` | 3 × Qwen2.5-14B, homogeneous (backbone substitution) |
| Heterogeneous agents | an analysis ("we explore the impact of heterogeneous agent configurations"), not the headline | `gen_*_multi-agents.py` (GLM-4-air, GPT-4o-mini, DeepSeek-V3); no evaluator reads its output | excluded |
| Mechanisms | (1) structured inter-agent summarization of agreements and disagreements; (2) confidence-aware aggregation "based on the most reliable and well-reasoned contributions" | (1) `generate_summary` of the other agents' latest turns; (2) the highest stated confidence among all 9 turns (the comment says "second highest") | released evaluator reproduced as is, plus a controlled offline evaluator |
| Number of agents / rounds | *full text* (the abstract mentions "varying numbers of debate rounds") | 3 agents × 3 rounds | 3 × 3 |
| Datasets | three benchmarks: yes/no, factoid, open-ended | PubMedQA 498, BioASQ-10b factoid 166, BioASQ-12b summary 383 (10% samples) | PubMedQA 498 |
| Baselines | zero-shot and CoT | `*_danlun.py` (1 agent × 1 round), `*_danluncot.py` | released zero-shot baseline as B_released, plus a strong single agent and token-matched SC |
| Exact aggregation description | *full text* | see §"Released evaluation code" | both are reported |
| Generation parameters | *full text* | unset (provider defaults) | T = 1.0, max_tokens 1,024 (declared as the execution parameters left unspecified by the release) |
| Ablations | summarization and aggregation "contribute independently" | no ablation scripts released | not reproduced |

**What the abstract settles.** The headline result uses a single backbone (DeepSeek-V3), and the
heterogeneous configuration is a secondary analysis. The homogeneous Qwen×3 instantiation is
therefore the released configuration under a controlled common backbone. It must be named "a
released homogeneous MedARC configuration under a controlled common backbone", not "MedARC main
setting replication", until the full text confirms the agent and round counts and the aggregation
rule.

**Needs the full text before freezing:**
1. The agent and round counts behind the headline number.
2. The paper's description of confidence-aware aggregation: highest, second highest, or something
   else, and whether round 0 is included.
3. The answer-parsing description.
4. The sampling parameters.
5. The test-set sizes and the PubMedQA split used.

## Full-text audit (2026-10-05; the user supplied the full text and figures)

| Item | Paper (full text) | Released code | Agreement |
|---|---|---|---|
| Rounds | "We set the number of debate rounds T to 2 as the default" (§2.5). Headline 0.772 = DR2; DR3 = 0.759 (Table 3) | `rounds = 3`, where round 0 is the initial answer and rounds 1–2 are debate | **consistent** (1 initial + 2 debate = DR2) |
| Agents | N agents; the figures show 3 | 3 | consistent |
| Homogeneous summarizer | "configured to match the primary model in homogeneous settings" | summarizer = deepseek-v3 in `gen_*_deepseek.py` | consistent |
| Summary input | "excluding self-response"; the other agents' latest turns | `construct_message` summarizes the other agents' turn at index 2r−1 | consistent |
| **Aggregation** | Eq. 4 aggregates the **final-round** opinions {P_i^(T)}. Responses with confidence **< 0.85** are discarded. "If no response meets the threshold, the system re-initiates the response generation process for that round." The highest remaining confidence is selected (§2.2.3; Fig. 1; Fig. 4: "Select answers from final debate rounds that exceed the confidence threshold, then choose … with the highest confidence") | pools **all 9 turns including round 0**, **no threshold**, no re-initiation, `argsort[::-1][0]` = highest (the comment says "second highest") | **differs** (round scope, threshold, re-initiation) |
| Baselines | zero-shot = initial-response template; CoT = zero-shot + "Let's think step by step." | `*_danlun.py`, `*_danluncot.py` (1 agent × 1 round) | consistent |
| Sampling parameters | not stated | unset | — |
| Datasets | PubMedQA, BioASQ-10B-Factoid, BioASQ-12B-Summary; splits in Appendix B (not seen) | PubMedQA 498 etc. | consistent as far as visible |
| Authors' own limitation | "our baselines did not cover stronger alternatives such as **self-consistency**" (§4) | — | motivates G_int directly |
