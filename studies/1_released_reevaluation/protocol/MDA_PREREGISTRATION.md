# MDAgents bridge ablation — condition definitions (frozen)

Frozen 2026-09-24 before any F1/F2/F3 result was inspected. MedQA test (5-option, US),
questions 0–199, 5 repeats per condition (run1..run5), each shard a fresh process.
Code: `external/MDAgents` on top of the released commit `3adbd76`.

| cond | code commit | RNG (option shuffle, few-shot sampling, emoji) | parser | model allocation | decoding | collaboration logic |
|---|---|---|---|---|---|---|
| F0 | `ff013b1` | released: unseeded | released (`MDA_STRICT_PARSE=1`) | released: router + intermediate recruiter gpt-3.5-turbo, rest gpt-4o-mini | released | released |
| F1 | `d5a42e0` | **fixed**: `random.seed(f"{run}:{question}")` | released | released | released | released |
| F2 | `ce9c41a` | fixed | **robust** (3 sites below) | released | released | released |
| F3 | `ce9c41a` | fixed | robust | **unified**: all calls gpt-4o-mini-2024-07-18 (`MDA_UNIFY=1`) | released | released |

Each step changes exactly one factor. "Released decoding" = no temperature sent (API default)
for all `chat` calls, temperature 0.0 for the `temp_responses` final-decision calls.
gpt-4o-mini calls are pinned to the `2024-07-18` snapshot in every condition (the release
calls the `gpt-4o-mini` alias).

## F2 parser-repair scope (exactly these three sites; nothing else)
Identified from failures observed while the F0 diagnostic run was in progress (before F1–F3
results existed):
1. advanced path — recruiter output in markdown (`### Group`, `**Member**`) yields empty
   groups -> `IndexError` in `Group.interact`. Repair: strip markdown markup from the recruiter
   output before the released group/member parsing.
2. intermediate path — agent-list loop crashes on recruiter lines that are not
   "N. Role - description" (e.g. a trailing sentence). Repair: skip such lines, as the released
   agent-construction loop right below already does; number agents in creation order.
3. intermediate path — `parse_hierarchy` crashes on "Hierarchy: X" without ">". Repair: read it
   as "this expert reports to X". The result only feeds the printed team tree.

## Explicitly NOT changed in F0–F3
- `noAns`: when no agent asks to talk in round 1, the released loop breaks before final answers
  are collected and the moderator receives `final_answer=None`. This is collaboration control
  flow, not parsing; kept as released and reported as a rate. A repair (F2b = F2 + collect agent
  answers before the moderator when `num_yes == 0`) is a separate, post-hoc exploratory
  robustness condition if run.
- decoding, prompts, number of agents/teams/rounds, routing rule.

## Scoring and reporting (all conditions identical)
- Answer = deterministic extraction from the final decision text (`experiments/mda_analyze.py`:
  explicit "answer (X)" > bold "(X)" > last "(X)", last 600 chars). Unparsed = wrong.
- Per run: completion rate, crash type counts (advanced / intermediate parser), noAns rate,
  accuracy on all 200 (crash = wrong) and among completed, route distribution, calls / tokens /
  $ per question, wall time, API retries / rate limits / timeouts.
- Across runs: accuracy mean ± SD, pairwise and all-5 answer agreement, route agreement,
  basic↔advanced switch rate.
- Strong-single reference: P0 `S` (zero-shot CoT, same questions 0–199, P0 epoch).
- F0–F3 run concurrently; accuracy is comparable across conditions, latency and completion
  may be affected by shared API load (logged per question).
- F0 has no per-question wall_time / retry fields (started before that logging); wall time is
  reconstructed from cache timestamps.

## Amendment 1 — F2 implementation correction (2026-09-24, before any F2/F3 accuracy was inspected)
F2 was preregistered to normalize superficial formatting of the recruiter output without
altering routing, prompts, decoding, or collaboration logic. An implementation audit of the
first F2/F3 launch (code `ce9c41a`; ~200/1000 questions each, crash counts only inspected:
F2 26, F3 55 advanced-path `IndexError`s) showed that (a) leading Markdown list markers
(`- **Member 1:**`) were not removed and (b) conversational text before the first
`Group <n>` header became a memberless pseudo-group (22% of advanced recruiter outputs).
Both are surface-format failures; the correction (code `6588e5b`, `_normalize_recruited`) strips
headings / bold / leading list markers and drops text before the first `Group <n>` header,
then runs the released parser unchanged. No LLM re-interpretation, no inferred members, no
recruiter re-call.
- Unit test `experiments/test_mda_f2_parser.py`: the five member-line variants parse
  identically; replaying all 422 cached advanced recruiter outputs (incl. all 86 from the
  buggy F2/F3 crashes) leaves 4 (0.9%) empty-group outputs, out of scope and reported as
  residual crashes (numbered member lists without "Member N:"; the word "Group" in member text).
- The first launch is archived as `runs/mda/F2_buggy`, `runs/mda/F3_buggy` and excluded from
  confirmatory results. F2/F3 restarted from scratch as `F2_final_v2` / `F3_final_v2` with the
  condition name in the cache nonce (no cached call from the buggy launch is reused).
- Harness change for the restart: API retry limit 8 -> 15. A retry re-sends the identical
  request after a transient API error (HTTP 429 rate limit, timeout, 5xx); it does not change the
  prompt, sampling parameters, or answer selection, and only a successful response is used.
  Rate-limit failures are infrastructure, not treatment; any question that still ends in
  `RateLimitError` is re-run in the same condition namespace and reported.

| cond | code commit |
|---|---|
| F2_final_v2 | `6588e5b` |
| F3_final_v2 | `6588e5b` + `MDA_UNIFY=1` |
