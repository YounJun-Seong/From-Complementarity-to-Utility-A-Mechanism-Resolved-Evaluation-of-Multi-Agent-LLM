# Framework Patches

We evaluate the two **released** multi-agent frameworks, MedAgents and MDAgents.
We modify **only the model-interface (network) layer** and add minimal robustness
guards; the agent orchestration and decision logic are otherwise unchanged. The
modified files are in `code/patches/`. Apply them on top of each framework's
public release.

## MedAgents (faithful)

- **`api_utils.py`** — routes every LLM call through our cached client
  (`llm.py`) instead of a direct API call, so results are deterministic, cached,
  and answered on the same model as the single-agent baselines. `MODEL_MAP` adds
  pinned aliases: `chatgpt -> gpt-4o-mini-2024-07-18`, `gpt41 -> gpt-4.1-2025-04-14`,
  `gpt35 -> gpt-3.5-turbo` (for the capability sweep). No change to agent logic.
- **`run.py`** — (i) widens the `model_name` whitelist to accept the sweep
  backbones; (ii) adds the ablation methods `base_cot_strong`, `base_cot_role`,
  `base_cot_budget` to the argument choices; (iii) optional per-question usage
  logging via `COST_LOG` (cost accounting only). Control flow is unchanged.
- **`utils.py`** — adds the single-agent baseline ablation branches used in the
  2x2 analysis:
  - `base_cot` (unchanged upstream): empty system role, 300-token cap.
  - `base_cot_role`: expert role, **300**-token cap.
  - `base_cot_budget`: empty role, **700**-token cap.
  - `base_cot_strong`: expert role, **700**-token cap.
  Only the system role string and `max_tokens` differ; the user prompt, answer
  extraction, and all multi-agent methods are untouched.
- **`prompt_generator.py`** — included unmodified for reference (defines the CoT
  prompt used by the baselines).

## MDAgents (exploratory — not part of the primary claims)

MDAgents could not be ported faithfully; the following changes affect behavior and
are the reason its results are reported only as exploratory:

- **`utils.py`** — routes calls through the cached client; **unifies all agent
  roles to a single pinned model** (the release mixes `gpt-3.5`/`gpt-4o-mini`);
  adds an **empty-expert-group fallback** in `Group.interact` (returns an empty
  contribution instead of crashing).
- **`main.py`** — per-question `try/except` with incremental saving so one failed
  question does not abort the run; optional `COST_LOG` usage logging; fixed a
  Python-3.12 f-string in the moderator print.

These MDAgents guards alter the advanced-difficulty path (it collapses to far
fewer calls than intended), so its measured accuracy is unstable and is excluded
from the paper's primary conclusions.
