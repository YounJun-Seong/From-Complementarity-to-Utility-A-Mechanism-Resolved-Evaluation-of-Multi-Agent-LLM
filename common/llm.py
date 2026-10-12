"""Deterministic, cached LLM client for the Tier-2 agents (DESIGN_tier2.md §1).

Pinned snapshot + temperature=0 + on-disk response cache keyed by a hash of
(model, temperature, messages, response_format). Re-runs are deterministic, cheap,
and auditable (every prompt+response is a file the paper can release).
"""
import hashlib
import json
import os
import random
import re
import time

CACHE_DIR = os.environ.get("TMC_CACHE", "data/llm_cache")   # env.sh pins one shared cache
MODEL = "gpt-4o-mini-2024-07-18"      # pinned snapshot (reproducibility)
TEMP = 0.0
_client = None

# --- lightweight usage accumulator (for cost/token accounting) ---
USAGE = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}


# --- API error / retry counters (per process; callers snapshot deltas per unit of work) ---
STATS = {"requests": 0, "retries": 0, "rate_limit": 0, "timeout": 0, "api_error": 0}


def reset_usage():
    USAGE.update(calls=0, prompt_tokens=0, completion_tokens=0)


def _accumulate(u):
    if not u:
        return
    USAGE["calls"] += 1
    USAGE["prompt_tokens"] += u.get("prompt_tokens", 0)
    USAGE["completion_tokens"] += u.get("completion_tokens", 0)


def _load_env(path="../.env"):
    if os.environ.get("OPENAI_API_KEY"):
        return
    for p in (path, ".env", "../.env", "<HOME>/.env"):
        if os.path.exists(p):
            for line in open(p):
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip())
            break


def _key(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]


def _retry_create(kw, max_tries=15):
    """Call the API with exponential backoff on rate-limit / transient errors.
    Honors the 'try again in Xs' hint OpenAI returns for RPD/RPM limits."""
    from openai import APIError, APITimeoutError, RateLimitError
    for attempt in range(max_tries):
        try:
            STATS["requests"] += 1
            return _client.chat.completions.create(**kw)
        except (RateLimitError, APITimeoutError, APIError) as e:
            STATS["rate_limit" if isinstance(e, RateLimitError) else
                  "timeout" if isinstance(e, APITimeoutError) else "api_error"] += 1
            if attempt == max_tries - 1:
                raise
            STATS["retries"] += 1
            wait = None
            m = re.search(r"try again in ([\d.]+)s", str(e))
            if m:
                wait = float(m.group(1)) + 0.5
            if wait is None:
                wait = min(60.0, 2.0 ** attempt) + random.uniform(0, 1)
            time.sleep(wait)


def chat(messages, json_mode=True, model=MODEL, temperature=TEMP, max_tokens=800):
    """Cached chat completion. Returns the assistant string (JSON text if json_mode)."""
    return chat_meta(messages, json_mode, model, temperature, max_tokens)["response"]


def chat_meta(messages, json_mode=True, model=MODEL, temperature=TEMP, max_tokens=800,
              schema=None, nonce=None):
    """Like chat() but returns the full cache record: response, usage, finish_reason.
    finish_reason is None for records cached before it was logged. `schema` (a JSON
    schema dict) requests strict structured output instead of plain json_mode.
    temperature=None / max_tokens=None leave the API defaults (for frameworks that set
    neither). `nonce` only enters the cache key, so repeated stochastic calls with the same
    messages stay distinct but resumable."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    requested = model
    if os.environ.get("TMC_MODEL_OVERRIDE"):   # local backend: every call goes to this model
        model = os.environ["TMC_MODEL_OVERRIDE"]
    payload = {"model": model, "temperature": temperature, "messages": messages,
               "json_mode": json_mode, "max_tokens": max_tokens}
    if schema is not None:                 # only keyed when used: old cache keys unchanged
        payload["schema"] = schema
    if nonce is None and os.environ.get("TMC_NONCE_NS"):   # per-run namespace for whole frameworks
        nonce = os.environ["TMC_NONCE_NS"]
    if nonce is not None:
        payload["nonce"] = nonce
    cpath = os.path.join(CACHE_DIR, _key(payload) + ".json")
    if os.path.exists(cpath):
        rec = json.load(open(cpath))
        _accumulate(rec.get("usage"))
        rec.setdefault("finish_reason", None)
        return rec

    _init_client()
    kw = {"model": model, "messages": messages}
    if temperature is not None:
        kw["temperature"] = temperature
    if max_tokens is not None:
        kw["max_tokens"] = max_tokens
    if schema is not None:
        kw["response_format"] = {"type": "json_schema",
                                 "json_schema": {"name": "answer", "strict": True, "schema": schema}}
    elif json_mode:
        kw["response_format"] = {"type": "json_object"}
    resp = _retry_create(kw)
    text = resp.choices[0].message.content
    usage = resp.usage.model_dump()
    _accumulate(usage)
    rec = {"request": payload, "response": text, "usage": usage,
           "finish_reason": resp.choices[0].finish_reason,
           "served_model": resp.model, "system_fingerprint": resp.system_fingerprint,
           "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "run_id": os.environ.get("TMC_RUN_ID"), "prompt_hash": _key(payload),
           "requested_model": requested, "resolved_model": model,
           "backend": os.environ.get("TMC_BASE_URL") or "openai"}
    json.dump(rec, open(cpath, "w"), indent=2)
    return rec


def _init_client():
    """OpenAI API by default; TMC_BASE_URL (e.g. http://127.0.0.1:8000/v1) = OpenAI-compatible
    local server (vLLM). Frameworks keep calling this module unchanged."""
    global _client
    if _client is None:
        from openai import OpenAI
        if os.environ.get("TMC_BASE_URL"):
            _client = OpenAI(base_url=os.environ["TMC_BASE_URL"], api_key="EMPTY", timeout=600)
        else:
            _load_env()
            _client = OpenAI()


def chat_json(messages, **kw):
    return json.loads(chat(messages, json_mode=True, **kw))


def sample_n(messages, k, model=MODEL, temperature=0.7, max_tokens=800, seed=0):
    return sample_n_meta(messages, k, model, temperature, max_tokens, seed)[0]


def sample_n_meta(messages, k, model=MODEL, temperature=0.7, max_tokens=800, seed=0):
    """sample_n that also returns this call's usage (thread-safe accounting)."""
    """Return k sampled completions (for self-consistency / debate diversity).
    One API request with n=k at temperature>0; cached by (messages,temp,k,seed)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.environ.get("TMC_MODEL_OVERRIDE"):
        model = os.environ["TMC_MODEL_OVERRIDE"]
    payload = {"model": model, "temperature": temperature, "messages": messages,
               "n": k, "max_tokens": max_tokens, "seed": seed}
    cpath = os.path.join(CACHE_DIR, _key(payload) + ".json")
    if os.path.exists(cpath):
        rec = json.load(open(cpath))
        _accumulate(rec.get("usage"))
        return rec["responses"], rec.get("usage") or {}
    _init_client()
    kw = {"model": model, "temperature": temperature, "messages": messages,
          "n": k, "max_tokens": max_tokens, "seed": seed}
    resp = _retry_create(kw)
    texts = [c.message.content for c in resp.choices]
    usage = resp.usage.model_dump()
    _accumulate(usage)
    json.dump({"request": payload, "responses": texts, "usage": usage,
               "finish_reasons": [c.finish_reason for c in resp.choices], "resolved_model": model,
               "backend": os.environ.get("TMC_BASE_URL") or "openai",
               "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, open(cpath, "w"))
    return texts, usage
