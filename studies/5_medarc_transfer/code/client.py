"""Seeded, cached chat client for Phase V (local vLLM via TMC_BASE_URL / TMC_MODEL_OVERRIDE). Cache key = full payload."""
import hashlib
import json
import os
import time

from openai import OpenAI

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
CACHE = os.environ.get("P5_CACHE", f"{ROOT}/cache/phase5")
_client = None


def chat(messages, temperature, max_tokens, seed, n=1):
    global _client
    model = os.environ["TMC_MODEL_OVERRIDE"]
    payload = dict(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, seed=seed, n=n)
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:32]
    path = f"{CACHE}/{key}.json"
    if os.path.exists(path):
        return json.load(open(path))
    if _client is None:
        _client = OpenAI(base_url=os.environ["TMC_BASE_URL"], api_key="EMPTY", timeout=900)
    for attempt in range(6):
        try:
            r = _client.chat.completions.create(**payload)
            break
        except Exception:
            if attempt == 5:
                raise
            time.sleep(2 * 2 ** attempt)
    rec = dict(texts=[c.message.content for c in r.choices], finish=[c.finish_reason for c in r.choices],
               input_tokens=r.usage.prompt_tokens, output_tokens=r.usage.completion_tokens)
    os.makedirs(CACHE, exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    json.dump(rec, open(tmp, "w")); os.replace(tmp, path)
    return rec
