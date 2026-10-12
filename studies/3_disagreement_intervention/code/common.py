"""Shared Phase III helpers: frozen prompt rendering, E3 extraction, I/O. No gold is ever read here."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "common"))
from llm import chat_meta, sample_n_meta   # noqa: E402,F401

MODEL = "gpt-4o-mini-2024-07-18"           # placeholder name; TMC_MODEL_OVERRIDE selects the local model
DATASETS = f"{ROOT}/data/splits"


def prompt(name, **kw):
    """Render a frozen prompt file (sections '### system|user|assistant') into chat messages.
    Placeholders are replaced literally ({name}); other braces (e.g. JSON) are left untouched."""
    msgs, role, buf = [], None, []
    for line in open(f"{HERE}/prompts/{name}.txt").read().split("\n"):
        if line.startswith("### "):
            if role:
                msgs.append({"role": role, "content": "\n".join(buf).strip("\n")})
            role, buf = line[4:].strip(), []
        else:
            buf.append(line)
    msgs.append({"role": role, "content": "\n".join(buf).strip("\n")})
    for m in msgs:
        for k, v in kw.items():
            m["content"] = m["content"].replace("{" + k + "}", str(v))
    return msgs


def opts(q):
    return "\n".join(f"{k}. {v}" for k, v in q["options"].items())


def jchoice(t):
    try:
        return str(json.loads(t).get("choice", "")).strip()[:1].upper()
    except Exception:
        return ""


def e3(reasoning, q, tag, tail=False):
    """E3 extractor (extract_answer.txt). First 2,500 chars of the reasoning; last 2,500 for revisions."""
    r = (reasoning or "")
    r = r[-2500:] if tail else r[:2500]
    rec = chat_meta(prompt("extract_answer", reasoning_first_2500_chars=r, options=opts(q)),
                    json_mode=True, model=MODEL, temperature=0, max_tokens=20, nonce=tag)
    a = jchoice(rec["response"])
    return (a if a in q["options"] else ""), usage(rec.get("usage"))


def usage(u):
    u = u or {}
    return dict(input_tokens=int(u.get("prompt_tokens", 0)), output_tokens=int(u.get("completion_tokens", 0)))


def add(*us):
    return dict(input_tokens=sum(u["input_tokens"] for u in us), output_tokens=sum(u["output_tokens"] for u in us))


def load_split(split):
    return [json.loads(l) for l in open(f"{DATASETS}/MedMCQA_p3{split}/test.jsonl")]


def jl(p):
    return [json.loads(l) for l in open(p)]


def write_jl(p, rows):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
