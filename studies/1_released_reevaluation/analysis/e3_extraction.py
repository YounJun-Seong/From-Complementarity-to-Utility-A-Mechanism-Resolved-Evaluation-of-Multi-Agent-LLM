"""E3 — is the weak base_cot baseline a FAILURE OF ANSWER EXTRACTION rather than
reasoning? Take MedAgents' base_cot raw CoT generations and re-extract the answer
robustly (regex + a clean extractor LLM call). If accuracy jumps, the gap to the
multi-agent system is largely an extraction artifact, not a reasoning gain."""
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from llm import chat

L = [json.loads(l) for l in open("external/MedAgents/outputs/MedQA/chatgpt-base_cot")]
MODEL = "gpt-4o-mini-2024-07-18"


def regex_extract(raw, options):
    # last explicit "answer is X" / "(X)" / final letter
    for pat in [r"answer\s*(?:is|:)?\s*\(?([ABCD])\)?", r"\b([ABCD])\b\s*$",
                r"\(([ABCD])\)"]:
        m = list(re.finditer(pat, raw, re.IGNORECASE))
        if m:
            return m[-1].group(1).upper()
    return ""


def llm_extract(raw, options):
    opts = "\n".join(f"{k}. {v}" for k, v in options.items())
    try:
        o = json.loads(chat([{"role": "system", "content":
              'Extract the final chosen answer from the analysis. Respond ONLY as JSON: {"choice":"A|B|C|D"}'},
              {"role": "user", "content": f"Options:\n{opts}\n\nAnalysis:\n{raw[:3000]}"}],
              json_mode=True, model=MODEL, max_tokens=20))
        return str(o.get("choice", "")).strip()[:1].upper()
    except Exception:
        return ""


def main():
    their = sum(x["pred_answer"] == x["gold_answer"] for x in L) / len(L)

    def re_ex(x):
        raw = x.get("raw_output", "")
        return (regex_extract(raw, x["options"]) == x["gold_answer"],
                llm_extract(raw, x["options"]) == x["gold_answer"])
    with ThreadPoolExecutor(max_workers=8) as ex:
        r = list(ex.map(re_ex, L))
    regex_acc = sum(a for a, _ in r) / len(r)
    llm_acc = sum(b for _, b in r) / len(r)
    print(f"E3 base_cot re-extraction (n={len(L)}):")
    print(f"  their extraction (as reported): {their:.3f}")
    print(f"  robust REGEX re-extraction:     {regex_acc:.3f}")
    print(f"  robust LLM re-extraction:       {llm_acc:.3f}")
    print(f"  extraction-only gain: +{llm_acc-their:.3f} (from SAME CoT generations)")
    print("E3_DONE")


if __name__ == "__main__":
    main()
