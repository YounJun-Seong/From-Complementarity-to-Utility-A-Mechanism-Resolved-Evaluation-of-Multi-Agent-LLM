"""Checks that released.py reproduces the MedARC@3b6b5a6 prompt strings byte-for-byte (AST string constants of the
original files, rendered with the same .format arguments). Usage: python test_released_strings.py <MedARC repo dir>"""
import ast
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import released as R   # noqa: E402


def consts(path):
    return [n.value for n in ast.walk(ast.parse(open(path).read())) if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def main():
    d = os.path.join(sys.argv[1], "yes_or_no-QA")
    ctx, q = ["ctx one.", "ctx two"], "Is X associated with Y?"
    gen = consts(f"{d}/gen_pubmedqa_deepseek.py")
    init = [c for c in gen if c.startswith("You are a skilled medical expert")]
    zs = [c for c in consts(f"{d}/gen_pubmedqa_deepseek_danlun.py") if c.startswith("You are a skilled medical expert")]
    cot = [c for c in consts(f"{d}/gen_pubmedqa_deepseek_danluncot.py") if c.startswith("You are a skilled medical expert")]
    checks = {
        "initial": init and init[0].format(ctx, q) == R.initial_prompt(ctx, q),
        "zero-shot": zs and zs[0].format(ctx, q) == R.zeroshot_prompt(ctx, q),
        "cot": cot and cot[0].format(ctx, q) == R.cot_prompt(ctx, q),
        "summary system": "You are an expert at identifying consensus and disagreement." in gen,
        "summary instruction": "Provide a clear summary of key similarities and differences in these responses. Keep it concise." in gen,
        "debate prefix": "\n\nPlease re-analyze the question using these provided insights. Provide your updated answer to the question: " in gen
                         and "\nPresent your reasoning before final answer." in gen,
    }
    for k, v in checks.items():
        print(f"{k:20s} {'ok' if v else 'MISMATCH'}")
    print("RELEASED STRINGS", "PASS" if all(checks.values()) else "FAIL")


if __name__ == "__main__":
    main()
