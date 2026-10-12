"""Verbatim strings and logic from asdmiao/MedARC@3b6b5a6, yes_or_no-QA (PubMedQA). Only the client is replaced.
Sources: gen_pubmedqa_deepseek.py (MedARC, homogeneous), gen_pubmedqa_deepseek_danlun.py (zero-shot),
gen_pubmedqa_deepseek_danluncot.py (CoT), eval_pubmedqa_agents_confidence.py, eval_pubmedqa_zeroshot_danlun.py."""
import re

import numpy as np

AGENTS, ROUNDS = 3, 3          # gen_pubmedqa_deepseek.py: agents = 3, rounds = 3 (1 initial + 2 debate = paper DR2)


def initial_prompt(context_reference, question):              # gen_pubmedqa_deepseek.py, agent_contexts
    return """You are a skilled medical expert. Based on the following reference context: {}. what is the most appropriate answer to the following question: {} Answer with ``yes'', ``no'', or ``maybe'', and place it in \\boxed{{}}, at the end of your response. In addition to your answer, please provide a confidence score (0%-100%) that reflects how certain you are in the correctness of your response. Provide a logical explanation and reasoning before giving your final answer.""".format(context_reference, question)


def zeroshot_prompt(context_reference, question):             # gen_pubmedqa_deepseek_danlun.py
    return """You are a skilled medical expert. Based on the following reference context: {}. what is the most appropriate answer to the following question: {} Answer with ``yes'', ``no'', or ``maybe'', and place it in \\boxed{{}}, at the end of your response.""".format(context_reference, question)


def cot_prompt(context_reference, question):                  # gen_pubmedqa_deepseek_danluncot.py
    return """You are a skilled medical expert. Based on the following reference context: {}. what is the most appropriate answer to the following question: {} Answer with ``yes'', ``no'', or ``maybe'', and place it in \\boxed{{}}, at the end of your response. Let's think step by step.""".format(context_reference, question)


def summary_messages(responses):                               # generate_summary
    content = "Provide a clear summary of key similarities and differences in these responses. Keep it concise."
    for i, response in enumerate(responses, 1):
        content += f"Response {i}: {response}\n"
    return [{"role": "system", "content": "You are an expert at identifying consensus and disagreement."},
            {"role": "user", "content": content}]


def other_responses(agents, idx):                              # construct_message (part before the summary call)
    return [agent[idx]["content"] for agent in agents if len(agent) > idx]


def debate_message(summary, question):                         # construct_message
    prefix = f"Summary of other agents' responses:\n{summary}\n\nPlease re-analyze the question using these provided insights. Provide your updated answer to the question: {question}\nPresent your reasoning before final answer."
    return {"role": "user", "content": prefix}


# ---------------- released evaluation behaviour (eval_pubmedqa_agents_confidence.py) ----------------
def parse_answer(input_str):
    input_str_lower = input_str.lower()
    if "yes" in input_str_lower:
        return "yes"
    elif "no" in input_str_lower:
        return "no"
    elif "maybe" in input_str_lower:
        return "maybe"
    else:
        return None


def extract_confidence(response):
    confidence_match = re.search(r'confidence\s*[:\-]?\s*(\d+(?:\.\d+)?)', response, re.IGNORECASE)
    if confidence_match:
        try:
            return float(confidence_match.group(1))
        except ValueError:
            return 0.0
    return 0.0


def released_select(agent_contexts):
    """Verbatim selection: all assistant turns of all agents; first \\boxed{...} (greedy-to-first-'}'), else the
    agent's *last* assistant turn (as released); highest released confidence via argsort[::-1][0]."""
    pred_solutions, confidences = [], []
    for response in agent_contexts:
        assistant_responses = [r['content'] for r in response if r['role'] == 'assistant']
        for pred_solution in assistant_responses:
            confidences.append(extract_confidence(pred_solution))
            match = re.search(r"\\boxed{([^}]+)}", pred_solution)
            if match:
                pred_solutions.append(match.group(1))
            else:
                pred_solutions.append(assistant_responses[-1])
    if len(pred_solutions) >= 2 and len(confidences) >= 2:
        selected = pred_solutions[np.argsort(confidences)[::-1][0]]
    elif len(pred_solutions) == 1:
        selected = pred_solutions[0]
    else:
        selected = None
    return parse_answer(selected) if selected else None


def released_zeroshot(content):                                # eval_pubmedqa_zeroshot_danlun.py
    match = re.search(r"\\boxed{(.*?)}", content)
    s = match.group(1) if match else content
    s = s.lower()
    return "yes" if "yes" in s else "no" if "no" in s else "maybe" if "maybe" in s else None
