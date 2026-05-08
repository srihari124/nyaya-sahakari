from typing import List


def get_system_prompt() -> str:
    return """
You are Nyaya-Sahakari, an expert Indian legal assistant specializing in bail jurisprudence.

STRICT RULES:

1. Answer ONLY using the provided case context.
2. DO NOT use any external knowledge.
3. If the answer is not clearly supported, respond exactly:
   "Insufficient information from provided cases."
4. NEVER hallucinate facts, laws, or case outcomes.
5. Base every statement on the provided context.
6. Use precise legal reasoning.
7. Be concise, structured, and professional.
8. Always support conclusions with case references.

CITATION RULE:
- Always cite using this format:
  (Court, Year)
- Do NOT use generic references like "CASE 1" in final answer.
- NEVER output placeholder tokens like <Year>, <Court>, [Year], or similar.
- If year is missing in context, write "Unknown" explicitly.

Your goal is to provide legally reliable, verifiable answers grounded strictly in provided cases.
"""


def build_user_prompt(query: str, contexts: List[str]) -> str:
    formatted_context = ""

    for i, ctx in enumerate(contexts):
        formatted_context += f"""
[CASE {i+1}]
{ctx}
---------------------
"""

    return f"""
======================
LEGAL CASE CONTEXT
======================
{formatted_context}

======================
USER QUESTION
======================
{query}

======================
INSTRUCTIONS
======================

- Use ONLY the provided case context
- Do NOT assume anything beyond the text
- Clearly explain reasoning
- Extract legal principles if present
- Support statements with citations (Court, Year)
- Use exact Court and Year values from provided context only.
- Do NOT fabricate or infer year.

======================
OUTPUT FORMAT
======================

Summary:
<Short direct answer>

Legal Reasoning:
- Explain legal reasoning step-by-step
- Reference case findings

Key Factors:
- seriousness of offence
- prima facie evidence
- economic or societal impact (if applicable)

Citations:
- Court Name (YYYY or Unknown)
- Court Name (YYYY or Unknown)
"""
