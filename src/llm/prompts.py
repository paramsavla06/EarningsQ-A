"""LLM prompts for earnings QA system."""

SYSTEM_PROMPT = """You are a professional financial analyst specializing in earnings call transcripts.

STRICT CONSTRAINTS:
1. Use ONLY the provided context blocks to answer questions. 
2. If the context is empty or doesn't have the answer, say "I don't have reliable data on this from the provided transcripts."
3. DO NOT mention companies that are not in the context (e.g., Polycab, Reliance). 
4. If the user asks about "all companies" and you only see data for one, say: "I only have data for [Company X] in the provided transcripts. Information for others is missing."
5. For every figure, mention the company ID and quarter as seen in the brackets [e.g., 543654 Q32025].

FORMAT:
- Direct Answer
- Detailed metrics with citations
- List of missing data if applicable
"""

def get_retrieval_prompt(question: str, context: str, company_filter: str = None, quarter_filter: str = None) -> str:
    """Generate user prompt for LLM with retrieved context.
    
    Args:
        question: User's question
        context: Retrieved relevant excerpts from transcripts
        company_filter: Optional company filter applied
        quarter_filter: Optional quarter filter applied
    
    Returns:
        Formatted user prompt
    """
    filters_info = ""
    if company_filter or quarter_filter:
        filters = []
        if company_filter:
            filters.append(f"Company: {company_filter}")
        if quarter_filter:
            filters.append(f"Quarter: {quarter_filter}")
        filters_info = f"\n[Search filters: {', '.join(filters)}]"
    
    return f"""Context from earnings calls:{filters_info}
---
{context}
---

User Question: {question}

Please answer based on the context above."""


def get_scope_check_prompt(question: str) -> str:
    """Generate prompt for checking if question is in-scope."""
    return f"""Given this question: "{question}"

Is this question about company earnings calls, financial performance, guidance, or related earnings-call topics?
Answer with ONLY 'YES' or 'NO'."""
