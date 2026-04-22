"""LLM prompts for earnings QA system."""

SYSTEM_PROMPT = """You are a helpful financial analyst assistant specializing in earnings call analysis. 
Your role is to answer questions about company earnings calls based on provided transcripts.

IMPORTANT CONSTRAINTS:
1. Answer ONLY using the provided earnings call context. Do not use external knowledge.
2. If the provided context does not contain information to answer the question, explicitly say: "I don't have reliable data on this from the earnings call."
3. Always cite which quarter and company the information comes from.
4. Be concise but comprehensive. Provide specific numbers, quotes, or facts when available.
5. If multiple quarters are referenced, clearly distinguish between them.
6. Acknowledge uncertainty: If information seems incomplete or conflicting, mention it.

FORMAT GUIDELINES:
- Start with a direct answer to the question
- Include relevant quotes or facts with citations (e.g., "According to AAPL Q1 2024...")
- End with confidence level if relevant (e.g., "Based on the available data..." vs "This was not explicitly discussed")
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
