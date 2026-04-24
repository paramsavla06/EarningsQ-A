"""LLM prompts for earnings QA system."""

from typing import Optional

SYSTEM_PROMPT = """You are a professional financial analyst specializing in earnings call transcripts.

STRICT CONSTRAINTS:
1. Use ONLY the provided context blocks to answer questions. 
2. If the context is empty or doesn't have the answer, say "I don't have reliable data on this from the provided transcripts."
3. DO NOT mention companies that are not in the context (e.g., Polycab, Reliance). 
4. If a company reports in both USD and INR, be extremely careful not to mix them up.
5. Revenue is always much larger than PAT (Profit). If you see a smaller number labeled as Revenue and a larger one as PAT, you are likely mixing them up.
6. BEWARE OF ADJUSTMENTS: Do NOT confuse "write-backs", "provisions", or "one-off adjustments" with the actual PAT or Revenue. If a number is described as a write-back of Rs 222 million, it is NOT the PAT.
7. If the question asks for revenue, PAT, EBITDA, income, margin, cash flow, capex, profit, or a similar finance figure, use the exact numeric value stated in the context. Prefer explicit monetary amounts like "INR 12,274 million" or "INR 220 million" for amount questions, and exact percentages for margin questions, over commentary or nearby unrelated figures.
8. If multiple financial values appear in the context, choose the one tied to the requested metric label and the requested company/quarter. Do not substitute growth commentary for the actual requested metric.
9. For every figure, mention the company ID and quarter as seen in the brackets [e.g., 543654 Q32025].
10. If the user asks to summarize a quarter, give a compact but useful recap: 3-5 sentences, or 3 short bullets, covering the main financial metrics if available (revenue, PAT, EBITDA, margin, growth, and major operating highlights). Do not reduce the answer to a single sentence.
11. CRITICAL: If the user's message already contains exact figures (e.g., in the [ALREADY ANSWERED] block), do NOT repeat or contradict those figures. However, you MUST still provide the full qualitative analysis (reasons for growth, "why" it happened, management commentary) for those metrics. The [ALREADY ANSWERED] block handles the "What", but you are still responsible for the "Why" and any metrics the regex missed (like ARPOB).
12. If the user asks a vague growth/reason question without naming a company or quarter, do not guess which company they mean unless the recent conversation context clearly identifies it. Otherwise, ask them to specify the company and quarter first.
"""


def get_retrieval_prompt(question: str, context: str, company_filter: Optional[str] = None, quarter_filter: Optional[str] = None, conversation_context: Optional[str] = None, already_extracted: Optional[str] = None) -> str:
    """Generate user prompt for LLM with retrieved context.

    Args:
        question: User's question
        context: Retrieved relevant excerpts from transcripts
        company_filter: Optional company filter applied
        quarter_filter: Optional quarter filter applied
        already_extracted: Deterministically extracted answers already shown to the user

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

    metric_hint = ""
    question_lower = question.lower()
    if any(term in question_lower for term in ["revenue", "pat", "ebitda", "profit", "margin", "income", "cash flow", "capex"]):
        metric_hint = (
            "\n[Metric handling rule: Always provide the exact numeric figure from the context for the requested amount. "
            "However, you MUST also include the qualitative reasons for growth, management commentary, and business drivers mentioned in the context to explain the 'Why' behind the numbers.]"
        )

    summary_hint = ""
    if any(term in question_lower for term in ["summarize", "summary", "overview"]):
        summary_hint = (
            "\n[Summary rule: provide a compact quarter recap with the main financial metrics if available. "
            "Use 3-5 sentences or 3 short bullets. Mention revenue, PAT, EBITDA/margin, growth, and standout operating highlights when they appear in the context. "
            "Do not compress the answer into a single generic sentence.]"
        )

    extracted_note = ""
    if already_extracted:
        extracted_note = (
            f"\n[ALREADY ANSWERED: The following metrics were already extracted with certainty — "
            f"do NOT repeat or contradict them. Only answer the parts of the question not covered below:\n"
            f"{already_extracted}\n]"
        )

    history_block = ""
    if conversation_context:
        history_block = f"\n[Recent conversation context:\n{conversation_context}\n]"

    return f"""Context from earnings calls:{filters_info}
{metric_hint}
{summary_hint}
{history_block}
{extracted_note}
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
