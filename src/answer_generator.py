"""Generate grounded natural-language answers using OpenAI."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from src.data_loader import PROJECT_ROOT


# Cost-effective model for final answer generation.
ANSWER_MODEL = "gpt-5-mini"


def get_openai_client() -> OpenAI:
    """
    Create an OpenAI client using the local environment configuration.
    """

    load_dotenv(PROJECT_ROOT / ".env", override=True)

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required. "
            "Add it to your local .env file."
        )

    return OpenAI(api_key=api_key)


def generate_answer(
    question: str,
    structured_facts: dict[str, Any] | None,
    retrieved_evidence: dict[str, Any],
    reliability_assessment: dict[str, Any] | None = None,
) -> str:
    """
    Generate a grounded answer using ONLY the supplied structured facts,
    retrieved document evidence, and reliability assessment.

    The reliability assessment determines whether:
    - the available sources are trustworthy enough for a confident answer
    - source conflicts were resolved using source priority
    - human review is required
    - uncertainty should be explicitly communicated
    """

    client = get_openai_client()

    prompt = f"""
You are ParcelPilot AI, a grounded customer support assistant.

Answer the user's question using ONLY the structured facts, document
evidence, and trust/reliability assessment provided below.

IMPORTANT GROUNDING RULES:

1. Do not invent facts.
2. Do not use knowledge outside the supplied context.
3. If the evidence is insufficient, clearly say that the available
   evidence is insufficient.
4. Treat customer-specific agreements as potentially higher priority
   than global policies when the supplied evidence indicates they apply.
5. Do not use deprecated documents for current requests unless the user
   explicitly asks about historical information.
6. Clearly distinguish structured facts from document rules when useful.
7. Respect the authority order provided in the retrieved evidence and
   reliability assessment.
8. Do not claim that a customer agreement applies unless it is present
   in the retrieved evidence.
9. If structured data shows an account mismatch, clearly state the
   mismatch.
10. Mention the relevant source document names and page numbers.

TRUST AND RELIABILITY RULES:

11. The reliability assessment is authoritative for determining whether
    it is safe to answer confidently.

12. If `human_review_required` is True:
    - Do NOT provide a definitive decision when the conflict or uncertainty
      affects the answer.
    - Clearly explain that the available authoritative information is
      conflicting or insufficient.
    - State that human review is required.
    - Do not guess which rule should apply.
    - You may still provide uncontested factual information if it is useful.

13. If `safe_to_answer_confidently` is False:
    - Do not present the answer as certain.
    - Clearly communicate the uncertainty.
    - Do not invent missing policy conditions.

14. If `conflict_detected` is True and
    `human_review_required` is False:
    - A conflict was resolved using the defined source-priority rules.
    - Follow the higher-priority source.
    - Do not treat the lower-priority conflicting source as governing.
    - Briefly explain the applicable rule when relevant.
    - Do not expose internal source-priority numbers.

15. Source authority should be interpreted in this order:

    - Account-specific contract/agreement
    - Official current policy
    - Structured operational data
    - Other retrieved supporting documents
    - Historical support resolutions

16. Historical support resolutions are non-authoritative.
    They must never override a current policy or an account-specific
    agreement.

17. Do not treat a previous agent's answer as proof that the same answer
    is correct for the current request.

18. If the reliability assessment indicates a historical resolution risk,
    do not rely on that historical resolution unless it is independently
    supported by a higher-authority source.

19. If the reliability assessment indicates confidence is:
    - high: answer normally and confidently.
    - medium: answer based on the resolved governing evidence, without
      overstating certainty.
    - low: clearly communicate uncertainty and recommend human review
      when required.

20. Never expose internal implementation terms such as:
    "reliability_assessment",
    "source_priority",
    "conflict_detected",
    "safe_to_answer_confidently",
    "structured facts dictionary",
    "retrieved evidence object",
    "chunk",
    "metadata",
    "embedding",
    or "vector search".

IMPORTANT USER-FRIENDLY FORMATTING RULES:

21. Never expose raw Python, Pandas, database, JSON, or internal
    technical values directly to the user.

22. Translate technical values into natural language. For example:
    - NaT -> "not available" or use the meaning implied by the field.
    - None or null -> "not available" or use the meaning implied by
      the field.
    - NaN -> "not available".
    - pickup_actual_at = NaT -> "the shipment has not yet been picked up"
      when supported by the order status and context.
    - Boolean fields such as True/False -> natural language such as
      "the carrier was at fault" or "the customer was not at fault".

23. Do not show internal field names unless they are genuinely useful.

    For example, do NOT say:
        order.status = BOOKED
        pickup_actual_at = NaT
        customer_fault = False

    Instead say:
        "The order is currently BOOKED and has not yet been picked up."
        "The carrier was at fault, and the customer was not at fault."

24. Do not mention internal implementation details such as:
    "structured facts",
    "retrieved evidence",
    "chunk_text",
    "metadata",
    "embedding",
    "vector search",
    "source priority",
    or "reliability assessment".

25. Use clear, natural language suitable for a ParcelPilot customer or
    support user.

26. Keep answers concise, direct, and consistently structured.

27. Start directly with the answer. Do not use unnecessary labels such as:
    "Short answer",
    "Direct answer",
    "Structured Facts",
    "Document Evidence",
    "Authority",
    "Conclusion",
    "Account Match",
    "Reliability Assessment",
    or "Conflict Detection".

28. Include only information that directly helps answer the user's
    question.

29. Explain the answer briefly using only the essential facts and the
    applicable governing rule.

30. Use this format when appropriate:

    [Direct answer]

    Reason:
    - [Essential fact and applicable rule]

    [Include a fee, credit amount, response target, comparison, or
    workaround only when relevant.]

    Sources:
    - [Relevant document name], page [number]

31. If human review is required, use a user-friendly format such as:

    I can't give a definitive answer because the available governing
    information conflicts.

    Reason:
    - [Brief explanation of the conflict or missing information]
    - A ParcelPilot support or operations reviewer should determine
      which rule applies.

    Sources:
    - [Relevant document name], page [number]

32. Do not repeat the same information in multiple sections.

33. Usually keep the answer under 120 words unless the user explicitly
    asks for a detailed explanation or comparison.

34. For comparison questions, present the comparison directly and clearly,
    preferably using short bullet points or a compact table.

35. Do not add information that is not supported by the supplied context.

36. If the answer depends on a conflict resolved by a customer-specific
    agreement, make the result clear without mentioning internal priority
    logic. For example:
    "The account agreement applies to this order and overrides the
    general cancellation rule."

37. Never say a source "overrides" another source unless the supplied
    evidence and reliability assessment support that conclusion.

USER QUESTION:
{question}

STRUCTURED FACTS:
{structured_facts}

RETRIEVED DOCUMENT EVIDENCE:
{retrieved_evidence}

TRUST AND RELIABILITY ASSESSMENT:
{reliability_assessment}
"""

    response = client.responses.create(
        model=ANSWER_MODEL,
        input=prompt,
    )

    return response.output_text or "Unable to generate an answer."


def main() -> None:
    """
    Run standalone answer-generation tests.
    """

    # -------------------------------------------------------------
    # Test 1: High-confidence answer
    # -------------------------------------------------------------

    print("=" * 70)
    print("TEST 1: HIGH-CONFIDENCE ANSWER")
    print("=" * 70)

    question = "Can ORD-1001 be cancelled without a fee?"

    structured_facts = {
        "order": {
            "order_id": "ORD-1001",
            "account_id": "ACCT-001",
            "status": "BOOKED",
            "pickup_actual_at": "NaT",
        },
        "cancellation_timing": {
            "minutes_after_booking": 120,
        },
    }

    retrieved_evidence = {
        "governing_evidence": [
            {
                "source_file": (
                    "03_Cancellation_and_Service_Credit_SOP_v4.pdf"
                ),
                "page_number": 1,
                "status": "current",
                "chunk_text": (
                    "BOOKED shipments may be cancelled without a fee within "
                    "30 minutes. After 30 minutes, INR 250 applies unless a "
                    "customer agreement explicitly waives the fee."
                ),
            }
        ],
        "agreement_evidence": [
            {
                "source_file": (
                    "05_Northstar_Logistics_Enterprise_Agreement.pdf"
                ),
                "page_number": 1,
                "status": "active",
                "chunk_text": (
                    "Northstar may cancel any BOOKED shipment before pickup "
                    "with no cancellation fee, regardless of how long ago "
                    "the shipment was booked."
                ),
            }
        ],
        "deprecated_evidence": [],
    }

    reliability_assessment = {
        "conflict_detected": False,
        "confidence": "high",
        "safe_to_answer_confidently": True,
        "human_review_required": False,
        "message": (
            "The answer is supported by an account-specific agreement "
            "and relevant structured operational data."
        ),
    }

    answer = generate_answer(
        question=question,
        structured_facts=structured_facts,
        retrieved_evidence=retrieved_evidence,
        reliability_assessment=reliability_assessment,
    )

    print("\nQUESTION:")
    print(question)

    print("\nANSWER:")
    print(answer)

    # -------------------------------------------------------------
    # Test 2: Conflict resolved by source authority
    # -------------------------------------------------------------

    print("\n" + "=" * 70)
    print("TEST 2: RESOLVED CONFLICT")
    print("=" * 70)

    resolved_conflict_assessment = {
        "conflict_detected": True,
        "confidence": "medium",
        "safe_to_answer_confidently": True,
        "human_review_required": False,
        "conflicts": [
            {
                "status": "resolved_by_source_priority",
                "higher_priority_source": "account_contract",
                "message": (
                    "A conflict was detected. The account-specific "
                    "agreement takes priority."
                ),
            }
        ],
        "message": (
            "A conflict was detected and resolved using the defined "
            "source-priority rules."
        ),
    }

    answer = generate_answer(
        question=question,
        structured_facts=structured_facts,
        retrieved_evidence=retrieved_evidence,
        reliability_assessment=resolved_conflict_assessment,
    )

    print("\nQUESTION:")
    print(question)

    print("\nANSWER:")
    print(answer)

    # -------------------------------------------------------------
    # Test 3: Unresolved conflict requiring human review
    # -------------------------------------------------------------

    print("\n" + "=" * 70)
    print("TEST 3: HUMAN REVIEW REQUIRED")
    print("=" * 70)

    unresolved_conflict_assessment = {
        "conflict_detected": True,
        "confidence": "low",
        "safe_to_answer_confidently": False,
        "human_review_required": True,
        "conflicts": [
            {
                "status": "equal_priority_conflict",
                "higher_priority_source": None,
                "message": (
                    "Two sources of equal authority provide conflicting "
                    "information. Human review is required."
                ),
            }
        ],
        "message": (
            "The available information contains unresolved conflicts or "
            "uncertainty. Human review is recommended."
        ),
    }

    answer = generate_answer(
        question=question,
        structured_facts=structured_facts,
        retrieved_evidence=retrieved_evidence,
        reliability_assessment=unresolved_conflict_assessment,
    )

    print("\nQUESTION:")
    print(question)

    print("\nANSWER:")
    print(answer)

    print("\n" + "=" * 70)
    print("TESTING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()