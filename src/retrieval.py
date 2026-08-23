"""Authority-aware document retrieval for ParcelPilot AI Agent.

This module sits above document_store.py.

Its responsibility is NOT to generate answers. It determines which current,
account-specific, and topic-relevant documents should be used as evidence.

Authority precedence:
1. Customer agreement, only where it explicitly changes the relevant topic.
2. Current governing document for that topic.
3. Historical ticket/context is non-authoritative.
4. Deprecated documents are excluded unless explicitly requested.
"""

from __future__ import annotations

import re
from typing import Any

from src.document_store import (
    account_agreement_filter,
    global_documents_filter,
    search_chunks,
)


# ---------------------------------------------------------------------
# Topic configuration
# ---------------------------------------------------------------------

TOPIC_DOCUMENT_TYPES = {
    "support_sla": "support_policy",
    "cancellation": "cancellation_service_credit_sop",
    "service_credit": "cancellation_service_credit_sop",
    "product_operations": "product_operations_guide",
}


# Keywords used for simple deterministic topic classification.
# The future agent can provide more context, but retrieval should remain
# deterministic and testable.
TOPIC_KEYWORDS = {
    "cancellation": [
        "cancel",
        "cancellation",
        "cancelled",
        "canceled",
        "cancellation fee",
        "return to origin",
        "return-to-origin",
    ],
    "service_credit": [
        "service credit",
        "credit",
        "failed pickup",
        "failed-pickup",
        "pickup delay",
        "carrier fault",
        "compensation",
        "refund",
    ],
    "support_sla": [
        "sla",
        "response time",
        "response target",
        "severity",
        "p1",
        "p2",
        "p3",
        "priority",
        "support target",
    ],
    "product_operations": [
        "bulk upload",
        "upload limit",
        "csv",
        "known issue",
        "ki-",
        "product limit",
        "webhook",
        "swiftship",
        "confirmation delay",
    ],
}


# ---------------------------------------------------------------------
# ID extraction
# ---------------------------------------------------------------------

ACCOUNT_ID_PATTERN = re.compile(r"\bACCT-\d+\b", re.IGNORECASE)
ORDER_ID_PATTERN = re.compile(r"\bORD-\d+\b", re.IGNORECASE)
TICKET_ID_PATTERN = re.compile(r"\bTKT-\d+\b", re.IGNORECASE)


def extract_ids(text: str) -> dict[str, list[str]]:
    """Extract known ParcelPilot identifiers from text."""
    if not text.strip():
        return {
            "account_ids": [],
            "order_ids": [],
            "ticket_ids": [],
        }

    return {
        "account_ids": sorted(
            {match.upper() for match in ACCOUNT_ID_PATTERN.findall(text)}
        ),
        "order_ids": sorted(
            {match.upper() for match in ORDER_ID_PATTERN.findall(text)}
        ),
        "ticket_ids": sorted(
            {match.upper() for match in TICKET_ID_PATTERN.findall(text)}
        ),
    }


# ---------------------------------------------------------------------
# Topic detection
# ---------------------------------------------------------------------

def detect_topic(question: str) -> str | None:
    """Detect the main policy topic using deterministic keyword matching."""
    normalized = question.lower()

    scores: dict[str, int] = {}

    for topic, keywords in TOPIC_KEYWORDS.items():
        score = sum(keyword in normalized for keyword in keywords)
        if score:
            scores[topic] = score

    if not scores:
        return None

    # Deterministic tie-breaking follows dictionary insertion order.
    return max(scores, key=scores.get)


def requests_historical_comparison(question: str) -> bool:
    """Return True only when the user explicitly asks about historical policy."""
    normalized = question.lower()

    historical_phrases = [
        "old policy",
        "previous policy",
        "deprecated policy",
        "historical policy",
        "policy v2",
        "support policy v2",
        "before v3",
        "compare policies",
        "comparison with old",
    ]

    return any(phrase in normalized for phrase in historical_phrases)


# ---------------------------------------------------------------------
# Authority-aware retrieval
# ---------------------------------------------------------------------

def retrieve_governing_evidence(
    question: str,
    account_id: str | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """Retrieve evidence according to ParcelPilot authority rules.

    The returned evidence is separated into:
    - agreement_evidence: account-specific contract evidence
    - governing_evidence: current topic-specific global policy/SOP/guide
    - deprecated_evidence: returned only when explicitly requested

    This function deliberately does not decide whether a contract actually
    overrides a specific clause. That final topic-level reasoning is performed
    later by the grounded agent using the evidence.
    """
    if not question.strip():
        raise ValueError("question must not be blank.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    ids = extract_ids(question)
    topic = detect_topic(question)
    historical_request = requests_historical_comparison(question)

    # Explicit function argument takes priority over IDs in the question.
    resolved_account_id = account_id.upper() if account_id else None

    if resolved_account_id is None and ids["account_ids"]:
        resolved_account_id = ids["account_ids"][0]

    agreement_evidence: list[dict[str, Any]] = []
    governing_evidence: list[dict[str, Any]] = []
    deprecated_evidence: list[dict[str, Any]] = []

    # -------------------------------------------------------------
    # 1. Account agreement
    # -------------------------------------------------------------
    if resolved_account_id:
        agreement_evidence = search_chunks(
            question,
            top_k=top_k,
            where=account_agreement_filter(resolved_account_id),
        )

    # -------------------------------------------------------------
    # 2. Current governing document for the detected topic
    # -------------------------------------------------------------
    if topic:
        document_type = TOPIC_DOCUMENT_TYPES[topic]

        governing_evidence = search_chunks(
            question,
            top_k=top_k,
            where=global_documents_filter(
                document_type=document_type,
                status="current",
            ),
        )

    # -------------------------------------------------------------
    # 3. Deprecated material only when explicitly requested
    # -------------------------------------------------------------
    if historical_request:
        deprecated_evidence = search_chunks(
            question,
            top_k=top_k,
            where={"status": "deprecated"},
        )

    return {
        "question": question,
        "topic": topic,
        "account_id": resolved_account_id,
        "ids": ids,
        "authority_order": [
            "customer_agreement_when_topic_relevant",
            "current_governing_document",
            "historical_context_non_authoritative",
            "deprecated_only_if_explicitly_requested",
        ],
        "agreement_evidence": agreement_evidence,
        "governing_evidence": governing_evidence,
        "deprecated_evidence": deprecated_evidence,
    }


# ---------------------------------------------------------------------
# Executable validation
# ---------------------------------------------------------------------

def _print_evidence(label: str, evidence: list[dict[str, Any]]) -> None:
    """Print compact retrieval validation output."""
    print(f"{label}: {len(evidence)} result(s)")

    for item in evidence:
        print(
            f"  {item['source_file']} | "
            f"type={item['document_type']} | "
            f"status={item['status']} | "
            f"account={item['account_id']} | "
            f"page={item['page_range']}"
        )


def _validate_case(
    label: str,
    question: str,
    account_id: str | None = None,
) -> None:
    """Run and print one retrieval validation case."""
    print("\n" + "=" * 70)
    print(label)
    print(f"Question: {question}")

    result = retrieve_governing_evidence(
        question=question,
        account_id=account_id,
    )

    print(f"Topic: {result['topic']}")
    print(f"Account: {result['account_id']}")

    _print_evidence(
        "Agreement evidence",
        result["agreement_evidence"],
    )
    _print_evidence(
        "Governing evidence",
        result["governing_evidence"],
    )
    _print_evidence(
        "Deprecated evidence",
        result["deprecated_evidence"],
    )


def main() -> None:
    """Run focused authority-aware retrieval validation cases."""

    _validate_case(
        "A. Northstar cancellation",
        "Can ORD-1001 be cancelled without a fee?",
        account_id="ACCT-001",
    )

    _validate_case(
        "B. LumenWorks failed pickup service credit",
        "Does ORD-2002 qualify for a service credit after a failed pickup?",
        account_id="ACCT-002",
    )

    _validate_case(
        "C. Product bulk upload issue",
        "Why does TKT-502 fail when uploading a 4200-row CSV?",
    )

    _validate_case(
        "D. Support SLA",
        "What is the P1 response target?",
    )

    _validate_case(
        "E. Explicit historical request",
        "What did the old Support Policy v2 say about response targets?",
    )


if __name__ == "__main__":
    main()