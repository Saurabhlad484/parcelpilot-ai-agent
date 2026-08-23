"""
ParcelPilot AI Agent - Trust and Reliability Layer

This module helps the agent decide:

1. Which source is more authoritative.
2. Whether sources conflict.
3. Whether historical answers can be trusted.
4. When the system should express uncertainty.
5. When human review may be required.
"""


from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------
# Source priority
# ---------------------------------------------------------------------

# Lower number means higher authority.
SOURCE_PRIORITY = {
    "account_contract": 1,
    "official_policy": 2,
    "structured_data": 3,
    "retrieved_document": 4,
    "historical_resolution": 5,
}


SOURCE_LABELS = {
    "account_contract": "Account-specific contract",
    "official_policy": "Official policy",
    "structured_data": "Structured operational data",
    "retrieved_document": "Retrieved supporting document",
    "historical_resolution": "Historical support resolution",
}


# ---------------------------------------------------------------------
# Source priority helpers
# ---------------------------------------------------------------------

def get_source_priority(source_type: str) -> int | None:
    """
    Return the authority priority of a source.

    Lower number means higher authority.

    Returns:
        Integer priority, or None if the source is unknown.
    """

    return SOURCE_PRIORITY.get(source_type)


def get_source_label(source_type: str) -> str:
    """
    Return a human-readable label for a source type.
    """

    return SOURCE_LABELS.get(
        source_type,
        "Unknown source",
    )


def compare_source_priority(
    source_a: str,
    source_b: str,
) -> int | None:
    """
    Compare two source types.

    Returns:
        1  -> source_a is more authoritative
        -1 -> source_b is more authoritative
        0  -> both have the same priority
        None -> one or both sources are unknown
    """

    priority_a = get_source_priority(source_a)
    priority_b = get_source_priority(source_b)

    if priority_a is None or priority_b is None:
        return None

    if priority_a < priority_b:
        return 1

    if priority_b < priority_a:
        return -1

    return 0


# ---------------------------------------------------------------------
# Historical resolution reliability
# ---------------------------------------------------------------------

def assess_historical_resolution(
    historical_resolution: Any | None,
) -> dict[str, Any]:
    """
    Assess whether a historical support resolution should be used
    as an authoritative source.

    Historical resolutions are useful context, but they must not
    override current policy or account-specific agreements.
    """

    if historical_resolution is None:
        return {
            "available": False,
            "authoritative": False,
            "reliability": "not_available",
            "message": (
                "No historical support resolution is available."
            ),
        }

    # Handle pandas NaN without importing pandas.
    normalised_value = str(
        historical_resolution
    ).strip().lower()

    if normalised_value in {
        "",
        "nan",
        "none",
        "null",
        "nat",
    }:
        return {
            "available": False,
            "authoritative": False,
            "reliability": "not_available",
            "message": (
                "No historical support resolution is available."
            ),
        }

    resolution_text = str(
        historical_resolution
    ).strip()

    return {
        "available": True,
        "authoritative": False,
        "reliability": "low",
        "message": (
            "Historical support guidance is available, but it is "
            "non-authoritative and must be checked against current "
            "policy and any account-specific agreement."
        ),
        "historical_resolution": resolution_text,
    }


# ---------------------------------------------------------------------
# Conflict detection
# ---------------------------------------------------------------------

def detect_conflict(
    source_a_type: str,
    source_a_value: Any,
    source_b_type: str,
    source_b_value: Any,
) -> dict[str, Any]:
    """
    Detect a potential conflict between two source findings.

    This function compares two conclusions or values.

    It reports:
    - whether a conflict exists
    - which source has higher authority
    - whether human review may be needed
    """

    priority_comparison = compare_source_priority(
        source_a_type,
        source_b_type,
    )

    # Unknown source types should not produce a confident decision.
    if priority_comparison is None:
        return {
            "conflict_detected": True,
            "status": "unknown_source",
            "higher_priority_source": None,
            "human_review_required": True,
            "message": (
                "One or more source types are unknown. "
                "The conflict cannot be resolved automatically."
            ),
        }

    # Normalise values for comparison.
    normalised_a = str(
        source_a_value
    ).strip().lower()

    normalised_b = str(
        source_b_value
    ).strip().lower()

    # Same conclusion means no conflict.
    if normalised_a == normalised_b:
        return {
            "conflict_detected": False,
            "status": "consistent",
            "higher_priority_source": None,
            "human_review_required": False,
            "message": (
                "The available sources are consistent."
            ),
        }

    # Values differ. Determine which source has higher authority.
    if priority_comparison == 1:

        higher_priority_source = source_a_type

    elif priority_comparison == -1:

        higher_priority_source = source_b_type

    else:
        # Same priority but conflicting conclusions.
        return {
            "conflict_detected": True,
            "status": "equal_priority_conflict",
            "higher_priority_source": None,
            "human_review_required": True,
            "message": (
                "Two sources of equal authority provide conflicting "
                "information. Human review is required."
            ),
        }

    return {
        "conflict_detected": True,
        "status": "resolved_by_source_priority",
        "higher_priority_source": higher_priority_source,
        "higher_priority_source_label": get_source_label(
            higher_priority_source
        ),
        "human_review_required": False,
        "message": (
            f"A conflict was detected. "
            f"The {get_source_label(higher_priority_source)} "
            f"takes priority."
        ),
    }


# ---------------------------------------------------------------------
# Main reliability assessment
# ---------------------------------------------------------------------

def assess_answer_reliability(
    account_contract_available: bool = False,
    official_policy_available: bool = False,
    structured_data_available: bool = False,
    retrieved_document_available: bool = False,
    historical_resolution: Any | None = None,
    conflicts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Produce a high-level reliability assessment for an answer.

    This function is the main interface used by agent.py.

    Parameters:
        account_contract_available:
            True when an applicable account-specific agreement exists.

        official_policy_available:
            True when current official policy evidence exists.

        structured_data_available:
            True when relevant order, ticket, or account facts exist.

        retrieved_document_available:
            True when supporting retrieved documents exist.

        historical_resolution:
            Optional previous support resolution.

        conflicts:
            Optional list of conflict dictionaries returned by
            detect_conflict().

    Returns:
        A complete reliability assessment.
    """

    # -------------------------------------------------------------
    # Normalise conflicts
    # -------------------------------------------------------------

    if conflicts is None:
        conflicts = []

    # -------------------------------------------------------------
    # Assess historical resolution
    # -------------------------------------------------------------

    historical_assessment = assess_historical_resolution(
        historical_resolution
    )

    # -------------------------------------------------------------
    # Calculate conflict status
    # -------------------------------------------------------------

    conflict_detected = any(
        conflict.get("conflict_detected", False)
        for conflict in conflicts
    )

    human_review_required = any(
        conflict.get("human_review_required", False)
        for conflict in conflicts
    )

    # -------------------------------------------------------------
    # Highest uncertainty:
    # unresolved conflict requires human review
    # -------------------------------------------------------------

    if human_review_required:
        confidence = "low"
        safe_to_answer_confidently = False
        message = (
            "The available information contains unresolved conflicts "
            "or uncertainty. Human review is recommended."
        )

    # -------------------------------------------------------------
    # Conflict exists but was resolved by source priority
    # -------------------------------------------------------------

    elif conflict_detected:
        confidence = "medium"
        safe_to_answer_confidently = True
        message = (
            "A conflict was detected and resolved using the defined "
            "source-priority rules."
        )

    # -------------------------------------------------------------
    # Strongest evidence:
    # account-specific contract + structured facts
    # -------------------------------------------------------------

    elif (
        account_contract_available
        and structured_data_available
    ):
        confidence = "high"
        safe_to_answer_confidently = True
        message = (
            "The answer is supported by an account-specific agreement "
            "and relevant structured operational data."
        )

    # -------------------------------------------------------------
    # Current policy + structured facts
    # -------------------------------------------------------------

    elif (
        official_policy_available
        and structured_data_available
    ):
        confidence = "high"
        safe_to_answer_confidently = True
        message = (
            "The answer is supported by current policy and relevant "
            "structured operational data."
        )

    # -------------------------------------------------------------
    # Account contract alone
    # -------------------------------------------------------------

    elif account_contract_available:
        confidence = "medium"
        safe_to_answer_confidently = True
        message = (
            "The answer is supported by an account-specific agreement, "
            "but some operational facts may be missing."
        )

    # -------------------------------------------------------------
    # Official policy alone
    # -------------------------------------------------------------

    elif official_policy_available:
        confidence = "medium"
        safe_to_answer_confidently = True
        message = (
            "The answer is supported by current official policy, "
            "but some operational facts may be missing."
        )

    # -------------------------------------------------------------
    # Structured facts only
    # -------------------------------------------------------------

    elif structured_data_available:
        confidence = "medium"
        safe_to_answer_confidently = True
        message = (
            "The answer is supported by structured operational data, "
            "but no governing policy or account-specific agreement "
            "was found."
        )

    # -------------------------------------------------------------
    # Supporting document only
    # -------------------------------------------------------------

    elif retrieved_document_available:
        confidence = "low"
        safe_to_answer_confidently = False
        human_review_required = True
        message = (
            "Supporting documents were found, but there is insufficient "
            "authoritative information to provide a reliable answer."
        )

    # -------------------------------------------------------------
    # Historical answer only
    # -------------------------------------------------------------

    elif historical_assessment["available"]:
        confidence = "low"
        safe_to_answer_confidently = False
        human_review_required = True
        message = (
            "Only historical support guidance is available. It is "
            "non-authoritative and should be reviewed against current "
            "policy or an applicable account agreement."
        )

    # -------------------------------------------------------------
    # Nothing sufficiently reliable found
    # -------------------------------------------------------------

    else:
        confidence = "low"
        safe_to_answer_confidently = False
        human_review_required = True
        message = (
            "There is insufficient authoritative information to "
            "provide a reliable answer."
        )

    return {
        "source_priority": SOURCE_PRIORITY.copy(),
        "historical_resolution_assessment": (
            historical_assessment
        ),
        "conflicts": conflicts,
        "conflict_detected": conflict_detected,
        "confidence": confidence,
        "safe_to_answer_confidently": (
            safe_to_answer_confidently
        ),
        "human_review_required": human_review_required,
        "message": message,
    }


# ---------------------------------------------------------------------
# Backward-compatible combined evaluation
# ---------------------------------------------------------------------

def evaluate_answer_reliability(
    has_account_contract: bool = False,
    has_official_policy: bool = False,
    has_structured_data: bool = False,
    historical_resolution: Any | None = None,
    conflicts: list[dict[str, Any]] | None = None,
    has_retrieved_document: bool = False,
) -> dict[str, Any]:
    """
    Backward-compatible wrapper around assess_answer_reliability().

    This allows older code and tests to continue using names such as:
    - has_account_contract
    - has_official_policy
    - has_structured_data
    """

    return assess_answer_reliability(
        account_contract_available=has_account_contract,
        official_policy_available=has_official_policy,
        structured_data_available=has_structured_data,
        retrieved_document_available=has_retrieved_document,
        historical_resolution=historical_resolution,
        conflicts=conflicts,
    )


# ---------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------

def main() -> None:

    print("=" * 70)
    print("TESTING TRUST AND RELIABILITY LAYER")
    print("=" * 70)

    # -------------------------------------------------------------
    # Test 1: Source priority
    # -------------------------------------------------------------

    print("\n1. SOURCE PRIORITY")

    print(
        "Account contract priority:",
        get_source_priority("account_contract"),
    )

    print(
        "Official policy priority:",
        get_source_priority("official_policy"),
    )

    print(
        "Historical resolution priority:",
        get_source_priority("historical_resolution"),
    )

    # -------------------------------------------------------------
    # Test 2: Compare sources
    # -------------------------------------------------------------

    print("\n2. SOURCE COMPARISON")

    print(
        "Contract vs Policy:",
        compare_source_priority(
            "account_contract",
            "official_policy",
        ),
    )

    print(
        "Policy vs Historical:",
        compare_source_priority(
            "official_policy",
            "historical_resolution",
        ),
    )

    # -------------------------------------------------------------
    # Test 3: Historical resolution
    # -------------------------------------------------------------

    print("\n3. HISTORICAL RESOLUTION ASSESSMENT")

    print(
        assess_historical_resolution(
            "Agent told customer a INR 250 cancellation "
            "fee applied after 30 minutes."
        )
    )

    # -------------------------------------------------------------
    # Test 4: No conflict
    # -------------------------------------------------------------

    print("\n4. CONSISTENT SOURCES")

    print(
        detect_conflict(
            "official_policy",
            "No cancellation fee",
            "account_contract",
            "No cancellation fee",
        )
    )

    # -------------------------------------------------------------
    # Test 5: Contract overrides historical answer
    # -------------------------------------------------------------

    print("\n5. CONTRACT OVERRIDES HISTORICAL ANSWER")

    print(
        detect_conflict(
            "account_contract",
            "No cancellation fee",
            "historical_resolution",
            "INR 250 cancellation fee",
        )
    )

    # -------------------------------------------------------------
    # Test 6: Equal priority conflict
    # -------------------------------------------------------------

    print("\n6. EQUAL PRIORITY CONFLICT")

    print(
        detect_conflict(
            "official_policy",
            "Eligible for service credit",
            "official_policy",
            "Not eligible for service credit",
        )
    )

    # -------------------------------------------------------------
    # Test 7: High confidence
    # -------------------------------------------------------------

    print("\n7. HIGH CONFIDENCE ASSESSMENT")

    print(
        assess_answer_reliability(
            account_contract_available=True,
            structured_data_available=True,
        )
    )

    # -------------------------------------------------------------
    # Test 8: Historical conflict resolved by contract
    # -------------------------------------------------------------

    print("\n8. RESOLVED CONFLICT")

    conflict = detect_conflict(
        "account_contract",
        "No cancellation fee",
        "historical_resolution",
        "INR 250 cancellation fee",
    )

    print(
        assess_answer_reliability(
            account_contract_available=True,
            structured_data_available=True,
            historical_resolution=(
                "Agent told customer a INR 250 cancellation "
                "fee applied after 30 minutes."
            ),
            conflicts=[conflict],
        )
    )

    # -------------------------------------------------------------
    # Test 9: Unresolved conflict
    # -------------------------------------------------------------

    print("\n9. HUMAN REVIEW REQUIRED")

    conflict = detect_conflict(
        "official_policy",
        "Eligible",
        "official_policy",
        "Not eligible",
    )

    print(
        assess_answer_reliability(
            official_policy_available=True,
            structured_data_available=True,
            conflicts=[conflict],
        )
    )

    print("\n" + "=" * 70)
    print("TESTING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()