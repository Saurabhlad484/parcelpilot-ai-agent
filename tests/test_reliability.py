from src.reliability import (
    assess_answer_reliability,
    assess_historical_resolution,
    compare_source_priority,
    detect_conflict,
    get_source_label,
    get_source_priority,
)


# ---------------------------------------------------------------------
# Test 1: Source priority
# ---------------------------------------------------------------------

def test_source_priority():
    """
    Account contracts should have the highest authority,
    while historical resolutions should have the lowest.
    """

    assert get_source_priority("account_contract") == 1
    assert get_source_priority("official_policy") == 2
    assert get_source_priority("structured_data") == 3
    assert get_source_priority("retrieved_document") == 4
    assert get_source_priority("historical_resolution") == 5


# ---------------------------------------------------------------------
# Test 2: Unknown source priority
# ---------------------------------------------------------------------

def test_unknown_source_priority():
    """
    Unknown source types should return None.
    """

    assert get_source_priority("unknown_source") is None


# ---------------------------------------------------------------------
# Test 3: Source labels
# ---------------------------------------------------------------------

def test_source_labels():
    """
    Known source types should return readable labels.
    """

    assert (
        get_source_label("account_contract")
        == "Account-specific contract"
    )

    assert (
        get_source_label("official_policy")
        == "Official policy"
    )

    assert (
        get_source_label("unknown_source")
        == "Unknown source"
    )


# ---------------------------------------------------------------------
# Test 4: Compare source priority
# ---------------------------------------------------------------------

def test_account_contract_higher_than_policy():
    """
    Account-specific contracts should override official policy.
    """

    result = compare_source_priority(
        "account_contract",
        "official_policy",
    )

    assert result == 1


# ---------------------------------------------------------------------
# Test 5: Policy higher than historical resolution
# ---------------------------------------------------------------------

def test_policy_higher_than_historical_resolution():
    """
    Official policy should override historical support guidance.
    """

    result = compare_source_priority(
        "official_policy",
        "historical_resolution",
    )

    assert result == 1


# ---------------------------------------------------------------------
# Test 6: Reverse source comparison
# ---------------------------------------------------------------------

def test_historical_resolution_lower_than_contract():
    """
    Historical guidance should lose when compared with
    an account-specific contract.
    """

    result = compare_source_priority(
        "historical_resolution",
        "account_contract",
    )

    assert result == -1


# ---------------------------------------------------------------------
# Test 7: Same source priority
# ---------------------------------------------------------------------

def test_same_source_priority():
    """
    Comparing the same source type should return 0.
    """

    result = compare_source_priority(
        "official_policy",
        "official_policy",
    )

    assert result == 0


# ---------------------------------------------------------------------
# Test 8: Unknown source comparison
# ---------------------------------------------------------------------

def test_unknown_source_comparison():
    """
    Comparing an unknown source should return None.
    """

    result = compare_source_priority(
        "unknown_source",
        "official_policy",
    )

    assert result is None


# ---------------------------------------------------------------------
# Test 9: No historical resolution
# ---------------------------------------------------------------------

def test_no_historical_resolution():
    """
    None should be treated as no historical guidance.
    """

    result = assess_historical_resolution(None)

    assert result["available"] is False
    assert result["authoritative"] is False
    assert result["reliability"] == "not_available"


# ---------------------------------------------------------------------
# Test 10: Empty historical resolution
# ---------------------------------------------------------------------

def test_empty_historical_resolution():
    """
    Empty strings and NaN-like values should not be treated
    as valid historical guidance.
    """

    result = assess_historical_resolution("")

    assert result["available"] is False
    assert result["reliability"] == "not_available"


# ---------------------------------------------------------------------
# Test 11: Valid historical resolution
# ---------------------------------------------------------------------

def test_valid_historical_resolution():
    """
    Historical guidance can be available but must remain
    non-authoritative.
    """

    historical_text = (
        "A previous agent said a cancellation fee applied."
    )

    result = assess_historical_resolution(
        historical_text
    )

    assert result["available"] is True
    assert result["authoritative"] is False
    assert result["reliability"] == "low"
    assert result["historical_resolution"] == historical_text


# ---------------------------------------------------------------------
# Test 12: Consistent sources
# ---------------------------------------------------------------------

def test_consistent_sources_no_conflict():
    """
    Two sources with the same conclusion should not conflict.
    """

    result = detect_conflict(
        "official_policy",
        "No cancellation fee",
        "account_contract",
        "No cancellation fee",
    )

    assert result["conflict_detected"] is False
    assert result["status"] == "consistent"
    assert result["human_review_required"] is False


# ---------------------------------------------------------------------
# Test 13: Contract resolves conflict with policy
# ---------------------------------------------------------------------

def test_contract_overrides_policy_conflict():
    """
    When conclusions differ, the account contract should
    override the official policy.
    """

    result = detect_conflict(
        "account_contract",
        "No cancellation fee",
        "official_policy",
        "Cancellation fee applies",
    )

    assert result["conflict_detected"] is True
    assert result["status"] == "resolved_by_source_priority"
    assert result["higher_priority_source"] == (
        "account_contract"
    )
    assert result["human_review_required"] is False


# ---------------------------------------------------------------------
# Test 14: Policy overrides historical resolution
# ---------------------------------------------------------------------

def test_policy_overrides_historical_conflict():
    """
    Current official policy should override old support guidance.
    """

    result = detect_conflict(
        "official_policy",
        "Eligible for service credit",
        "historical_resolution",
        "Not eligible for service credit",
    )

    assert result["conflict_detected"] is True
    assert result["higher_priority_source"] == (
        "official_policy"
    )
    assert result["human_review_required"] is False


# ---------------------------------------------------------------------
# Test 15: Equal priority conflict requires human review
# ---------------------------------------------------------------------

def test_equal_priority_conflict_requires_human_review():
    """
    Two equally authoritative sources with conflicting conclusions
    cannot be resolved automatically.
    """

    result = detect_conflict(
        "official_policy",
        "Eligible",
        "official_policy",
        "Not eligible",
    )

    assert result["conflict_detected"] is True
    assert result["status"] == "equal_priority_conflict"
    assert result["higher_priority_source"] is None
    assert result["human_review_required"] is True


# ---------------------------------------------------------------------
# Test 16: Unknown source requires human review
# ---------------------------------------------------------------------

def test_unknown_source_conflict_requires_human_review():
    """
    Unknown source types must not produce an automatic decision.
    """

    result = detect_conflict(
        "unknown_source",
        "Eligible",
        "official_policy",
        "Not eligible",
    )

    assert result["conflict_detected"] is True
    assert result["status"] == "unknown_source"
    assert result["human_review_required"] is True


# ---------------------------------------------------------------------
# Test 17: High confidence from contract + structured data
# ---------------------------------------------------------------------

def test_high_confidence_contract_and_structured_data():
    """
    Account contract plus structured operational data is the
    strongest normal evidence combination.
    """

    result = assess_answer_reliability(
        account_contract_available=True,
        structured_data_available=True,
    )

    assert result["confidence"] == "high"
    assert result["safe_to_answer_confidently"] is True
    assert result["human_review_required"] is False


# ---------------------------------------------------------------------
# Test 18: High confidence from policy + structured data
# ---------------------------------------------------------------------

def test_high_confidence_policy_and_structured_data():
    """
    Official policy plus structured data should produce
    high confidence.
    """

    result = assess_answer_reliability(
        official_policy_available=True,
        structured_data_available=True,
    )

    assert result["confidence"] == "high"
    assert result["safe_to_answer_confidently"] is True
    assert result["human_review_required"] is False


# ---------------------------------------------------------------------
# Test 19: Contract alone gives medium confidence
# ---------------------------------------------------------------------

def test_contract_alone_medium_confidence():
    """
    A contract without operational facts should still be usable,
    but with medium confidence.
    """

    result = assess_answer_reliability(
        account_contract_available=True,
    )

    assert result["confidence"] == "medium"
    assert result["safe_to_answer_confidently"] is True


# ---------------------------------------------------------------------
# Test 20: Official policy alone gives medium confidence
# ---------------------------------------------------------------------

def test_policy_alone_medium_confidence():
    """
    Official policy alone should produce medium confidence.
    """

    result = assess_answer_reliability(
        official_policy_available=True,
    )

    assert result["confidence"] == "medium"
    assert result["safe_to_answer_confidently"] is True


# ---------------------------------------------------------------------
# Test 21: Structured data alone gives medium confidence
# ---------------------------------------------------------------------

def test_structured_data_alone_medium_confidence():
    """
    Structured data alone is useful but does not establish
    the governing rule.
    """

    result = assess_answer_reliability(
        structured_data_available=True,
    )

    assert result["confidence"] == "medium"
    assert result["safe_to_answer_confidently"] is True


# ---------------------------------------------------------------------
# Test 22: Retrieved document only requires review
# ---------------------------------------------------------------------

def test_retrieved_document_only_requires_review():
    """
    A supporting document without authoritative evidence should
    not be enough for a confident answer.
    """

    result = assess_answer_reliability(
        retrieved_document_available=True,
    )

    assert result["confidence"] == "low"
    assert result["safe_to_answer_confidently"] is False
    assert result["human_review_required"] is True


# ---------------------------------------------------------------------
# Test 23: Historical resolution only requires review
# ---------------------------------------------------------------------

def test_historical_resolution_only_requires_review():
    """
    Historical support guidance alone must not determine
    the final answer.
    """

    result = assess_answer_reliability(
        historical_resolution=(
            "A previous agent approved a service credit."
        ),
    )

    assert result["confidence"] == "low"
    assert result["safe_to_answer_confidently"] is False
    assert result["human_review_required"] is True


# ---------------------------------------------------------------------
# Test 24: No evidence requires review
# ---------------------------------------------------------------------

def test_no_evidence_requires_human_review():
    """
    With no reliable evidence, the agent should require review.
    """

    result = assess_answer_reliability()

    assert result["confidence"] == "low"
    assert result["safe_to_answer_confidently"] is False
    assert result["human_review_required"] is True


# ---------------------------------------------------------------------
# Test 25: Resolved conflict gives medium confidence
# ---------------------------------------------------------------------

def test_resolved_conflict_medium_confidence():
    """
    A conflict resolved by source priority should produce
    medium confidence rather than high confidence.
    """

    conflict = detect_conflict(
        "account_contract",
        "No cancellation fee",
        "official_policy",
        "Cancellation fee applies",
    )

    result = assess_answer_reliability(
        account_contract_available=True,
        structured_data_available=True,
        conflicts=[conflict],
    )

    assert result["conflict_detected"] is True
    assert result["confidence"] == "medium"
    assert result["safe_to_answer_confidently"] is True
    assert result["human_review_required"] is False


# ---------------------------------------------------------------------
# Test 26: Unresolved conflict requires human review
# ---------------------------------------------------------------------

def test_unresolved_conflict_requires_human_review():
    """
    An equal-priority conflict must override otherwise strong
    evidence and require human review.
    """

    conflict = detect_conflict(
        "official_policy",
        "Eligible",
        "official_policy",
        "Not eligible",
    )

    result = assess_answer_reliability(
        official_policy_available=True,
        structured_data_available=True,
        conflicts=[conflict],
    )

    assert result["conflict_detected"] is True
    assert result["confidence"] == "low"
    assert result["safe_to_answer_confidently"] is False
    assert result["human_review_required"] is True
    