import pytest

from src.retrieval import (
    extract_ids,
    detect_topic,
    requests_historical_comparison,
    retrieve_governing_evidence,
)


# =====================================================================
# ID EXTRACTION TESTS
# =====================================================================

# ---------------------------------------------------------------------
# Test 1: Extract one account ID
# ---------------------------------------------------------------------

def test_extract_account_id():
    result = extract_ids(
        "Show me information about ACCT-001"
    )

    assert result["account_ids"] == ["ACCT-001"]
    assert result["order_ids"] == []
    assert result["ticket_ids"] == []


# ---------------------------------------------------------------------
# Test 2: Extract one order ID
# ---------------------------------------------------------------------

def test_extract_order_id():
    result = extract_ids(
        "Can ORD-1001 be cancelled?"
    )

    assert result["account_ids"] == []
    assert result["order_ids"] == ["ORD-1001"]
    assert result["ticket_ids"] == []


# ---------------------------------------------------------------------
# Test 3: Extract one ticket ID
# ---------------------------------------------------------------------

def test_extract_ticket_id():
    result = extract_ids(
        "Please check TKT-501"
    )

    assert result["account_ids"] == []
    assert result["order_ids"] == []
    assert result["ticket_ids"] == ["TKT-501"]


# ---------------------------------------------------------------------
# Test 4: Extract multiple IDs
# ---------------------------------------------------------------------

def test_extract_multiple_ids():
    result = extract_ids(
        "Check ACCT-001, ORD-1001 and TKT-501"
    )

    assert result["account_ids"] == ["ACCT-001"]
    assert result["order_ids"] == ["ORD-1001"]
    assert result["ticket_ids"] == ["TKT-501"]


# ---------------------------------------------------------------------
# Test 5: ID extraction is case-insensitive
# ---------------------------------------------------------------------

def test_extract_ids_case_insensitive():
    result = extract_ids(
        "Check acct-001, ord-1001 and tkt-501"
    )

    assert result["account_ids"] == ["ACCT-001"]
    assert result["order_ids"] == ["ORD-1001"]
    assert result["ticket_ids"] == ["TKT-501"]


# ---------------------------------------------------------------------
# Test 6: Duplicate IDs are returned only once
# ---------------------------------------------------------------------

def test_extract_duplicate_ids_once():
    result = extract_ids(
        "Check ORD-1001 and ord-1001 again"
    )

    assert result["order_ids"] == ["ORD-1001"]


# ---------------------------------------------------------------------
# Test 7: Empty text returns empty ID lists
# ---------------------------------------------------------------------

def test_extract_ids_from_empty_text():
    result = extract_ids("")

    assert result == {
        "account_ids": [],
        "order_ids": [],
        "ticket_ids": [],
    }


# =====================================================================
# TOPIC DETECTION TESTS
# =====================================================================

# ---------------------------------------------------------------------
# Test 8: Detect cancellation topic
# ---------------------------------------------------------------------

def test_detect_cancellation_topic():
    assert detect_topic(
        "Can I cancel my order?"
    ) == "cancellation"


# ---------------------------------------------------------------------
# Test 9: Detect service credit topic
# ---------------------------------------------------------------------

def test_detect_service_credit_topic():
    assert detect_topic(
        "Do I qualify for a service credit?"
    ) == "service_credit"


# ---------------------------------------------------------------------
# Test 10: Detect support SLA topic
# ---------------------------------------------------------------------

def test_detect_support_sla_topic():
    assert detect_topic(
        "What is the P1 response time SLA?"
    ) == "support_sla"


# ---------------------------------------------------------------------
# Test 11: Detect product operations topic
# ---------------------------------------------------------------------

def test_detect_product_operations_topic():
    assert detect_topic(
        "Why does my bulk upload CSV fail?"
    ) == "product_operations"


# ---------------------------------------------------------------------
# Test 12: Unknown topic returns None
# ---------------------------------------------------------------------

def test_detect_unknown_topic():
    assert detect_topic(
        "What is the capital of France?"
    ) is None


# ---------------------------------------------------------------------
# Test 13: Topic detection is case-insensitive
# ---------------------------------------------------------------------

def test_detect_topic_case_insensitive():
    assert detect_topic(
        "CAN I CANCEL MY ORDER?"
    ) == "cancellation"


# =====================================================================
# HISTORICAL REQUEST DETECTION TESTS
# =====================================================================

# ---------------------------------------------------------------------
# Test 14: Detect old policy request
# ---------------------------------------------------------------------

def test_detect_old_policy_request():
    assert requests_historical_comparison(
        "What did the old policy say?"
    ) is True


# ---------------------------------------------------------------------
# Test 15: Detect previous policy request
# ---------------------------------------------------------------------

def test_detect_previous_policy_request():
    assert requests_historical_comparison(
        "Show me the previous policy"
    ) is True


# ---------------------------------------------------------------------
# Test 16: Detect deprecated policy request
# ---------------------------------------------------------------------

def test_detect_deprecated_policy_request():
    assert requests_historical_comparison(
        "Can I see the deprecated policy?"
    ) is True


# ---------------------------------------------------------------------
# Test 17: Detect policy v2 request
# ---------------------------------------------------------------------

def test_detect_policy_v2_request():
    assert requests_historical_comparison(
        "What did Support Policy v2 say?"
    ) is True


# ---------------------------------------------------------------------
# Test 18: Normal current-policy question is not historical
# ---------------------------------------------------------------------

def test_current_policy_is_not_historical_request():
    assert requests_historical_comparison(
        "What is the current cancellation policy?"
    ) is False


# ---------------------------------------------------------------------
# Test 19: Historical comparison detection is case-insensitive
# ---------------------------------------------------------------------

def test_historical_request_case_insensitive():
    assert requests_historical_comparison(
        "SHOW ME THE OLD POLICY"
    ) is True


# =====================================================================
# GOVERNING EVIDENCE RETRIEVAL TESTS
# =====================================================================

# ---------------------------------------------------------------------
# Test 20: Blank question raises ValueError
# ---------------------------------------------------------------------

def test_blank_question_raises_value_error():
    with pytest.raises(ValueError) as exc_info:
        retrieve_governing_evidence("")

    assert "question must not be blank" in str(
        exc_info.value
    )


# ---------------------------------------------------------------------
# Test 21: Whitespace-only question raises ValueError
# ---------------------------------------------------------------------

def test_whitespace_question_raises_value_error():
    with pytest.raises(ValueError):
        retrieve_governing_evidence("   ")


# ---------------------------------------------------------------------
# Test 22: Invalid top_k raises ValueError
# ---------------------------------------------------------------------

def test_zero_top_k_raises_value_error():
    with pytest.raises(ValueError) as exc_info:
        retrieve_governing_evidence(
            "Can I cancel ORD-1001?",
            top_k=0,
        )

    assert "top_k must be at least 1" in str(
        exc_info.value
    )


# ---------------------------------------------------------------------
# Test 23: Negative top_k raises ValueError
# ---------------------------------------------------------------------

def test_negative_top_k_raises_value_error():
    with pytest.raises(ValueError):
        retrieve_governing_evidence(
            "Can I cancel ORD-1001?",
            top_k=-1,
        )


# ---------------------------------------------------------------------
# Test 24: Explicit account ID is resolved correctly
# ---------------------------------------------------------------------

def test_explicit_account_id_is_resolved():
    result = retrieve_governing_evidence(
        question="Can ORD-1001 be cancelled?",
        account_id="ACCT-001",
    )

    assert result["account_id"] == "ACCT-001"


# ---------------------------------------------------------------------
# Test 25: Account ID argument is case-insensitive
# ---------------------------------------------------------------------

def test_explicit_account_id_is_case_insensitive():
    result = retrieve_governing_evidence(
        question="Can ORD-1001 be cancelled?",
        account_id="acct-001",
    )

    assert result["account_id"] == "ACCT-001"


# ---------------------------------------------------------------------
# Test 26: Account ID can be extracted from question
# ---------------------------------------------------------------------

def test_account_id_extracted_from_question():
    result = retrieve_governing_evidence(
        question=(
            "What is the cancellation policy for ACCT-001?"
        )
    )

    assert result["account_id"] == "ACCT-001"


# ---------------------------------------------------------------------
# Test 27: Explicit account argument takes priority
# ---------------------------------------------------------------------

def test_explicit_account_id_overrides_question_account():
    result = retrieve_governing_evidence(
        question=(
            "What is the cancellation policy for ACCT-002?"
        ),
        account_id="ACCT-001",
    )

    assert result["account_id"] == "ACCT-001"


# ---------------------------------------------------------------------
# Test 28: Cancellation question detects correct topic
# ---------------------------------------------------------------------

def test_retrieval_detects_cancellation_topic():
    result = retrieve_governing_evidence(
        question="Can ORD-1001 be cancelled?"
    )

    assert result["topic"] == "cancellation"


# ---------------------------------------------------------------------
# Test 29: Service credit question detects correct topic
# ---------------------------------------------------------------------

def test_retrieval_detects_service_credit_topic():
    result = retrieve_governing_evidence(
        question=(
            "Does ORD-2002 qualify for a service credit "
            "after a failed pickup?"
        )
    )

    assert result["topic"] == "service_credit"


# ---------------------------------------------------------------------
# Test 30: SLA question detects correct topic
# ---------------------------------------------------------------------

def test_retrieval_detects_sla_topic():
    result = retrieve_governing_evidence(
        question="What is the P1 response target?"
    )

    assert result["topic"] == "support_sla"


# ---------------------------------------------------------------------
# Test 31: Product operations question detects correct topic
# ---------------------------------------------------------------------

def test_retrieval_detects_product_operations_topic():
    result = retrieve_governing_evidence(
        question=(
            "Why does my bulk upload CSV fail?"
        )
    )

    assert result["topic"] == "product_operations"


# ---------------------------------------------------------------------
# Test 32: Unknown question has no topic
# ---------------------------------------------------------------------

def test_retrieval_unknown_question_has_no_topic():
    result = retrieve_governing_evidence(
        question="What is the capital of France?"
    )

    assert result["topic"] is None


# ---------------------------------------------------------------------
# Test 33: Result contains required top-level fields
# ---------------------------------------------------------------------

def test_retrieval_result_structure():
    result = retrieve_governing_evidence(
        question="Can I cancel ORD-1001?"
    )

    expected_keys = {
        "question",
        "topic",
        "account_id",
        "ids",
        "authority_order",
        "agreement_evidence",
        "governing_evidence",
        "deprecated_evidence",
    }

    assert expected_keys.issubset(result.keys())


# ---------------------------------------------------------------------
# Test 34: Authority order is correct
# ---------------------------------------------------------------------

def test_authority_order():
    result = retrieve_governing_evidence(
        question="Can I cancel ORD-1001?"
    )

    assert result["authority_order"] == [
        "customer_agreement_when_topic_relevant",
        "current_governing_document",
        "historical_context_non_authoritative",
        "deprecated_only_if_explicitly_requested",
    ]


# ---------------------------------------------------------------------
# Test 35: Normal request excludes deprecated evidence
# ---------------------------------------------------------------------

def test_normal_request_excludes_deprecated_evidence():
    result = retrieve_governing_evidence(
        question="What is the cancellation policy?"
    )

    assert result["deprecated_evidence"] == []


# ---------------------------------------------------------------------
# Test 36: Historical request includes deprecated search
# ---------------------------------------------------------------------

def test_historical_request_returns_deprecated_evidence_field():
    result = retrieve_governing_evidence(
        question=(
            "What did the old Support Policy v2 say "
            "about response targets?"
        )
    )

    # We verify that the historical path executes and returns
    # a list. The exact number depends on the document store.
    assert isinstance(
        result["deprecated_evidence"],
        list,
    )


# ---------------------------------------------------------------------
# Test 37: Account-specific retrieval returns evidence lists
# ---------------------------------------------------------------------

def test_account_retrieval_returns_evidence_lists():
    result = retrieve_governing_evidence(
        question="Can ORD-1001 be cancelled?",
        account_id="ACCT-001",
    )

    assert isinstance(
        result["agreement_evidence"],
        list,
    )

    assert isinstance(
        result["governing_evidence"],
        list,
    )


# ---------------------------------------------------------------------
# Test 38: Extracted IDs are included in result
# ---------------------------------------------------------------------

def test_retrieval_returns_extracted_ids():
    result = retrieve_governing_evidence(
        question=(
            "Check ACCT-001 and ORD-1001 for TKT-501"
        )
    )

    assert result["ids"]["account_ids"] == ["ACCT-001"]
    assert result["ids"]["order_ids"] == ["ORD-1001"]
    assert result["ids"]["ticket_ids"] == ["TKT-501"]