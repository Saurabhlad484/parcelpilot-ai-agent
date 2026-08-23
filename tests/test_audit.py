from src.audit import (
    add_audit_step,
    add_reliability_result,
    clear_audit_logs,
    create_audit_trace,
    finalize_audit_trace,
    get_audit_logs,
    get_recent_audit_logs,
)


# ---------------------------------------------------------------------
# Test helper
# ---------------------------------------------------------------------

def get_test_user_context():
    """
    Return a reusable mock user context for audit tests.
    """

    return {
        "user_id": "northstar_user",
        "role": "customer",
        "account_id": "ACCT-001",
    }


# ---------------------------------------------------------------------
# Test 1: Create audit trace
# ---------------------------------------------------------------------

def test_create_audit_trace():
    """
    A new audit trace should contain the correct initial data.
    """

    clear_audit_logs()

    user_context = get_test_user_context()

    audit_trace = create_audit_trace(
        question="Can ORD-1001 be cancelled without a fee?",
        user_context=user_context,
    )

    assert audit_trace["question"] == (
        "Can ORD-1001 be cancelled without a fee?"
    )
    assert audit_trace["user_id"] == "northstar_user"
    assert audit_trace["role"] == "customer"
    assert audit_trace["account_id"] == "ACCT-001"

    assert audit_trace["steps"] == []
    assert audit_trace["final_status"] == "in_progress"
    assert audit_trace["reliability"] is None
    assert audit_trace["human_review_required"] is False

    assert "timestamp" in audit_trace


# ---------------------------------------------------------------------
# Test 2: Add audit step
# ---------------------------------------------------------------------

def test_add_audit_step():
    """
    An audit step should be added correctly to the trace.
    """

    clear_audit_logs()

    audit_trace = create_audit_trace(
        question="Can ORD-1001 be cancelled?",
        user_context=get_test_user_context(),
    )

    add_audit_step(
        audit_trace=audit_trace,
        step="access_control",
        status="passed",
        details="Order belongs to the authorised account.",
    )

    assert len(audit_trace["steps"]) == 1

    step = audit_trace["steps"][0]

    assert step["step"] == "access_control"
    assert step["status"] == "passed"
    assert step["details"] == (
        "Order belongs to the authorised account."
    )

    assert "timestamp" in step


# ---------------------------------------------------------------------
# Test 3: Add multiple audit steps
# ---------------------------------------------------------------------

def test_add_multiple_audit_steps():
    """
    Multiple audit steps should be stored in the order they were added.
    """

    clear_audit_logs()

    audit_trace = create_audit_trace(
        question="Can ORD-1001 be cancelled?",
        user_context=get_test_user_context(),
    )

    add_audit_step(
        audit_trace,
        step="question_validation",
        status="passed",
        details="Question is valid.",
    )

    add_audit_step(
        audit_trace,
        step="access_control",
        status="passed",
        details="Access granted.",
    )

    add_audit_step(
        audit_trace,
        step="structured_data",
        status="retrieved",
        details="Order information retrieved.",
    )

    assert len(audit_trace["steps"]) == 3

    assert audit_trace["steps"][0]["step"] == (
        "question_validation"
    )

    assert audit_trace["steps"][1]["step"] == (
        "access_control"
    )

    assert audit_trace["steps"][2]["step"] == (
        "structured_data"
    )


# ---------------------------------------------------------------------
# Test 4: Add reliability result
# ---------------------------------------------------------------------

def test_add_reliability_result():
    """
    Reliability confidence and human review status should be stored
    correctly in the audit trace.
    """

    clear_audit_logs()

    audit_trace = create_audit_trace(
        question="Can ORD-1001 be cancelled?",
        user_context=get_test_user_context(),
    )

    reliability = {
        "confidence": "high",
        "human_review_required": False,
        "message": (
            "The answer is supported by authoritative evidence."
        ),
    }

    add_reliability_result(
        audit_trace=audit_trace,
        reliability=reliability,
    )

    assert audit_trace["reliability"] == "high"

    assert audit_trace["human_review_required"] is False

    assert len(audit_trace["steps"]) == 1

    step = audit_trace["steps"][0]

    assert step["step"] == "reliability_assessment"
    assert step["status"] == "high"

    assert step["details"] == (
        "The answer is supported by authoritative evidence."
    )


# ---------------------------------------------------------------------
# Test 5: Reliability result requiring human review
# ---------------------------------------------------------------------

def test_reliability_result_requires_human_review():
    """
    The audit trace should correctly record when human review
    is required.
    """

    clear_audit_logs()

    audit_trace = create_audit_trace(
        question="Is this order eligible for a service credit?",
        user_context=get_test_user_context(),
    )

    reliability = {
        "confidence": "low",
        "human_review_required": True,
        "message": (
            "Conflicting evidence requires human review."
        ),
    }

    add_reliability_result(
        audit_trace=audit_trace,
        reliability=reliability,
    )

    assert audit_trace["reliability"] == "low"

    assert audit_trace["human_review_required"] is True

    assert audit_trace["steps"][0]["step"] == (
        "reliability_assessment"
    )

    assert audit_trace["steps"][0]["status"] == "low"


# ---------------------------------------------------------------------
# Test 6: Finalize audit trace
# ---------------------------------------------------------------------

def test_finalize_audit_trace():
    """
    Finalizing a trace should update its status and save it
    in the audit log.
    """

    clear_audit_logs()

    audit_trace = create_audit_trace(
        question="Can ORD-1001 be cancelled?",
        user_context=get_test_user_context(),
    )

    finalize_audit_trace(
        audit_trace=audit_trace,
        final_status="completed",
    )

    assert audit_trace["final_status"] == "completed"

    assert "completed_at" in audit_trace

    logs = get_audit_logs()

    assert len(logs) == 1

    assert logs[0]["question"] == (
        "Can ORD-1001 be cancelled?"
    )

    assert logs[0]["final_status"] == "completed"


# ---------------------------------------------------------------------
# Test 7: Retrieve audit logs
# ---------------------------------------------------------------------

def test_get_audit_logs():
    """
    get_audit_logs() should return all saved audit traces.
    """

    clear_audit_logs()

    first_trace = create_audit_trace(
        question="Question one",
        user_context=get_test_user_context(),
    )

    second_trace = create_audit_trace(
        question="Question two",
        user_context=get_test_user_context(),
    )

    finalize_audit_trace(
        first_trace,
        final_status="completed",
    )

    finalize_audit_trace(
        second_trace,
        final_status="completed",
    )

    logs = get_audit_logs()

    assert len(logs) == 2

    assert logs[0]["question"] == "Question one"

    assert logs[1]["question"] == "Question two"


# ---------------------------------------------------------------------
# Test 8: Retrieve recent audit logs
# ---------------------------------------------------------------------

def test_get_recent_audit_logs():
    """
    Recent logs should be returned in reverse chronological order.
    """

    clear_audit_logs()

    for question in [
        "First request",
        "Second request",
        "Third request",
    ]:

        audit_trace = create_audit_trace(
            question=question,
            user_context=get_test_user_context(),
        )

        finalize_audit_trace(
            audit_trace,
            final_status="completed",
        )

    recent_logs = get_recent_audit_logs(
        limit=2,
    )

    assert len(recent_logs) == 2

    # Most recent trace should come first.
    assert recent_logs[0]["question"] == "Third request"

    assert recent_logs[1]["question"] == "Second request"


# ---------------------------------------------------------------------
# Test 9: Recent logs with zero limit
# ---------------------------------------------------------------------

def test_get_recent_audit_logs_with_zero_limit():
    """
    A limit of zero should return an empty list.
    """

    clear_audit_logs()

    audit_trace = create_audit_trace(
        question="Test request",
        user_context=get_test_user_context(),
    )

    finalize_audit_trace(
        audit_trace,
        final_status="completed",
    )

    recent_logs = get_recent_audit_logs(
        limit=0,
    )

    assert recent_logs == []


# ---------------------------------------------------------------------
# Test 10: Recent logs with negative limit
# ---------------------------------------------------------------------

def test_get_recent_audit_logs_with_negative_limit():
    """
    A negative limit should return an empty list.
    """

    clear_audit_logs()

    recent_logs = get_recent_audit_logs(
        limit=-5,
    )

    assert recent_logs == []


# ---------------------------------------------------------------------
# Test 11: Clear audit logs
# ---------------------------------------------------------------------

def test_clear_audit_logs():
    """
    clear_audit_logs() should remove all saved audit traces.
    """

    clear_audit_logs()

    audit_trace = create_audit_trace(
        question="Can ORD-1001 be cancelled?",
        user_context=get_test_user_context(),
    )

    finalize_audit_trace(
        audit_trace,
        final_status="completed",
    )

    assert len(get_audit_logs()) == 1

    clear_audit_logs()

    assert get_audit_logs() == []


# ---------------------------------------------------------------------
# Test 12: Different final statuses
# ---------------------------------------------------------------------

def test_finalize_audit_trace_with_access_denied():
    """
    The audit layer should correctly store final statuses other
    than 'completed'.
    """

    clear_audit_logs()

    audit_trace = create_audit_trace(
        question="Show me information about ACCT-002",
        user_context=get_test_user_context(),
    )

    finalize_audit_trace(
        audit_trace,
        final_status="access_denied",
    )

    logs = get_audit_logs()

    assert len(logs) == 1

    assert logs[0]["final_status"] == "access_denied"

    assert "completed_at" in logs[0]