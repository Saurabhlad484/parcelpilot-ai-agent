from src.agent import (
    get_user_context,
    run_agent,
)

from src.audit import (
    clear_audit_logs,
    get_audit_logs,
)


# ---------------------------------------------------------------------
# Test 1: Valid Northstar cancellation request
# ---------------------------------------------------------------------

def test_valid_northstar_cancellation():
    """
    ORD-1001 belongs to ACCT-001.

    The Northstar user should be authorised to access it and
    receive a grounded answer.
    """

    user_context = get_user_context("northstar_user")

    answer, pending_action = run_agent(
        question="Can ORD-1001 be cancelled without a fee?",
        user_context=user_context,
    )

    assert pending_action is None
    assert "ORD-1001" in answer
    assert "Access denied" not in answer


# ---------------------------------------------------------------------
# Test 2: Cross-account order access must be denied
# ---------------------------------------------------------------------

def test_cross_account_access_denied():
    """
    northstar_user belongs to ACCT-001.

    ORD-2002 belongs to ACCT-002, so access must be denied.
    """

    user_context = get_user_context("northstar_user")

    answer, pending_action = run_agent(
        question="Can ORD-2002 be cancelled without a fee?",
        user_context=user_context,
    )

    assert pending_action is None
    assert "Access denied" in answer
    assert "ORD-2002" in answer


# ---------------------------------------------------------------------
# Test 3: Valid LumenWorks cancellation request
# ---------------------------------------------------------------------

def test_valid_lumenworks_cancellation():
    """
    lumenworks_user belongs to ACCT-002.

    ORD-2002 belongs to ACCT-002, so the request should proceed.
    """

    user_context = get_user_context("lumenworks_user")

    answer, pending_action = run_agent(
        question="Can ORD-2002 be cancelled without a fee?",
        user_context=user_context,
    )

    assert pending_action is None
    assert "Access denied" not in answer


# ---------------------------------------------------------------------
# Test 4: Invalid order ID
# ---------------------------------------------------------------------

def test_invalid_order_id():
    """
    The agent should reject an order ID that does not exist.
    """

    user_context = get_user_context("northstar_user")

    answer, pending_action = run_agent(
        question="Can ORD-9999 be cancelled?",
        user_context=user_context,
    )

    assert pending_action is None
    assert "couldn't find order" in answer


# ---------------------------------------------------------------------
# Test 5: Missing required order ID
# ---------------------------------------------------------------------

def test_missing_order_id():
    """
    Cancellation questions require an order ID.
    """

    user_context = get_user_context("northstar_user")

    answer, pending_action = run_agent(
        question="Can I cancel my order?",
        user_context=user_context,
    )

    assert pending_action is None
    assert "Please provide the order ID" in answer


# ---------------------------------------------------------------------
# Test 6: Out-of-scope question
# ---------------------------------------------------------------------

def test_out_of_scope_question():
    """
    The agent should reject unrelated questions.
    """

    user_context = get_user_context("northstar_user")

    answer, pending_action = run_agent(
        question="What is the capital of France?",
        user_context=user_context,
    )

    assert pending_action is None
    assert "cannot answer unrelated questions" in answer


# ---------------------------------------------------------------------
# Test 7: Escalation requires explicit confirmation
# ---------------------------------------------------------------------

def test_escalation_requires_confirmation():
    """
    An escalation should not be created immediately.

    The agent must first return a pending action and request
    explicit user confirmation.
    """

    user_context = get_user_context("northstar_user")

    answer, pending_action = run_agent(
        question="Please escalate TKT-501",
        user_context=user_context,
    )

    assert pending_action is not None
    assert pending_action["action"] == "create_escalation"
    assert "Reply `yes` to confirm" in answer


# ---------------------------------------------------------------------
# Test 8: Declining escalation does not create it
# ---------------------------------------------------------------------

def test_escalation_cancellation():
    """
    If the user replies 'no', the escalation must not be created.
    """

    user_context = get_user_context("northstar_user")

    _, pending_action = run_agent(
        question="Please escalate TKT-501",
        user_context=user_context,
    )

    answer, new_pending_action = run_agent(
        question="no",
        pending_action=pending_action,
        user_context=user_context,
    )

    assert new_pending_action is None
    assert "was not created" in answer


# ---------------------------------------------------------------------
# Test 9: Explicit account access must be denied
# ---------------------------------------------------------------------

def test_explicit_account_access_denied():
    """
    northstar_user belongs to ACCT-001 and must not access ACCT-002.
    """

    user_context = get_user_context("northstar_user")

    answer, pending_action = run_agent(
        question="Show me information about ACCT-002",
        user_context=user_context,
    )

    assert pending_action is None
    assert "Access denied" in answer


# ---------------------------------------------------------------------
# Test 10: Unauthorized ticket escalation must be denied
# ---------------------------------------------------------------------

def test_unauthorized_ticket_escalation_denied():
    """
    northstar_user belongs to ACCT-001.

    TKT-502 belongs to ACCT-002, so Northstar must not be allowed
    to prepare an escalation for it.
    """

    user_context = get_user_context("northstar_user")

    answer, pending_action = run_agent(
        question="Please escalate TKT-502",
        user_context=user_context,
    )

    assert pending_action is None
    assert "Access denied" in answer
    assert "TKT-502" in answer


# ---------------------------------------------------------------------
# Test 11: Escalation is created only after confirmation
# ---------------------------------------------------------------------

def test_escalation_confirmation_creates_action():
    """
    The escalation should be created only after the user explicitly
    replies with 'yes'.
    """

    user_context = get_user_context("northstar_user")

    _, pending_action = run_agent(
        question="Please escalate TKT-501",
        user_context=user_context,
    )

    assert pending_action is not None

    answer, new_pending_action = run_agent(
        question="yes",
        pending_action=pending_action,
        user_context=user_context,
    )

    assert new_pending_action is None
    assert "Escalation created successfully" in answer
    assert "Escalation ID:" in answer


# ---------------------------------------------------------------------
# Test 12: Real agent request creates an audit trace
# ---------------------------------------------------------------------

def test_agent_creates_audit_trace():
    """
    A real request through run_agent() should create and save
    an audit trace containing important decision steps.
    """

    clear_audit_logs()

    user_context = get_user_context("northstar_user")

    answer, pending_action = run_agent(
        question="Can ORD-1001 be cancelled without a fee?",
        user_context=user_context,
    )

    logs = get_audit_logs()

    assert pending_action is None
    assert "Access denied" not in answer
    assert len(logs) == 1

    audit_log = logs[0]

    assert audit_log["question"] == (
        "Can ORD-1001 be cancelled without a fee?"
    )

    assert audit_log["user_id"] == "northstar_user"
    assert audit_log["account_id"] == "ACCT-001"
    assert audit_log["final_status"] == "completed"

    step_names = [
        step["step"]
        for step in audit_log["steps"]
    ]

    assert "question_validation" in step_names
    assert "identifier_validation" in step_names
    assert "access_control" in step_names
    assert "structured_data" in step_names
    assert "evidence_retrieval" in step_names
    assert "reliability_assessment" in step_names
    assert "answer_generation" in step_names