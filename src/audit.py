"""
ParcelPilot AI Agent - Audit and Decision Trace Layer

This module records the important decisions made during an
agent request.

The audit trail helps operations users understand:
1. Who made the request.
2. Which account was involved.
3. Which access checks were performed.
4. Which structured data was retrieved.
5. Which evidence sources were used.
6. Whether conflicts were detected.
7. How source priority resolved a conflict.
8. What reliability level was assigned.
9. Whether human review was required.

The audit trail stores internal decision information only.
It is not shown to normal customer users.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any


# ---------------------------------------------------------------------
# In-memory audit storage
# ---------------------------------------------------------------------

AUDIT_LOGS: list[dict[str, Any]] = []


# ---------------------------------------------------------------------
# Create audit trace
# ---------------------------------------------------------------------

def create_audit_trace(
    question: str,
    user_context: dict[str, Any],
) -> dict[str, Any]:
    """
    Create a new audit trace for one agent request.

    The trace is updated throughout the agent pipeline.
    """

    return {
        "timestamp": datetime.now().isoformat(
            timespec="seconds"
        ),
        "question": question,
        "user_id": user_context.get("user_id"),
        "role": user_context.get("role"),
        "account_id": user_context.get("account_id"),
        "steps": [],
        "final_status": "in_progress",
        "reliability": None,
        "human_review_required": False,
    }


# ---------------------------------------------------------------------
# Add audit step
# ---------------------------------------------------------------------

def add_audit_step(
    audit_trace: dict[str, Any],
    step: str,
    status: str,
    details: str | None = None,
) -> None:
    """
    Add one decision step to an audit trace.

    Example:

    add_audit_step(
        audit_trace,
        step="access_control",
        status="passed",
        details="Order belongs to the authorised account.",
    )
    """

    audit_trace["steps"].append(
        {
            "timestamp": datetime.now().isoformat(
                timespec="seconds"
            ),
            "step": step,
            "status": status,
            "details": details,
        }
    )


# ---------------------------------------------------------------------
# Add reliability result
# ---------------------------------------------------------------------

def add_reliability_result(
    audit_trace: dict[str, Any],
    reliability: dict[str, Any],
) -> None:
    """
    Store the final trust and reliability assessment.
    """

    audit_trace["reliability"] = reliability.get(
        "confidence"
    )

    audit_trace["human_review_required"] = (
        reliability.get(
            "human_review_required",
            False,
        )
    )

    add_audit_step(
        audit_trace=audit_trace,
        step="reliability_assessment",
        status=reliability.get(
            "confidence",
            "unknown",
        ),
        details=reliability.get(
            "message",
            None,
        ),
    )


# ---------------------------------------------------------------------
# Finalize audit trace
# ---------------------------------------------------------------------

def finalize_audit_trace(
    audit_trace: dict[str, Any],
    final_status: str,
) -> None:
    """
    Mark an audit trace as completed and save it.
    """

    audit_trace["final_status"] = final_status

    audit_trace["completed_at"] = datetime.now().isoformat(
        timespec="seconds"
    )

    AUDIT_LOGS.append(audit_trace)


# ---------------------------------------------------------------------
# Retrieve audit logs
# ---------------------------------------------------------------------

def get_audit_logs() -> list[dict[str, Any]]:
    """
    Return all saved audit traces.

    A copy is returned so callers cannot directly modify
    the internal audit log list.
    """

    return AUDIT_LOGS.copy()


def get_recent_audit_logs(
    limit: int = 20,
) -> list[dict[str, Any]]:
    """
    Return the most recent audit traces.
    """

    if limit <= 0:
        return []

    return AUDIT_LOGS[-limit:][::-1]


# ---------------------------------------------------------------------
# Clear logs
# ---------------------------------------------------------------------

def clear_audit_logs() -> None:
    """
    Clear all in-memory audit logs.

    Intended for development and testing.
    """

    AUDIT_LOGS.clear()


# ---------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------

def main() -> None:
    """
    Run a simple standalone test of the audit layer.
    """

    print("=" * 70)
    print("TESTING AUDIT AND DECISION TRACE LAYER")
    print("=" * 70)

    # -------------------------------------------------------------
    # Test user context
    # -------------------------------------------------------------

    user_context = {
        "user_id": "northstar_user",
        "role": "customer",
        "account_id": "ACCT-001",
    }

    # -------------------------------------------------------------
    # Create trace
    # -------------------------------------------------------------

    audit_trace = create_audit_trace(
        question=(
            "Can ORD-1001 be cancelled without a fee?"
        ),
        user_context=user_context,
    )

    print("\n1. AUDIT TRACE CREATED")

    print(
        f"User: {audit_trace['user_id']}"
    )

    print(
        f"Account: {audit_trace['account_id']}"
    )

    # -------------------------------------------------------------
    # Add sample steps
    # -------------------------------------------------------------

    add_audit_step(
        audit_trace,
        step="question_validation",
        status="passed",
        details="Question is not empty.",
    )

    add_audit_step(
        audit_trace,
        step="access_control",
        status="passed",
        details=(
            "ORD-1001 belongs to authorised account "
            "ACCT-001."
        ),
    )

    add_audit_step(
        audit_trace,
        step="structured_data",
        status="retrieved",
        details=(
            "Order details and cancellation timing "
            "were retrieved."
        ),
    )

    add_audit_step(
        audit_trace,
        step="evidence_retrieval",
        status="completed",
        details=(
            "Account agreement and official cancellation "
            "policy were found."
        ),
    )

    add_audit_step(
        audit_trace,
        step="conflict_resolution",
        status="resolved",
        details=(
            "Account-specific agreement takes priority "
            "over official policy."
        ),
    )

    # -------------------------------------------------------------
    # Add reliability
    # -------------------------------------------------------------

    reliability = {
        "confidence": "high",
        "human_review_required": False,
        "message": (
            "The answer is supported by an account-specific "
            "agreement and relevant structured operational data."
        ),
    }

    add_reliability_result(
        audit_trace,
        reliability,
    )

    # -------------------------------------------------------------
    # Finalize
    # -------------------------------------------------------------

    finalize_audit_trace(
        audit_trace,
        final_status="completed",
    )

    print("\n2. AUDIT TRACE SAVED")

    print(
        f"Final status: {audit_trace['final_status']}"
    )

    print(
        f"Reliability: {audit_trace['reliability']}"
    )

    print(
        "Human review required:",
        audit_trace["human_review_required"],
    )

    # -------------------------------------------------------------
    # Retrieve logs
    # -------------------------------------------------------------

    logs = get_audit_logs()

    print("\n3. SAVED AUDIT LOGS")

    print(f"Total logs: {len(logs)}")

    print("\nAudit steps:")

    for audit_step in logs[0]["steps"]:

        print(
            f"- {audit_step['step']} | "
            f"{audit_step['status']} | "
            f"{audit_step['details']}"
        )

    print("\n" + "=" * 70)
    print("TESTING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()