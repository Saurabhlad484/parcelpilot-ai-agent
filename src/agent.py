"""
Main ParcelPilot AI agent.

This module orchestrates:
1. Access control checks
2. Structured data lookups from tools.py
3. Document retrieval from retrieval.py
4. Trust and reliability assessment from reliability.py
5. Audit and decision tracing from audit.py
6. State-changing escalation actions from actions.py
7. Final grounded answer generation from answer_generator.py

The current project uses mocked customer authentication.
Each session has an authorised account context, and access is enforced
before sensitive data is retrieved or passed to the language model.

Trust and reliability rules are applied before final answer generation
to determine source priority, confidence, conflicts, and whether
human review is required.

An internal audit trace records important decisions made during each
request. The audit trail is intended for internal operations and is
not shown to normal customer users.
"""

from __future__ import annotations

import re
from typing import Any

from src.actions import create_escalation
from src.answer_generator import generate_answer
from src.reliability import assess_answer_reliability
from src.retrieval import retrieve_governing_evidence
from src.audit import (
    add_audit_step,
    add_reliability_result,
    create_audit_trace,
    finalize_audit_trace,
)


from src.tools import (
    calculate_cancellation_timing,
    calculate_failed_pickup_timing,
    lookup_account,
    lookup_order,
    lookup_ticket,
)


# ---------------------------------------------------------------------
# Mock authentication / access control
# ---------------------------------------------------------------------

MOCK_USERS = {
    "northstar_user": {
        "role": "customer",
        "account_id": "ACCT-001",
    },
    "lumenworks_user": {
        "role": "customer",
        "account_id": "ACCT-002",
    },
}


def get_user_context(user_id: str) -> dict[str, str] | None:
    """
    Return the mocked authenticated user's context.

    In a production system, this information would come from the
    authentication/session layer rather than a local dictionary.
    """

    user = MOCK_USERS.get(user_id)

    if user is None:
        return None

    return {
        "user_id": user_id,
        "role": user["role"],
        "account_id": user["account_id"],
    }


def has_account_access(
    user_context: dict[str, Any],
    account_id: str,
) -> bool:
    """
    Enforce account-level access.

    Customer users may only access data belonging to their own account.
    """

    authorised_account = str(
        user_context["account_id"]
    ).upper()

    return authorised_account == account_id.upper()


# ---------------------------------------------------------------------
# ID extraction
# ---------------------------------------------------------------------

def extract_id(question: str, prefix: str) -> str | None:
    """
    Extract a ParcelPilot ID such as:
    ORD-1001
    TKT-502
    ACCT-001
    """

    pattern = rf"\b{prefix}-\d+\b"
    match = re.search(pattern, question.upper())

    if match:
        return match.group(0)

    return None


# ---------------------------------------------------------------------
# Topic detection
# ---------------------------------------------------------------------

def detect_topic(question: str) -> str:
    """
    Detect the broad topic of the user's question.
    """

    question_lower = question.lower()

    if any(
        word in question_lower
        for word in [
            "cancel",
            "cancellation",
            "cancellation fee",
        ]
    ):
        return "cancellation"

    if any(
        word in question_lower
        for word in [
            "service credit",
            "failed pickup",
            "pickup failed",
            "pickup delay",
        ]
    ):
        return "service_credit"

    if any(
        word in question_lower
        for word in [
            "bulk upload",
            "upload",
            "csv",
            "product",
        ]
    ):
        return "product_operations"

    if any(
        word in question_lower
        for word in [
            "p1",
            "p2",
            "p3",
            "sla",
            "response target",
            "support",
            "response time",
        ]
    ):
        return "support_sla"

    return "general"


# ---------------------------------------------------------------------
# Escalation detection
# ---------------------------------------------------------------------

def is_escalation_request(question: str) -> bool:
    """
    Detect whether the user is asking to create an escalation.
    """

    question_lower = question.lower()

    escalation_phrases = [
        "escalate",
        "create an escalation",
        "create escalation",
        "raise an escalation",
    ]

    return any(
        phrase in question_lower
        for phrase in escalation_phrases
    )


# ---------------------------------------------------------------------
# Scope validation
# ---------------------------------------------------------------------

def is_in_scope(
    question: str,
    topic: str,
    order_id: str | None,
    ticket_id: str | None,
    account_id: str | None,
) -> bool:
    """
    Check whether the question belongs to the ParcelPilot domain.
    """

    if order_id or ticket_id or account_id:
        return True

    if topic != "general":
        return True

    if is_escalation_request(question):
        return True

    return False


# ---------------------------------------------------------------------
# Required information validation
# ---------------------------------------------------------------------

def requires_order_id(
    topic: str,
    order_id: str | None,
) -> bool:
    """
    Check whether the question requires a specific order ID.
    """

    return topic in {
        "cancellation",
        "service_credit",
    } and order_id is None


# ---------------------------------------------------------------------
# Access validation
# ---------------------------------------------------------------------

def validate_user_access(
    user_context: dict[str, Any],
    order_id: str | None,
    ticket_id: str | None,
    account_id: str | None,
) -> str | None:
    """
    Enforce account-level access before sensitive data is returned.

    IMPORTANT:
    This happens before structured facts are sent to the LLM.
    """

    authorised_account = str(
        user_context["account_id"]
    ).upper()

    # ---------------------------------------------------------
    # Explicit account access
    # ---------------------------------------------------------
    if account_id is not None:
        if not has_account_access(
            user_context,
            account_id,
        ):
            return (
                f"Access denied. You are authorised to access only "
                f"account `{authorised_account}`."
            )

    # ---------------------------------------------------------
    # Order access
    # ---------------------------------------------------------
    if order_id is not None:
        order = lookup_order(order_id)

        if order is not None:
            order_account = str(
                order["account_id"]
            ).upper()

            if not has_account_access(
                user_context,
                order_account,
            ):
                return (
                    f"Access denied. Order `{order_id}` does not belong "
                    f"to your authorised account."
                )

    # ---------------------------------------------------------
    # Ticket access
    # ---------------------------------------------------------
    if ticket_id is not None:
        ticket = lookup_ticket(ticket_id)

        if ticket is not None:
            ticket_account = str(
                ticket["account_id"]
            ).upper()

            if not has_account_access(
                user_context,
                ticket_account,
            ):
                return (
                    f"Access denied. Ticket `{ticket_id}` does not belong "
                    f"to your authorised account."
                )

    return None


# ---------------------------------------------------------------------
# Identifier validation
# ---------------------------------------------------------------------

def validate_identifiers(
    order_id: str | None,
    ticket_id: str | None,
    account_id: str | None,
) -> str | None:
    """
    Validate that provided ParcelPilot IDs exist.

    Also validates that explicitly supplied account IDs match
    the related order or ticket.
    """

    order = None
    ticket = None

    # ---------------------------------------------------------
    # Validate order
    # ---------------------------------------------------------
    if order_id:
        order = lookup_order(order_id)

        if order is None:
            return (
                f"I couldn't find order `{order_id}`. "
                "Please check the order ID and try again."
            )

    # ---------------------------------------------------------
    # Validate ticket
    # ---------------------------------------------------------
    if ticket_id:
        ticket = lookup_ticket(ticket_id)

        if ticket is None:
            return (
                f"I couldn't find ticket `{ticket_id}`. "
                "Please check the ticket ID and try again."
            )

    # ---------------------------------------------------------
    # Validate account
    # ---------------------------------------------------------
    if account_id:
        account = lookup_account(account_id)

        if account is None:
            return (
                f"I couldn't find account `{account_id}`. "
                "Please check the account ID and try again."
            )

    # ---------------------------------------------------------
    # Validate order-account relationship
    # ---------------------------------------------------------
    if order is not None and account_id is not None:
        actual_account_id = str(
            order["account_id"]
        ).upper()

        if actual_account_id != account_id.upper():
            return (
                f"Account mismatch: order `{order_id}` belongs to "
                f"`{actual_account_id}`, not `{account_id.upper()}`. "
                "Please provide matching order and account details."
            )

    # ---------------------------------------------------------
    # Validate ticket-account relationship
    # ---------------------------------------------------------
    if ticket is not None and account_id is not None:
        actual_account_id = str(
            ticket["account_id"]
        ).upper()

        if actual_account_id != account_id.upper():
            return (
                f"Account mismatch: ticket `{ticket_id}` belongs to "
                f"`{actual_account_id}`, not `{account_id.upper()}`. "
                "Please provide matching ticket and account details."
            )

    return None


# ---------------------------------------------------------------------
# Structured data retrieval
# ---------------------------------------------------------------------

def get_structured_facts(
    topic: str,
    order_id: str | None,
    ticket_id: str | None,
    account_id: str | None,
) -> dict[str, Any]:
    """
    Collect relevant structured facts.

    Access control must already have passed before this function
    is called.

    This layer retrieves factual data only.
    """

    facts: dict[str, Any] = {}

    # ---------------------------------------------------------
    # Order facts
    # ---------------------------------------------------------
    if order_id:
        order = lookup_order(order_id)

        if order is not None:
            facts["order"] = order

            if account_id is None:
                account_id = order.get("account_id")

            if topic == "cancellation":
                facts["cancellation_timing"] = (
                    calculate_cancellation_timing(order_id)
                )

            if topic == "service_credit":
                facts["failed_pickup_timing"] = (
                    calculate_failed_pickup_timing(order_id)
                )

    # ---------------------------------------------------------
    # Ticket facts
    # ---------------------------------------------------------
    if ticket_id:
        ticket = lookup_ticket(ticket_id)

        if ticket is not None:
            facts["ticket"] = ticket

            if account_id is None:
                account_id = ticket.get("account_id")

    # ---------------------------------------------------------
    # Account facts
    # ---------------------------------------------------------
    if account_id:
        account = lookup_account(account_id)

        if account is not None:
            facts["account"] = account

    return facts


# ---------------------------------------------------------------------
# Trust and reliability context
# ---------------------------------------------------------------------

def build_reliability_context(
    structured_facts: dict[str, Any],
    evidence: Any,
) -> dict[str, Any]:
    """
    Build the source context used by the Trust & Reliability Layer.
    """

    source_context = {
        "account_contract": False,
        "official_policy": False,
        "structured_data": False,
        "retrieved_document": False,
        "historical_resolution": None,
    }

    # ---------------------------------------------------------
    # Structured operational data
    # ---------------------------------------------------------
    if structured_facts:
        source_context["structured_data"] = True

    # ---------------------------------------------------------
    # Account-specific contract
    # ---------------------------------------------------------
    account = structured_facts.get("account")

    if account:
        contract_file = account.get("contract_file")

        if (
            contract_file is not None
            and str(contract_file).strip()
            and str(contract_file).lower() != "nan"
        ):
            source_context["account_contract"] = True

    # ---------------------------------------------------------
    # Historical support resolution
    # ---------------------------------------------------------
    ticket = structured_facts.get("ticket")

    if ticket:
        historical_resolution = ticket.get(
            "historical_resolution"
        )

        if (
            historical_resolution is not None
            and str(historical_resolution).strip()
            and str(historical_resolution).lower() != "nan"
        ):
            source_context[
                "historical_resolution"
            ] = str(historical_resolution)

    # ---------------------------------------------------------
    # Retrieved evidence
    # ---------------------------------------------------------
    if evidence:
        source_context["retrieved_document"] = True

        evidence_text = str(evidence).lower()

        if any(
            keyword in evidence_text
            for keyword in [
                "official policy",
                "support policy",
                "cancellation policy",
                "service credit policy",
                "parcelpilot policy",
                "cancellation and service credit sop",
                "sop",
            ]
        ):
            source_context["official_policy"] = True

    return source_context


# ---------------------------------------------------------------------
# Audit helper
# ---------------------------------------------------------------------

def complete_audit(
    audit_trace: dict[str, Any],
    final_status: str,
) -> None:
    """
    Finalize and save an audit trace.

    This helper prevents audit finalization logic from being repeated
    throughout the main agent pipeline.
    """

    finalize_audit_trace(
        audit_trace=audit_trace,
        final_status=final_status,
    )


# ---------------------------------------------------------------------
# Main agent pipeline
# ---------------------------------------------------------------------

def run_agent(
    question: str,
    pending_action: dict[str, Any] | None = None,
    user_context: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any] | None]:
    """
    Run the complete ParcelPilot AI agent pipeline.

    The authenticated user context is provided by the interface
    and is checked before sensitive account, order, or ticket data
    is retrieved or passed to the language model.
    """

    # ---------------------------------------------------------
    # STEP 1: Validate question
    # ---------------------------------------------------------
    if not question.strip():
        return "Please enter a question.", pending_action

    # ---------------------------------------------------------
    # STEP 2: Validate authenticated user context
    # ---------------------------------------------------------
    if user_context is None:
        return (
            "Access denied. No authenticated user context was provided."
        ), None

    if (
        "account_id" not in user_context
        or "role" not in user_context
    ):
        return (
            "Access denied. Invalid authenticated user context."
        ), None

    # ---------------------------------------------------------
    # STEP 3: Create audit trace
    # ---------------------------------------------------------
    audit_trace = create_audit_trace(
        question=question,
        user_context=user_context,
    )

    add_audit_step(
        audit_trace,
        step="question_validation",
        status="passed",
        details="Question is not empty and authenticated user context is valid.",
    )

    # ---------------------------------------------------------
    # STEP 4: Handle pending action confirmation FIRST
    # ---------------------------------------------------------
    if pending_action is not None:

        answer_lower = question.strip().lower()

        confirmation_words = {
            "yes",
            "confirm",
            "confirmed",
            "proceed",
        }

        cancellation_words = {
            "no",
            "cancel",
            "stop",
        }

        # -----------------------------------------------------
        # Explicit confirmation
        # -----------------------------------------------------
        if answer_lower in confirmation_words:

            if pending_action.get("action") == "create_escalation":

                if not has_account_access(
                    user_context,
                    pending_action["account_id"],
                ):
                    add_audit_step(
                        audit_trace,
                        step="access_control",
                        status="failed",
                        details=(
                            "User was not authorised to create an "
                            "escalation for the requested account."
                        ),
                    )

                    complete_audit(
                        audit_trace,
                        final_status="access_denied",
                    )

                    return (
                        "Access denied. You are not authorised to create "
                        "an escalation for this account."
                    ), None

                add_audit_step(
                    audit_trace,
                    step="access_control",
                    status="passed",
                    details=(
                        "Account access was re-checked before creating "
                        "the escalation."
                    ),
                )

                escalation = create_escalation(
                    ticket_id=pending_action["ticket_id"],
                    account_id=pending_action["account_id"],
                    reason=pending_action["reason"],
                )

                add_audit_step(
                    audit_trace,
                    step="state_changing_action",
                    status="completed",
                    details=(
                        f"Escalation {escalation['escalation_id']} "
                        "was created after explicit confirmation."
                    ),
                )

                complete_audit(
                    audit_trace,
                    final_status="completed",
                )

                return (
                    "Escalation created successfully.\n\n"
                    f"Escalation ID: {escalation['escalation_id']}\n"
                    f"Ticket: {escalation['ticket_id']}\n"
                    f"Status: {escalation['status']}"
                ), None

        # -----------------------------------------------------
        # Explicit cancellation
        # -----------------------------------------------------
        if answer_lower in cancellation_words:

            add_audit_step(
                audit_trace,
                step="state_changing_action",
                status="cancelled",
                details=(
                    "The user declined confirmation. "
                    "No escalation was created."
                ),
            )

            complete_audit(
                audit_trace,
                final_status="cancelled",
            )

            return (
                "Okay. The escalation was not created."
            ), None

        # -----------------------------------------------------
        # Keep waiting for explicit confirmation
        # -----------------------------------------------------
        add_audit_step(
            audit_trace,
            step="state_changing_action",
            status="awaiting_confirmation",
            details=(
                "Escalation remains pending explicit user confirmation."
            ),
        )

        complete_audit(
            audit_trace,
            final_status="awaiting_confirmation",
        )

        return (
            "The escalation has not been created. "
            "Reply `yes` to confirm or `no` to cancel."
        ), pending_action

    # ---------------------------------------------------------
    # STEP 5: Extract IDs
    # ---------------------------------------------------------
    order_id = extract_id(question, "ORD")
    ticket_id = extract_id(question, "TKT")
    account_id = extract_id(question, "ACCT")

    # ---------------------------------------------------------
    # STEP 6: Detect topic
    # ---------------------------------------------------------
    topic = detect_topic(question)

    # ---------------------------------------------------------
    # STEP 7: Check scope
    # ---------------------------------------------------------
    if not is_in_scope(
        question=question,
        topic=topic,
        order_id=order_id,
        ticket_id=ticket_id,
        account_id=account_id,
    ):
        add_audit_step(
            audit_trace,
            step="scope_validation",
            status="out_of_scope",
            details=(
                "The request does not belong to the ParcelPilot domain."
            ),
        )

        complete_audit(
            audit_trace,
            final_status="out_of_scope",
        )

        return (
            "I can help with ParcelPilot orders, accounts, support policies, "
            "cancellations, service credits, product issues, and escalations. "
            "I cannot answer unrelated questions."
        ), None

    # ---------------------------------------------------------
    # STEP 8: Check for missing required order ID
    # ---------------------------------------------------------
    if requires_order_id(
        topic=topic,
        order_id=order_id,
    ):
        add_audit_step(
            audit_trace,
            step="identifier_validation",
            status="missing_required_identifier",
            details=(
                f"An order ID is required for the {topic} request."
            ),
        )

        complete_audit(
            audit_trace,
            final_status="incomplete_request",
        )

        if topic == "cancellation":
            return (
                "Please provide the order ID, for example `ORD-1001`, "
                "so I can check the order details and applicable "
                "cancellation rules."
            ), None

        if topic == "service_credit":
            return (
                "Please provide the order ID, for example `ORD-2002`, "
                "so I can check the pickup details and applicable "
                "service-credit rules."
            ), None

    # ---------------------------------------------------------
    # STEP 9: Validate identifiers
    # ---------------------------------------------------------
    validation_error = validate_identifiers(
        order_id=order_id,
        ticket_id=ticket_id,
        account_id=account_id,
    )

    if validation_error is not None:

        add_audit_step(
            audit_trace,
            step="identifier_validation",
            status="failed",
            details=validation_error,
        )

        complete_audit(
            audit_trace,
            final_status="validation_failed",
        )

        return validation_error, None

    add_audit_step(
        audit_trace,
        step="identifier_validation",
        status="passed",
        details="All supplied identifiers were validated successfully.",
    )

    # ---------------------------------------------------------
    # STEP 10: Enforce access control
    # ---------------------------------------------------------
    access_error = validate_user_access(
        user_context=user_context,
        order_id=order_id,
        ticket_id=ticket_id,
        account_id=account_id,
    )

    if access_error is not None:

        add_audit_step(
            audit_trace,
            step="access_control",
            status="failed",
            details=access_error,
        )

        complete_audit(
            audit_trace,
            final_status="access_denied",
        )

        return access_error, None

    add_audit_step(
        audit_trace,
        step="access_control",
        status="passed",
        details=(
            "The requested account, order, or ticket is authorised "
            "for the current user."
        ),
    )

    # ---------------------------------------------------------
    # STEP 11: Prepare escalation and request confirmation
    # ---------------------------------------------------------
    if is_escalation_request(question):

        if ticket_id is None:

            add_audit_step(
                audit_trace,
                step="state_changing_action",
                status="missing_ticket_id",
                details=(
                    "An escalation request was detected but no ticket ID "
                    "was provided."
                ),
            )

            complete_audit(
                audit_trace,
                final_status="incomplete_request",
            )

            return (
                "Please provide the ticket ID, for example `TKT-502`, "
                "so I can prepare the escalation."
            ), None

        ticket = lookup_ticket(ticket_id)

        if ticket is None:

            add_audit_step(
                audit_trace,
                step="state_changing_action",
                status="failed",
                details=(
                    f"Ticket {ticket_id} could not be found."
                ),
            )

            complete_audit(
                audit_trace,
                final_status="validation_failed",
            )

            return (
                f"I couldn't find ticket `{ticket_id}`. "
                "Please check the ticket ID and try again."
            ), None

        actual_account_id = str(
            ticket["account_id"]
        ).upper()

        if not has_account_access(
            user_context,
            actual_account_id,
        ):

            add_audit_step(
                audit_trace,
                step="access_control",
                status="failed",
                details=(
                    f"Ticket {ticket_id} does not belong to the "
                    "authorised account."
                ),
            )

            complete_audit(
                audit_trace,
                final_status="access_denied",
            )

            return (
                f"Access denied. Ticket `{ticket_id}` does not belong "
                "to your authorised account."
            ), None

        action = {
            "action": "create_escalation",
            "ticket_id": ticket_id,
            "account_id": actual_account_id,
            "reason": f"Escalation requested: {question}",
        }

        add_audit_step(
            audit_trace,
            step="state_changing_action",
            status="awaiting_confirmation",
            details=(
                f"Escalation prepared for ticket {ticket_id}. "
                "Explicit confirmation is required before creation."
            ),
        )

        complete_audit(
            audit_trace,
            final_status="awaiting_confirmation",
        )

        return (
            f"I can create an escalation for ticket `{ticket_id}`.\n\n"
            "Reply `yes` to confirm or `no` to cancel."
        ), action

    # ---------------------------------------------------------
    # STEP 12: Get structured facts
    # ---------------------------------------------------------
    structured_facts = get_structured_facts(
        topic=topic,
        order_id=order_id,
        ticket_id=ticket_id,
        account_id=account_id,
    )

    add_audit_step(
        audit_trace,
        step="structured_data",
        status="retrieved",
        details=(
            "Relevant structured operational data was retrieved."
            if structured_facts
            else "No relevant structured operational data was found."
        ),
    )

    # ---------------------------------------------------------
    # STEP 13: Resolve authoritative account
    # ---------------------------------------------------------
    authoritative_account_id = str(
        user_context["account_id"]
    ).upper()

    if structured_facts.get("order"):

        actual_order_account = structured_facts[
            "order"
        ].get("account_id")

        if actual_order_account:
            authoritative_account_id = str(
                actual_order_account
            ).upper()

    elif structured_facts.get("ticket"):

        actual_ticket_account = structured_facts[
            "ticket"
        ].get("account_id")

        if actual_ticket_account:
            authoritative_account_id = str(
                actual_ticket_account
            ).upper()

    # ---------------------------------------------------------
    # STEP 14: Retrieve authority-aware document evidence
    # ---------------------------------------------------------
    evidence = retrieve_governing_evidence(
        question=question,
        account_id=authoritative_account_id,
    )

    add_audit_step(
        audit_trace,
        step="evidence_retrieval",
        status="completed",
        details=(
            "Authority-aware governing evidence was retrieved."
            if evidence
            else "No governing document evidence was found."
        ),
    )

    # ---------------------------------------------------------
    # STEP 15: Build reliability source context
    # ---------------------------------------------------------
    reliability_context = build_reliability_context(
        structured_facts=structured_facts,
        evidence=evidence,
    )

    # ---------------------------------------------------------
    # STEP 16: Assess trust and reliability
    # ---------------------------------------------------------
    reliability_assessment = assess_answer_reliability(
        account_contract_available=
            reliability_context["account_contract"],

        official_policy_available=
            reliability_context["official_policy"],

        structured_data_available=
            reliability_context["structured_data"],

        retrieved_document_available=
            reliability_context["retrieved_document"],

        historical_resolution=
            reliability_context["historical_resolution"],
    )

    add_reliability_result(
        audit_trace,
        reliability_assessment,
    )

    # ---------------------------------------------------------
    # STEP 17: Handle unresolved conflicts
    # ---------------------------------------------------------
    if reliability_assessment.get(
        "human_review_required"
    ):

        add_audit_step(
            audit_trace,
            step="human_review",
            status="required",
            details=(
                "The available information cannot be safely resolved "
                "using the current source-priority rules."
            ),
        )

        complete_audit(
            audit_trace,
            final_status="human_review_required",
        )

        return (
            "I found conflicting or uncertain information that cannot "
            "be safely resolved using the available source-priority "
            "rules.\n\n"
            "Human review is required before I can provide a final "
            "decision."
        ), None

    # ---------------------------------------------------------
    # STEP 18: Generate grounded answer
    # ---------------------------------------------------------
    answer = generate_answer(
        question=question,
        structured_facts=structured_facts,
        retrieved_evidence=evidence,
        reliability_assessment=reliability_assessment,
    )

    add_audit_step(
        audit_trace,
        step="answer_generation",
        status="completed",
        details=(
            "A grounded final answer was generated using the available "
            "authorised facts and governing evidence."
        ),
    )

    # ---------------------------------------------------------
    # STEP 19: Save completed audit trace
    # ---------------------------------------------------------
    complete_audit(
        audit_trace,
        final_status="completed",
    )

    return answer, None


# ---------------------------------------------------------------------
# Interactive CLI
# ---------------------------------------------------------------------

def main() -> None:
    """
    Run an interactive ParcelPilot AI agent.

    The CLI uses mocked authentication and passes the authenticated
    user context directly into run_agent().
    """

    print("=" * 70)
    print("PARCELPILOT AI AGENT")
    print("=" * 70)

    print("\nMock users:")
    print("1. northstar_user -> ACCT-001")
    print("2. lumenworks_user -> ACCT-002")

    user_id = input(
        "\nEnter user ID "
        "(press Enter for northstar_user): "
    ).strip()

    if not user_id:
        user_id = "northstar_user"

    user_context = get_user_context(user_id)

    if user_context is None:
        print("\nAccess denied. Unknown user.")
        return

    print(
        f"\nLogged in as: {user_context['user_id']} "
        f"({user_context['role']}, "
        f"{user_context['account_id']})"
    )

    print(
        "\nAsk a question about your ParcelPilot orders, tickets, "
        "support policies, product issues, or escalations."
    )

    print("Type 'exit' to quit.\n")

    pending_action = None

    while True:

        try:
            question = input("You: ").strip()

        except KeyboardInterrupt:
            print("\n\nParcelPilot AI Agent closed.")
            break

        except EOFError:
            print("\n\nParcelPilot AI Agent closed.")
            break

        if question.lower() in {"exit", "quit"}:
            print("\nParcelPilot AI Agent closed.")
            break

        if not question:
            print("\nPlease enter a question.\n")
            continue

        try:
            print("\nThinking...\n")

            answer, pending_action = run_agent(
                question=question,
                pending_action=pending_action,
                user_context=user_context,
            )

            print("ParcelPilot AI:")
            print(answer)

            print("\n" + "-" * 70 + "\n")

        except Exception as error:
            print("\nAn error occurred:")
            print(error)

            print("\n" + "-" * 70 + "\n")


if __name__ == "__main__":
    main()