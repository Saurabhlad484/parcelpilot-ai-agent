"""
ParcelPilot AI Agent - Streamlit Interface

Provides two interfaces:

1. Customer AI Assistant
   - Mock user authentication
   - Account-scoped access control
   - Natural-language questions
   - Document retrieval
   - Structured data lookup and calculations
   - Escalation confirmation workflow
   - Chat history
   - Switching users without restarting

2. Internal Operations Dashboard
   - Open support issues
   - Recurring issue detection
   - Unusual support patterns
   - Historical resolution risks
   - Proactive issue summary
"""

import streamlit as st

from src.agent import run_agent
from src.tools import get_proactive_issue_summary


# ---------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="ParcelPilot AI",
    page_icon="📦",
    layout="centered",
)


# ---------------------------------------------------------------------
# Mock users
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
    "ops_user": {
        "role": "operations",
        "account_id": None,
    },
}


# ---------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "user_context" not in st.session_state:
    st.session_state.user_context = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_action" not in st.session_state:
    st.session_state.pending_action = None


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def reset_chat() -> None:
    """
    Clear chat history and any action waiting for confirmation.
    """

    st.session_state.messages = []
    st.session_state.pending_action = None


def logout() -> None:
    """
    End the current session and return to the login screen.
    """

    st.session_state.authenticated = False
    st.session_state.user_context = None
    reset_chat()


# ---------------------------------------------------------------------
# Login page
# ---------------------------------------------------------------------

if not st.session_state.authenticated:

    st.title("📦 ParcelPilot AI")
    st.subheader("Support & Operations Assistant")

    st.write(
        "A grounded AI assistant for ParcelPilot support, "
        "operations, and proactive issue detection."
    )

    st.divider()

    st.write("### Login")

    selected_user = st.selectbox(
        "Select a demo user",
        options=list(MOCK_USERS.keys()),
    )

    user = MOCK_USERS[selected_user]

    # -------------------------------------------------------------
    # Show information based on selected user type
    # -------------------------------------------------------------

    if user["role"] == "operations":

        st.info(
            "Role: Operations | "
            "Access: Internal Operations Dashboard"
        )

    else:

        st.info(
            f"Role: {user['role']} | "
            f"Authorised Account: {user['account_id']}"
        )

    # -------------------------------------------------------------
    # Login button
    # -------------------------------------------------------------

    if st.button("Log in", type="primary"):

        st.session_state.authenticated = True

        st.session_state.user_context = {
            "user_id": selected_user,
            "role": user["role"],
            "account_id": user["account_id"],
        }

        reset_chat()

        st.rerun()


# ---------------------------------------------------------------------
# Authenticated application
# ---------------------------------------------------------------------

else:

    # Get authenticated user context.
    user_context = st.session_state.user_context

    # -----------------------------------------------------------------
    # Safety check
    # -----------------------------------------------------------------

    if user_context is None:

        st.error("Authentication error. Please log in again.")

        if st.button("Return to Login"):
            logout()
            st.rerun()

        st.stop()


    # =================================================================
    # INTERNAL OPERATIONS DASHBOARD
    # =================================================================

    if user_context["role"] == "operations":

        # -------------------------------------------------------------
        # Sidebar
        # -------------------------------------------------------------

        with st.sidebar:

            st.title("📦 ParcelPilot AI")

            st.write("### Logged in as")

            st.write(
                f"**User:** {user_context['user_id']}"
            )

            st.write("**Role:** Operations")

            st.divider()

            if st.button(
                "Switch User",
                use_container_width=True,
            ):
                logout()
                st.rerun()

            if st.button(
                "Log Out",
                use_container_width=True,
            ):
                logout()
                st.rerun()


        # -------------------------------------------------------------
        # Dashboard heading
        # -------------------------------------------------------------

        st.title("📊 Operations Dashboard")

        st.caption(
            "Internal view for authorised support and operations users "
            "to identify recurring, unusual, and potentially high-risk "
            "issues across support activity."
        )

        st.divider()


        # -------------------------------------------------------------
        # Load proactive issue data
        # -------------------------------------------------------------

        try:

            summary = get_proactive_issue_summary()

        except Exception as error:

            st.error(
                "Unable to load proactive issue data."
            )

            st.exception(error)

            st.stop()


        # -------------------------------------------------------------
        # KPI metrics
        # -------------------------------------------------------------

        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)

        col1.metric(
            "Open Tickets",
            summary.get("open_ticket_count", 0),
        )

        col2.metric(
            "Recurring Issues",
            summary.get("recurring_issue_count", 0),
        )

        col3.metric(
            "Unusual Patterns",
            summary.get("unusual_pattern_count", 0),
        )

        col4.metric(
            "Historical Answer Risks",
            summary.get(
                "historical_resolution_risk_count",
                0,
            ),
        )

        st.divider()


        # -------------------------------------------------------------
        # Open support issues
        # -------------------------------------------------------------

        st.subheader("🔴 Open Support Issues")

        open_issues = summary.get(
            "open_ticket_issues",
            [],
        )

        if open_issues:

            st.dataframe(
                open_issues,
                use_container_width=True,
                hide_index=True,
            )

        else:

            st.success(
                "No open support issues detected."
            )


        # -------------------------------------------------------------
        # Recurring issues
        # -------------------------------------------------------------

        st.divider()

        st.subheader("🔁 Recurring Issues")

        recurring_issues = summary.get(
            "recurring_issues",
            [],
        )

        if recurring_issues:

            for issue in recurring_issues:

                with st.container():

                    st.warning(
                        issue.get(
                            "message",
                            "Potential recurring issue detected.",
                        )
                    )

                    if issue.get("type"):

                        st.write(
                            f"**Issue Type:** "
                            f"{issue['type']}"
                        )

                    if issue.get("ticket_ids"):

                        st.write(
                            "**Related Tickets:** "
                            + ", ".join(
                                issue["ticket_ids"]
                            )
                        )

        else:

            st.info(
                "No recurring issues were detected using "
                "the current detection criteria."
            )


        # -------------------------------------------------------------
        # Unusual support patterns
        # -------------------------------------------------------------

        st.divider()

        st.subheader("⚠️ Unusual Support Patterns")

        unusual_patterns = summary.get(
            "unusual_patterns",
            [],
        )

        if unusual_patterns:

            for pattern in unusual_patterns:

                with st.container():

                    priority = pattern.get(
                        "priority",
                        "unknown",
                    ).upper()

                    st.warning(
                        pattern.get(
                            "message",
                            "Unusual support pattern detected.",
                        )
                    )

                    st.write(
                        f"**Priority:** {priority}"
                    )

                    if pattern.get("account_id"):

                        st.write(
                            f"**Affected Account:** "
                            f"{pattern['account_id']}"
                        )

                    if pattern.get("open_ticket_count") is not None:

                        st.write(
                            f"**Open Ticket Count:** "
                            f"{pattern['open_ticket_count']}"
                        )

                    if pattern.get("ticket_ids"):

                        st.write(
                            "**Related Tickets:** "
                            + ", ".join(
                                pattern["ticket_ids"]
                            )
                        )

        else:

            st.success(
                "No unusual support patterns detected."
            )


        # -------------------------------------------------------------
        # Trust and reliability risks
        # -------------------------------------------------------------

        st.divider()

        st.subheader("🛡️ Trust & Reliability Risks")

        st.caption(
            "Historical support answers are treated as context only. "
            "They must be checked against current governing policies "
            "and any applicable account-specific agreement."
        )

        historical_risks = summary.get(
            "historical_resolution_risks",
            [],
        )

        if historical_risks:

            for risk in historical_risks:

                ticket_id = risk.get(
                    "ticket_id",
                    "Unknown Ticket",
                )

                subject = risk.get(
                    "subject",
                    "Historical Resolution Risk",
                )

                with st.expander(
                    f"{ticket_id} — {subject}"
                ):

                    if risk.get("account_id"):

                        st.write(
                            f"**Account:** "
                            f"{risk['account_id']}"
                        )

                    st.write(
                        "**Previous Resolution:**"
                    )

                    st.write(
                        risk.get(
                            "historical_resolution",
                            "No historical resolution available.",
                        )
                    )

                    st.warning(
                        risk.get(
                            "risk",
                            "Historical guidance should be "
                            "validated against current policy.",
                        )
                    )

        else:

            st.success(
                "No historical resolution risks detected."
            )


        # -------------------------------------------------------------
        # Refresh dashboard
        # -------------------------------------------------------------

        st.divider()

        if st.button(
            "🔄 Refresh Dashboard",
            type="primary",
        ):
            st.rerun()


        # Operations users must not continue into customer chat.
        st.stop()


    # =================================================================
    # CUSTOMER AI CHAT
    # =================================================================

    else:

        # -------------------------------------------------------------
        # Sidebar
        # -------------------------------------------------------------

        with st.sidebar:

            st.title("📦 ParcelPilot AI")

            st.write("### Logged in as")

            st.write(
                f"**User:** {user_context['user_id']}"
            )

            st.write(
                f"**Role:** {user_context['role']}"
            )

            st.write(
                f"**Account:** {user_context['account_id']}"
            )

            st.divider()

            # Switch User
            if st.button(
                "Switch User",
                use_container_width=True,
            ):
                logout()
                st.rerun()

            # Clear Chat
            if st.button(
                "Clear Chat",
                use_container_width=True,
            ):
                reset_chat()
                st.rerun()

            # Log Out
            if st.button(
                "Log Out",
                use_container_width=True,
            ):
                logout()
                st.rerun()


        # -------------------------------------------------------------
        # Main heading
        # -------------------------------------------------------------

        st.title("📦 ParcelPilot AI")

        st.caption(
            "Ask questions about your orders, tickets, cancellations, "
            "service credits, support policies, product issues, "
            "or request an escalation."
        )


        # -------------------------------------------------------------
        # Display chat history
        # -------------------------------------------------------------

        for message in st.session_state.messages:

            with st.chat_message(message["role"]):

                st.markdown(
                    message["content"]
                )


        # -------------------------------------------------------------
        # Chat input
        # -------------------------------------------------------------

        prompt = st.chat_input(
            "Type your message..."
        )

        if prompt:

            # ---------------------------------------------------------
            # Display and save user message
            # ---------------------------------------------------------

            with st.chat_message("user"):

                st.markdown(prompt)

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": prompt,
                }
            )


            # ---------------------------------------------------------
            # Run agent
            # ---------------------------------------------------------

            with st.chat_message("assistant"):

                with st.spinner("Thinking..."):

                    try:

                        answer, pending_action = run_agent(
                            question=prompt,
                            pending_action=(
                                st.session_state.pending_action
                            ),
                            user_context=user_context,
                        )

                        # Store action for explicit confirmation.
                        st.session_state.pending_action = (
                            pending_action
                        )

                        st.markdown(answer)

                    except Exception as error:

                        answer = (
                            "An error occurred while processing "
                            "your request:\n\n"
                            f"`{error}`"
                        )

                        st.error(answer)


            # ---------------------------------------------------------
            # Save assistant response
            # ---------------------------------------------------------

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )