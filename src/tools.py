import pandas as pd

from src.data_loader import load_data


# ---------------------------------------------------------------------
# Account lookups
# ---------------------------------------------------------------------

def lookup_account(account_id):
    """
    Look up one account by account_id.

    Returns:
        dict if found
        None if not found
    """
    data = load_data()
    accounts = data["accounts"]

    result = accounts[
        accounts["account_id"].str.upper() == account_id.upper()
    ]

    if result.empty:
        return None

    return result.iloc[0].to_dict()


# ---------------------------------------------------------------------
# Order lookups
# ---------------------------------------------------------------------

def lookup_order(order_id):
    """
    Look up one order by order_id.

    Returns:
        dict if found
        None if not found
    """
    data = load_data()
    orders = data["orders"]

    result = orders[
        orders["order_id"].str.upper() == order_id.upper()
    ]

    if result.empty:
        return None

    return result.iloc[0].to_dict()


# ---------------------------------------------------------------------
# Ticket lookups
# ---------------------------------------------------------------------

def lookup_ticket(ticket_id):
    """
    Look up one ticket by ticket_id.

    Returns:
        dict if found
        None if not found
    """
    data = load_data()
    tickets = data["tickets"]

    result = tickets[
        tickets["ticket_id"].str.upper() == ticket_id.upper()
    ]

    if result.empty:
        return None

    return result.iloc[0].to_dict()


# ---------------------------------------------------------------------
# Relationship lookups
# ---------------------------------------------------------------------

def get_account_for_order(order_id):
    """
    Find an order and then return the account that owns it.
    """
    order = lookup_order(order_id)

    if order is None:
        return None

    return lookup_account(order["account_id"])


def get_account_for_ticket(ticket_id):
    """
    Find a ticket and then return the account that owns it.
    """
    ticket = lookup_ticket(ticket_id)

    if ticket is None:
        return None

    return lookup_account(ticket["account_id"])


def get_orders_for_account(account_id):
    """
    Return all orders belonging to an account.
    """
    data = load_data()
    orders = data["orders"]

    result = orders[
        orders["account_id"].str.upper() == account_id.upper()
    ]

    return result.to_dict(orient="records")


def get_tickets_for_account(account_id):
    """
    Return all tickets belonging to an account.
    """
    data = load_data()
    tickets = data["tickets"]

    result = tickets[
        tickets["account_id"].str.upper() == account_id.upper()
    ]

    return result.to_dict(orient="records")


# ---------------------------------------------------------------------
# Dataset snapshot time
# ---------------------------------------------------------------------

def get_dataset_snapshot_time():
    """
    Read the dataset snapshot time from the workbook README data.

    Returns:
        pandas Timestamp or None
    """

    data = load_data()
    readme = data["readme"]

    first_column = readme.columns[0]
    second_column = readme.columns[1]

    snapshot_row = readme[
        readme[first_column].astype(str).str.strip().str.lower()
        == "dataset snapshot"
    ]

    if snapshot_row.empty:
        return None

    snapshot_value = snapshot_row.iloc[0][second_column]

    # The workbook value includes the timezone name.
    # For calculations in this assessment, convert the date/time portion.
    snapshot_datetime = str(snapshot_value).replace(
        " Asia/Kolkata",
        ""
    )

    return pd.to_datetime(snapshot_datetime)


# ---------------------------------------------------------------------
# Failed pickup timing
# ---------------------------------------------------------------------

def calculate_failed_pickup_timing(order_id):
    """
    Calculate how long an order has remained without pickup
    after the scheduled pickup window ended.

    This function returns structured facts only.
    It does not decide final service-credit eligibility.
    """

    order = lookup_order(order_id)

    if order is None:
        return None

    snapshot_time = get_dataset_snapshot_time()

    if snapshot_time is None:
        return {
            "order_id": order["order_id"],
            "message": "Dataset snapshot time could not be determined.",
        }

    pickup_window_end = order["pickup_window_end"]
    pickup_actual_at = order["pickup_actual_at"]

    if pd.isna(pickup_window_end):
        return {
            "order_id": order["order_id"],
            "message": "Pickup window end time is unavailable.",
        }

    # If pickup actually happened, calculate whether it was late.
    if not pd.isna(pickup_actual_at):
        reference_time = pickup_actual_at
        reference_type = "pickup_actual_at"
    else:
        # If pickup has not happened, use the dataset snapshot time.
        reference_time = snapshot_time
        reference_type = "dataset_snapshot"

    minutes_after_window_end = (
        reference_time - pickup_window_end
    ).total_seconds() / 60

    return {
        "order_id": order["order_id"],
        "account_id": order["account_id"],
        "status": order["status"],
        "pickup_window_end": pickup_window_end,
        "reference_time": reference_time,
        "reference_type": reference_type,
        "minutes_after_window_end": minutes_after_window_end,
        "hours_after_window_end": minutes_after_window_end / 60,
        "carrier_fault": order["carrier_fault"],
        "customer_fault": order["customer_fault"],
        "message": "Failed-pickup timing calculated successfully.",
    }


# ---------------------------------------------------------------------
# Cancellation timing
# ---------------------------------------------------------------------

def calculate_cancellation_timing(order_id):
    """
    Calculate the time between booking and the cancellation request.

    This function only returns facts from structured data.
    It does not decide whether a cancellation fee applies.
    """

    order = lookup_order(order_id)

    if order is None:
        return None

    booked_at = order["booked_at"]
    cancellation_requested_at = order["cancellation_requested_at"]

    if pd.isna(cancellation_requested_at):
        return {
            "order_id": order["order_id"],
            "account_id": order["account_id"],
            "status": order["status"],
            "booking_time": booked_at,
            "cancellation_requested_at": None,
            "minutes_after_booking": None,
            "message": "No cancellation request timestamp is available.",
        }

    minutes_after_booking = (
        cancellation_requested_at - booked_at
    ).total_seconds() / 60

    return {
        "order_id": order["order_id"],
        "account_id": order["account_id"],
        "status": order["status"],
        "booking_time": booked_at,
        "cancellation_requested_at": cancellation_requested_at,
        "minutes_after_booking": minutes_after_booking,
        "message": "Cancellation timing calculated successfully.",
    }


# =====================================================================
# PROACTIVE ISSUE DETECTION
# =====================================================================

# ---------------------------------------------------------------------
# Helper: normalize text for simple issue grouping
# ---------------------------------------------------------------------

def normalize_issue_text(text):
    """
    Normalize issue text so similar ticket subjects can be grouped.

    This is intentionally data-driven and does not hard-code
    specific ticket IDs or customer names.
    """

    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Remove punctuation.
    text = "".join(
        character if character.isalnum() or character.isspace()
        else " "
        for character in text
    )

    # Remove very common words that add little meaning.
    stop_words = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "for",
        "to",
        "of",
        "and",
        "in",
        "on",
        "with",
        "after",
        "how",
        "do",
        "we",
        "my",
        "our",
    }

    words = [
        word
        for word in text.split()
        if word not in stop_words
    ]

    return " ".join(words)


# ---------------------------------------------------------------------
# Helper: create an issue signature
# ---------------------------------------------------------------------

def get_issue_signature(ticket):
    """
    Create a simple comparable signature for a ticket.

    The signature is based on the ticket subject and is used
    to identify potentially recurring issues.
    """

    subject = ticket.get("subject", "")

    normalized_subject = normalize_issue_text(subject)

    return normalized_subject


# ---------------------------------------------------------------------
# Detect recurring or similar issues
# ---------------------------------------------------------------------

def detect_recurring_issues(min_occurrences=2):
    """
    Identify similar issues appearing multiple times in support data.

    Returns groups of tickets with matching normalized issue signatures.

    This is a simple deterministic first version. A future version
    could use embeddings and semantic similarity.
    """

    data = load_data()
    tickets = data["tickets"].copy()

    if tickets.empty:
        return []

    tickets["issue_signature"] = tickets.apply(
        lambda row: get_issue_signature(row.to_dict()),
        axis=1,
    )

    grouped = (
        tickets.groupby("issue_signature", dropna=False)
        .agg(
            ticket_count=("ticket_id", "count"),
            ticket_ids=("ticket_id", list),
            account_ids=("account_id", list),
            subjects=("subject", list),
            statuses=("status", list),
        )
        .reset_index()
    )

    recurring_groups = grouped[
        grouped["ticket_count"] >= min_occurrences
    ]

    results = []

    for _, row in recurring_groups.iterrows():

        unique_accounts = sorted(
            {
                str(account_id)
                for account_id in row["account_ids"]
            }
        )

        results.append(
            {
                "issue_signature": row["issue_signature"],
                "ticket_count": int(row["ticket_count"]),
                "ticket_ids": row["ticket_ids"],
                "account_ids": unique_accounts,
                "affected_account_count": len(unique_accounts),
                "subjects": row["subjects"],
                "statuses": row["statuses"],
            }
        )

    return results


# ---------------------------------------------------------------------
# Detect open tickets requiring attention
# ---------------------------------------------------------------------

def detect_open_ticket_issues():
    """
    Identify currently open support tickets.

    The function returns structured facts for the internal
    operations view. It does not make unsupported policy decisions.
    """

    data = load_data()
    tickets = data["tickets"].copy()

    if tickets.empty:
        return []

    open_tickets = tickets[
        tickets["status"].astype(str).str.lower() == "open"
    ]

    results = []

    for _, ticket in open_tickets.iterrows():

        results.append(
            {
                "ticket_id": ticket["ticket_id"],
                "account_id": ticket["account_id"],
                "created_at": ticket["created_at"],
                "subject": ticket["subject"],
                "description": ticket["description"],
                "channel": ticket["channel"],
                "assigned_to": ticket["assigned_to"],
                "last_customer_message_at": (
                    ticket["last_customer_message_at"]
                ),
            }
        )

    return results


# ---------------------------------------------------------------------
# Detect unusual support activity
# ---------------------------------------------------------------------

def detect_unusual_support_patterns():
    """
    Detect simple patterns in support activity that may deserve
    internal review.

    The output deliberately uses 'requires review' language rather
    than claiming that an issue is definitely abnormal.
    """

    data = load_data()
    tickets = data["tickets"].copy()

    if tickets.empty:
        return []

    patterns = []

    # ---------------------------------------------------------
    # Pattern 1: Multiple tickets with the same issue signature
    # ---------------------------------------------------------

    recurring_issues = detect_recurring_issues(
        min_occurrences=2
    )

    for issue in recurring_issues:

        if issue["affected_account_count"] > 1:
            patterns.append(
                {
                    "type": "multi_customer_issue",
                    "priority": "high",
                    "message": (
                        "A similar issue appears across multiple "
                        "customer accounts and requires review."
                    ),
                    "issue_signature": issue["issue_signature"],
                    "ticket_ids": issue["ticket_ids"],
                    "account_ids": issue["account_ids"],
                    "affected_account_count": (
                        issue["affected_account_count"]
                    ),
                }
            )

        else:
            patterns.append(
                {
                    "type": "recurring_customer_issue",
                    "priority": "medium",
                    "message": (
                        "A similar issue appears multiple times and "
                        "may indicate a recurring problem."
                    ),
                    "issue_signature": issue["issue_signature"],
                    "ticket_ids": issue["ticket_ids"],
                    "account_ids": issue["account_ids"],
                    "affected_account_count": (
                        issue["affected_account_count"]
                    ),
                }
            )

    # ---------------------------------------------------------
    # Pattern 2: Multiple open tickets for one account
    # ---------------------------------------------------------

    open_tickets = tickets[
        tickets["status"].astype(str).str.lower() == "open"
    ]

    if not open_tickets.empty:

        account_counts = (
            open_tickets.groupby("account_id")
            .agg(
                open_ticket_count=("ticket_id", "count"),
                ticket_ids=("ticket_id", list),
            )
            .reset_index()
        )

        repeated_accounts = account_counts[
            account_counts["open_ticket_count"] >= 2
        ]

        for _, row in repeated_accounts.iterrows():

            patterns.append(
                {
                    "type": "multiple_open_tickets",
                    "priority": "medium",
                    "message": (
                        "One customer account has multiple open "
                        "support issues and may require attention."
                    ),
                    "account_id": row["account_id"],
                    "open_ticket_count": int(
                        row["open_ticket_count"]
                    ),
                    "ticket_ids": row["ticket_ids"],
                }
            )

    return patterns


# ---------------------------------------------------------------------
# Detect potentially unreliable historical resolutions
# ---------------------------------------------------------------------

def detect_historical_resolution_risks():
    """
    Surface closed tickets whose historical resolutions should not
    automatically be treated as authoritative.

    Historical support answers are useful evidence, but they may be
    outdated or wrong. This function identifies them for review rather
    than treating them as governing policy.
    """

    data = load_data()
    tickets = data["tickets"].copy()

    if tickets.empty:
        return []

    risks = []

    closed_tickets = tickets[
        tickets["status"].astype(str).str.lower() == "closed"
    ]

    for _, ticket in closed_tickets.iterrows():

        historical_resolution = ticket.get(
            "historical_resolution"
        )

        if pd.notna(historical_resolution):

            risks.append(
                {
                    "ticket_id": ticket["ticket_id"],
                    "account_id": ticket["account_id"],
                    "subject": ticket["subject"],
                    "historical_resolution": historical_resolution,
                    "risk": (
                        "Historical support guidance is non-authoritative "
                        "and should be checked against current governing "
                        "policy and any account-specific agreement."
                    ),
                }
            )

    return risks


# ---------------------------------------------------------------------
# Internal proactive issue summary
# ---------------------------------------------------------------------

def get_proactive_issue_summary():
    """
    Build a structured summary for an internal support/operations view.

    This combines:
    - Open support activity
    - Recurring issues
    - Multi-customer patterns
    - Multiple open issues per account
    - Historical resolution risks
    """

    open_ticket_issues = detect_open_ticket_issues()

    recurring_issues = detect_recurring_issues(
        min_occurrences=2
    )

    unusual_patterns = detect_unusual_support_patterns()

    historical_resolution_risks = (
        detect_historical_resolution_risks()
    )

    return {
        "open_ticket_count": len(open_ticket_issues),
        "open_ticket_issues": open_ticket_issues,
        "recurring_issue_count": len(recurring_issues),
        "recurring_issues": recurring_issues,
        "unusual_pattern_count": len(unusual_patterns),
        "unusual_patterns": unusual_patterns,
        "historical_resolution_risk_count": len(
            historical_resolution_risks
        ),
        "historical_resolution_risks": (
            historical_resolution_risks
        ),
    }


# ---------------------------------------------------------------------
# Local testing
# ---------------------------------------------------------------------

def main():

    print("=" * 70)
    print("TESTING STRUCTURED DATA AND PROACTIVE ISSUE TOOLS")
    print("=" * 70)

    print("\n1. ACCOUNT LOOKUP: ACCT-001")
    print(lookup_account("ACCT-001"))

    print("\n2. ORDER LOOKUP: ORD-1001")
    print(lookup_order("ORD-1001"))

    print("\n3. ACCOUNT FOR ORDER: ORD-1001")
    print(get_account_for_order("ORD-1001"))

    print("\n4. TICKET LOOKUP: TKT-502")
    print(lookup_ticket("TKT-502"))

    print("\n5. ACCOUNT FOR TICKET: TKT-502")
    print(get_account_for_ticket("TKT-502"))

    print("\n6. ORDERS FOR ACCT-001")
    print(get_orders_for_account("ACCT-001"))

    print("\n7. TICKETS FOR ACCT-002")
    print(get_tickets_for_account("ACCT-002"))

    print("\n8. UNKNOWN ORDER: ORD-9999")
    print(lookup_order("ORD-9999"))

    print("\n9. CANCELLATION TIMING: ORD-1001")
    print(calculate_cancellation_timing("ORD-1001"))

    print("\n10. CANCELLATION TIMING: ORD-2001")
    print(calculate_cancellation_timing("ORD-2001"))

    print("\n11. CANCELLATION TIMING: ORD-2002")
    print(calculate_cancellation_timing("ORD-2002"))

    print("\n12. DATASET SNAPSHOT TIME")
    print(get_dataset_snapshot_time())

    print("\n13. FAILED PICKUP TIMING: ORD-2002")
    print(calculate_failed_pickup_timing("ORD-2002"))

    print("\n14. FAILED PICKUP TIMING: ORD-1002")
    print(calculate_failed_pickup_timing("ORD-1002"))

    print("\n15. UNKNOWN FAILED PICKUP ORDER: ORD-9999")
    print(calculate_failed_pickup_timing("ORD-9999"))

    print("\n" + "=" * 70)
    print("PROACTIVE ISSUE DETECTION")
    print("=" * 70)

    print("\n16. OPEN TICKET ISSUES")
    print(detect_open_ticket_issues())

    print("\n17. RECURRING ISSUES")
    print(detect_recurring_issues())

    print("\n18. UNUSUAL SUPPORT PATTERNS")
    print(detect_unusual_support_patterns())

    print("\n19. HISTORICAL RESOLUTION RISKS")
    print(detect_historical_resolution_risks())

    print("\n20. PROACTIVE ISSUE SUMMARY")
    print(get_proactive_issue_summary())


if __name__ == "__main__":
    main()