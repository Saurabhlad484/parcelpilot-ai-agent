from src.tools import (
    lookup_account,
    lookup_order,
    lookup_ticket,
    get_account_for_order,
    get_account_for_ticket,
    get_orders_for_account,
    get_tickets_for_account,
    get_dataset_snapshot_time,
    calculate_failed_pickup_timing,
    calculate_cancellation_timing,
    normalize_issue_text,
    get_issue_signature,
    detect_recurring_issues,
    detect_open_ticket_issues,
    detect_unusual_support_patterns,
    detect_historical_resolution_risks,
    get_proactive_issue_summary,
)


# ---------------------------------------------------------------------
# Test 1: Valid account lookup
# ---------------------------------------------------------------------

def test_lookup_valid_account():
    account = lookup_account("ACCT-001")

    assert account is not None
    assert account["account_id"] == "ACCT-001"


# ---------------------------------------------------------------------
# Test 2: Account lookup is case-insensitive
# ---------------------------------------------------------------------

def test_lookup_account_case_insensitive():
    account = lookup_account("acct-001")

    assert account is not None
    assert account["account_id"] == "ACCT-001"


# ---------------------------------------------------------------------
# Test 3: Invalid account lookup
# ---------------------------------------------------------------------

def test_lookup_invalid_account():
    account = lookup_account("ACCT-999")

    assert account is None


# ---------------------------------------------------------------------
# Test 4: Valid order lookup
# ---------------------------------------------------------------------

def test_lookup_valid_order():
    order = lookup_order("ORD-1001")

    assert order is not None
    assert order["order_id"] == "ORD-1001"


# ---------------------------------------------------------------------
# Test 5: Order lookup is case-insensitive
# ---------------------------------------------------------------------

def test_lookup_order_case_insensitive():
    order = lookup_order("ord-1001")

    assert order is not None
    assert order["order_id"] == "ORD-1001"


# ---------------------------------------------------------------------
# Test 6: Invalid order lookup
# ---------------------------------------------------------------------

def test_lookup_invalid_order():
    order = lookup_order("ORD-9999")

    assert order is None


# ---------------------------------------------------------------------
# Test 7: Valid ticket lookup
# ---------------------------------------------------------------------

def test_lookup_valid_ticket():
    ticket = lookup_ticket("TKT-501")

    assert ticket is not None
    assert ticket["ticket_id"] == "TKT-501"


# ---------------------------------------------------------------------
# Test 8: Ticket lookup is case-insensitive
# ---------------------------------------------------------------------

def test_lookup_ticket_case_insensitive():
    ticket = lookup_ticket("tkt-501")

    assert ticket is not None
    assert ticket["ticket_id"] == "TKT-501"


# ---------------------------------------------------------------------
# Test 9: Invalid ticket lookup
# ---------------------------------------------------------------------

def test_lookup_invalid_ticket():
    ticket = lookup_ticket("TKT-999")

    assert ticket is None


# ---------------------------------------------------------------------
# Test 10: Get account for valid order
# ---------------------------------------------------------------------

def test_get_account_for_order():
    account = get_account_for_order("ORD-1001")

    assert account is not None
    assert account["account_id"] == "ACCT-001"


# ---------------------------------------------------------------------
# Test 11: Get account for invalid order
# ---------------------------------------------------------------------

def test_get_account_for_invalid_order():
    account = get_account_for_order("ORD-9999")

    assert account is None


# ---------------------------------------------------------------------
# Test 12: Get account for valid ticket
# ---------------------------------------------------------------------

def test_get_account_for_ticket():
    account = get_account_for_ticket("TKT-501")

    assert account is not None
    assert "account_id" in account


# ---------------------------------------------------------------------
# Test 13: Get account for invalid ticket
# ---------------------------------------------------------------------

def test_get_account_for_invalid_ticket():
    account = get_account_for_ticket("TKT-999")

    assert account is None


# ---------------------------------------------------------------------
# Test 14: Get orders for account
# ---------------------------------------------------------------------

def test_get_orders_for_account():
    orders = get_orders_for_account("ACCT-001")

    assert isinstance(orders, list)

    for order in orders:
        assert order["account_id"] == "ACCT-001"


# ---------------------------------------------------------------------
# Test 15: Get orders for unknown account
# ---------------------------------------------------------------------

def test_get_orders_for_unknown_account():
    orders = get_orders_for_account("ACCT-999")

    assert isinstance(orders, list)
    assert len(orders) == 0


# ---------------------------------------------------------------------
# Test 16: Get tickets for account
# ---------------------------------------------------------------------

def test_get_tickets_for_account():
    tickets = get_tickets_for_account("ACCT-001")

    assert isinstance(tickets, list)

    for ticket in tickets:
        assert ticket["account_id"] == "ACCT-001"


# ---------------------------------------------------------------------
# Test 17: Get tickets for unknown account
# ---------------------------------------------------------------------

def test_get_tickets_for_unknown_account():
    tickets = get_tickets_for_account("ACCT-999")

    assert isinstance(tickets, list)
    assert len(tickets) == 0


# ---------------------------------------------------------------------
# Test 18: Dataset snapshot time
# ---------------------------------------------------------------------

def test_get_dataset_snapshot_time():
    snapshot_time = get_dataset_snapshot_time()

    assert snapshot_time is not None


# ---------------------------------------------------------------------
# Test 19: Valid cancellation timing calculation
# ---------------------------------------------------------------------

def test_calculate_cancellation_timing():
    result = calculate_cancellation_timing("ORD-1001")

    assert result is not None
    assert result["order_id"] == "ORD-1001"
    assert "minutes_after_booking" in result


# ---------------------------------------------------------------------
# Test 20: Invalid cancellation timing order
# ---------------------------------------------------------------------

def test_calculate_cancellation_timing_invalid_order():
    result = calculate_cancellation_timing("ORD-9999")

    assert result is None


# ---------------------------------------------------------------------
# Test 21: Cancellation timing returns account information
# ---------------------------------------------------------------------

def test_cancellation_timing_contains_account():
    result = calculate_cancellation_timing("ORD-1001")

    assert result is not None
    assert "account_id" in result
    assert "status" in result


# ---------------------------------------------------------------------
# Test 22: Valid failed pickup timing calculation
# ---------------------------------------------------------------------

def test_calculate_failed_pickup_timing():
    result = calculate_failed_pickup_timing("ORD-2002")

    assert result is not None
    assert result["order_id"] == "ORD-2002"


# ---------------------------------------------------------------------
# Test 23: Failed pickup timing invalid order
# ---------------------------------------------------------------------

def test_calculate_failed_pickup_timing_invalid_order():
    result = calculate_failed_pickup_timing("ORD-9999")

    assert result is None


# ---------------------------------------------------------------------
# Test 24: Failed pickup timing contains structured facts
# ---------------------------------------------------------------------

def test_failed_pickup_timing_contains_expected_fields():
    result = calculate_failed_pickup_timing("ORD-2002")

    assert result is not None

    # Depending on available pickup data, the function may return
    # a message-only response. If timing was calculated, verify it.
    if "minutes_after_window_end" in result:
        assert "hours_after_window_end" in result
        assert "reference_time" in result
        assert "carrier_fault" in result
        assert "customer_fault" in result


# ---------------------------------------------------------------------
# Test 25: Normalize issue text
# ---------------------------------------------------------------------

def test_normalize_issue_text():
    result = normalize_issue_text(
        "The Pickup Was Delayed After Booking!"
    )

    assert isinstance(result, str)
    assert result == result.lower()
    assert "!" not in result


# ---------------------------------------------------------------------
# Test 26: Normalize missing issue text
# ---------------------------------------------------------------------

def test_normalize_missing_issue_text():
    result = normalize_issue_text(None)

    assert result == ""


# ---------------------------------------------------------------------
# Test 27: Issue signature is based on subject
# ---------------------------------------------------------------------

def test_get_issue_signature():
    ticket = {
        "subject": "Pickup Delay After Booking"
    }

    signature = get_issue_signature(ticket)

    assert isinstance(signature, str)
    assert signature != ""


# ---------------------------------------------------------------------
# Test 28: Recurring issue detection returns list
# ---------------------------------------------------------------------

def test_detect_recurring_issues():
    results = detect_recurring_issues()

    assert isinstance(results, list)


# ---------------------------------------------------------------------
# Test 29: Recurring issues respect minimum occurrence threshold
# ---------------------------------------------------------------------

def test_recurring_issues_respect_minimum_occurrences():
    results = detect_recurring_issues(
        min_occurrences=2
    )

    for issue in results:
        assert issue["ticket_count"] >= 2


# ---------------------------------------------------------------------
# Test 30: Recurring issue results contain expected fields
# ---------------------------------------------------------------------

def test_recurring_issue_structure():
    results = detect_recurring_issues()

    for issue in results:
        assert "issue_signature" in issue
        assert "ticket_count" in issue
        assert "ticket_ids" in issue
        assert "account_ids" in issue
        assert "affected_account_count" in issue


# ---------------------------------------------------------------------
# Test 31: Open ticket detection returns list
# ---------------------------------------------------------------------

def test_detect_open_ticket_issues():
    results = detect_open_ticket_issues()

    assert isinstance(results, list)


# ---------------------------------------------------------------------
# Test 32: Open ticket results contain expected fields
# ---------------------------------------------------------------------

def test_open_ticket_issue_structure():
    results = detect_open_ticket_issues()

    for ticket in results:
        assert "ticket_id" in ticket
        assert "account_id" in ticket
        assert "subject" in ticket


# ---------------------------------------------------------------------
# Test 33: Unusual support patterns return list
# ---------------------------------------------------------------------

def test_detect_unusual_support_patterns():
    results = detect_unusual_support_patterns()

    assert isinstance(results, list)


# ---------------------------------------------------------------------
# Test 34: Unusual patterns have required structure
# ---------------------------------------------------------------------

def test_unusual_support_pattern_structure():
    results = detect_unusual_support_patterns()

    for pattern in results:
        assert "type" in pattern
        assert "priority" in pattern
        assert "message" in pattern


# ---------------------------------------------------------------------
# Test 35: Historical resolution risks return list
# ---------------------------------------------------------------------

def test_detect_historical_resolution_risks():
    results = detect_historical_resolution_risks()

    assert isinstance(results, list)


# ---------------------------------------------------------------------
# Test 36: Historical resolution risks contain expected fields
# ---------------------------------------------------------------------

def test_historical_resolution_risk_structure():
    results = detect_historical_resolution_risks()

    for risk in results:
        assert "ticket_id" in risk
        assert "account_id" in risk
        assert "historical_resolution" in risk
        assert "risk" in risk


# ---------------------------------------------------------------------
# Test 37: Proactive issue summary
# ---------------------------------------------------------------------

def test_get_proactive_issue_summary():
    summary = get_proactive_issue_summary()

    assert isinstance(summary, dict)


# ---------------------------------------------------------------------
# Test 38: Proactive summary contains all expected sections
# ---------------------------------------------------------------------

def test_proactive_issue_summary_structure():
    summary = get_proactive_issue_summary()

    assert "open_ticket_count" in summary
    assert "open_ticket_issues" in summary

    assert "recurring_issue_count" in summary
    assert "recurring_issues" in summary

    assert "unusual_pattern_count" in summary
    assert "unusual_patterns" in summary

    assert "historical_resolution_risk_count" in summary
    assert "historical_resolution_risks" in summary


# ---------------------------------------------------------------------
# Test 39: Summary counts match result lengths
# ---------------------------------------------------------------------

def test_proactive_issue_summary_counts_match():
    summary = get_proactive_issue_summary()

    assert (
        summary["open_ticket_count"]
        == len(summary["open_ticket_issues"])
    )

    assert (
        summary["recurring_issue_count"]
        == len(summary["recurring_issues"])
    )

    assert (
        summary["unusual_pattern_count"]
        == len(summary["unusual_patterns"])
    )

    assert (
        summary["historical_resolution_risk_count"]
        == len(summary["historical_resolution_risks"])
    )