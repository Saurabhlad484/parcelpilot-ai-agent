import pandas as pd
import pytest

from src.data_loader import (
    WORKBOOK_PATH,
    EXPECTED_SHEETS,
    EXPECTED_COLUMNS,
    load_data,
    validate_columns,
    convert_datetime_columns,
    validate_ids,
)


# ---------------------------------------------------------------------
# Test 1: Workbook exists
# ---------------------------------------------------------------------

def test_workbook_exists():
    assert WORKBOOK_PATH.exists()


# ---------------------------------------------------------------------
# Test 2: Load data successfully
# ---------------------------------------------------------------------

def test_load_data_returns_expected_structure():
    data = load_data()

    assert isinstance(data, dict)

    assert "readme" in data
    assert "accounts" in data
    assert "orders" in data
    assert "tickets" in data


# ---------------------------------------------------------------------
# Test 3: Loaded objects are DataFrames
# ---------------------------------------------------------------------

def test_loaded_data_are_dataframes():
    data = load_data()

    assert isinstance(data["readme"], pd.DataFrame)
    assert isinstance(data["accounts"], pd.DataFrame)
    assert isinstance(data["orders"], pd.DataFrame)
    assert isinstance(data["tickets"], pd.DataFrame)


# ---------------------------------------------------------------------
# Test 4: Expected sheets are defined
# ---------------------------------------------------------------------

def test_expected_sheets():
    assert EXPECTED_SHEETS == {
        "README",
        "accounts",
        "orders",
        "tickets",
    }


# ---------------------------------------------------------------------
# Test 5: All expected account columns exist
# ---------------------------------------------------------------------

def test_account_columns():
    data = load_data()

    validate_columns(
        "accounts",
        data["accounts"],
    )

    assert EXPECTED_COLUMNS["accounts"].issubset(
        set(data["accounts"].columns)
    )


# ---------------------------------------------------------------------
# Test 6: All expected order columns exist
# ---------------------------------------------------------------------

def test_order_columns():
    data = load_data()

    validate_columns(
        "orders",
        data["orders"],
    )

    assert EXPECTED_COLUMNS["orders"].issubset(
        set(data["orders"].columns)
    )


# ---------------------------------------------------------------------
# Test 7: All expected ticket columns exist
# ---------------------------------------------------------------------

def test_ticket_columns():
    data = load_data()

    validate_columns(
        "tickets",
        data["tickets"],
    )

    assert EXPECTED_COLUMNS["tickets"].issubset(
        set(data["tickets"].columns)
    )


# ---------------------------------------------------------------------
# Test 8: Missing column is detected
# ---------------------------------------------------------------------

def test_validate_columns_detects_missing_column():
    dataframe = pd.DataFrame(
        {
            "account_id": ["ACCT-001"],
        }
    )

    with pytest.raises(ValueError, match="missing required column"):
        validate_columns(
            "accounts",
            dataframe,
        )


# ---------------------------------------------------------------------
# Test 9: Datetime columns are converted
# ---------------------------------------------------------------------

def test_convert_datetime_columns():
    dataframe = pd.DataFrame(
        {
            "created_at": [
                "2026-01-01 10:00:00",
                "2026-01-02 11:30:00",
            ]
        }
    )

    result = convert_datetime_columns(
        dataframe,
        ["created_at"],
    )

    assert pd.api.types.is_datetime64_any_dtype(
        result["created_at"]
    )


# ---------------------------------------------------------------------
# Test 10: Invalid datetime becomes NaT
# ---------------------------------------------------------------------

def test_invalid_datetime_becomes_nat():
    dataframe = pd.DataFrame(
        {
            "created_at": [
                "2026-01-01 10:00:00",
                "not-a-date",
            ]
        }
    )

    result = convert_datetime_columns(
        dataframe,
        ["created_at"],
    )

    assert pd.notna(result["created_at"].iloc[0])
    assert pd.isna(result["created_at"].iloc[1])


# ---------------------------------------------------------------------
# Test 11: Original DataFrame is not modified
# ---------------------------------------------------------------------

def test_datetime_conversion_does_not_modify_original():
    dataframe = pd.DataFrame(
        {
            "created_at": [
                "2026-01-01 10:00:00",
            ]
        }
    )

    original_dtype = dataframe["created_at"].dtype

    result = convert_datetime_columns(
        dataframe,
        ["created_at"],
    )

    assert dataframe["created_at"].dtype == original_dtype
    assert pd.api.types.is_datetime64_any_dtype(
        result["created_at"]
    )


# ---------------------------------------------------------------------
# Test 12: Valid IDs and relationships pass
# ---------------------------------------------------------------------

def test_validate_ids_valid_data():
    accounts = pd.DataFrame(
        {
            "account_id": [
                "ACCT-001",
                "ACCT-002",
            ]
        }
    )

    orders = pd.DataFrame(
        {
            "order_id": [
                "ORD-1001",
                "ORD-2001",
            ],
            "account_id": [
                "ACCT-001",
                "ACCT-002",
            ]
        }
    )

    tickets = pd.DataFrame(
        {
            "ticket_id": [
                "TKT-501",
                "TKT-502",
            ],
            "account_id": [
                "ACCT-001",
                "ACCT-002",
            ]
        }
    )

    # Should not raise an exception.
    validate_ids(
        accounts,
        orders,
        tickets,
    )


# ---------------------------------------------------------------------
# Test 13: Duplicate account IDs are rejected
# ---------------------------------------------------------------------

def test_validate_ids_duplicate_account_id():
    accounts = pd.DataFrame(
        {
            "account_id": [
                "ACCT-001",
                "ACCT-001",
            ]
        }
    )

    orders = pd.DataFrame(
        {
            "order_id": ["ORD-1001"],
            "account_id": ["ACCT-001"],
        }
    )

    tickets = pd.DataFrame(
        {
            "ticket_id": ["TKT-501"],
            "account_id": ["ACCT-001"],
        }
    )

    with pytest.raises(
        ValueError,
        match="Duplicate account_id",
    ):
        validate_ids(
            accounts,
            orders,
            tickets,
        )


# ---------------------------------------------------------------------
# Test 14: Duplicate order IDs are rejected
# ---------------------------------------------------------------------

def test_validate_ids_duplicate_order_id():
    accounts = pd.DataFrame(
        {
            "account_id": ["ACCT-001"],
        }
    )

    orders = pd.DataFrame(
        {
            "order_id": [
                "ORD-1001",
                "ORD-1001",
            ],
            "account_id": [
                "ACCT-001",
                "ACCT-001",
            ],
        }
    )

    tickets = pd.DataFrame(
        {
            "ticket_id": ["TKT-501"],
            "account_id": ["ACCT-001"],
        }
    )

    with pytest.raises(
        ValueError,
        match="Duplicate order_id",
    ):
        validate_ids(
            accounts,
            orders,
            tickets,
        )


# ---------------------------------------------------------------------
# Test 15: Duplicate ticket IDs are rejected
# ---------------------------------------------------------------------

def test_validate_ids_duplicate_ticket_id():
    accounts = pd.DataFrame(
        {
            "account_id": ["ACCT-001"],
        }
    )

    orders = pd.DataFrame(
        {
            "order_id": ["ORD-1001"],
            "account_id": ["ACCT-001"],
        }
    )

    tickets = pd.DataFrame(
        {
            "ticket_id": [
                "TKT-501",
                "TKT-501",
            ],
            "account_id": [
                "ACCT-001",
                "ACCT-001",
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="Duplicate ticket_id",
    ):
        validate_ids(
            accounts,
            orders,
            tickets,
        )


# ---------------------------------------------------------------------
# Test 16: Unknown order account is rejected
# ---------------------------------------------------------------------

def test_validate_ids_unknown_order_account():
    accounts = pd.DataFrame(
        {
            "account_id": ["ACCT-001"],
        }
    )

    orders = pd.DataFrame(
        {
            "order_id": ["ORD-1001"],
            "account_id": ["ACCT-999"],
        }
    )

    tickets = pd.DataFrame(
        {
            "ticket_id": ["TKT-501"],
            "account_id": ["ACCT-001"],
        }
    )

    with pytest.raises(
        ValueError,
        match="Orders contain unknown account_id",
    ):
        validate_ids(
            accounts,
            orders,
            tickets,
        )


# ---------------------------------------------------------------------
# Test 17: Unknown ticket account is rejected
# ---------------------------------------------------------------------

def test_validate_ids_unknown_ticket_account():
    accounts = pd.DataFrame(
        {
            "account_id": ["ACCT-001"],
        }
    )

    orders = pd.DataFrame(
        {
            "order_id": ["ORD-1001"],
            "account_id": ["ACCT-001"],
        }
    )

    tickets = pd.DataFrame(
        {
            "ticket_id": ["TKT-501"],
            "account_id": ["ACCT-999"],
        }
    )

    with pytest.raises(
        ValueError,
        match="Tickets contain unknown account_id",
    ):
        validate_ids(
            accounts,
            orders,
            tickets,
        )


# ---------------------------------------------------------------------
# Test 18: Loaded order datetime columns are converted
# ---------------------------------------------------------------------

def test_loaded_order_datetime_columns():
    data = load_data()

    orders = data["orders"]

    datetime_columns = [
        "booked_at",
        "pickup_window_start",
        "pickup_window_end",
        "pickup_actual_at",
        "cancellation_requested_at",
    ]

    for column in datetime_columns:
        assert pd.api.types.is_datetime64_any_dtype(
            orders[column]
        )


# ---------------------------------------------------------------------
# Test 19: Loaded ticket datetime columns are converted
# ---------------------------------------------------------------------

def test_loaded_ticket_datetime_columns():
    data = load_data()

    tickets = data["tickets"]

    datetime_columns = [
        "created_at",
        "last_customer_message_at",
    ]

    for column in datetime_columns:
        assert pd.api.types.is_datetime64_any_dtype(
            tickets[column]
        )


# ---------------------------------------------------------------------
# Test 20: Account IDs are unique in real dataset
# ---------------------------------------------------------------------

def test_real_account_ids_are_unique():
    data = load_data()

    accounts = data["accounts"]

    assert not accounts["account_id"].duplicated().any()


# ---------------------------------------------------------------------
# Test 21: Order IDs are unique in real dataset
# ---------------------------------------------------------------------

def test_real_order_ids_are_unique():
    data = load_data()

    orders = data["orders"]

    assert not orders["order_id"].duplicated().any()


# ---------------------------------------------------------------------
# Test 22: Ticket IDs are unique in real dataset
# ---------------------------------------------------------------------

def test_real_ticket_ids_are_unique():
    data = load_data()

    tickets = data["tickets"]

    assert not tickets["ticket_id"].duplicated().any()


# ---------------------------------------------------------------------
# Test 23: Real order account relationships are valid
# ---------------------------------------------------------------------

def test_real_order_account_relationships():
    data = load_data()

    account_ids = set(
        data["accounts"]["account_id"]
    )

    order_account_ids = set(
        data["orders"]["account_id"]
    )

    assert order_account_ids.issubset(account_ids)


# ---------------------------------------------------------------------
# Test 24: Real ticket account relationships are valid
# ---------------------------------------------------------------------

def test_real_ticket_account_relationships():
    data = load_data()

    account_ids = set(
        data["accounts"]["account_id"]
    )

    ticket_account_ids = set(
        data["tickets"]["account_id"]
    )

    assert ticket_account_ids.issubset(account_ids)


# ---------------------------------------------------------------------
# Test 25: Loaded datasets are not empty
# ---------------------------------------------------------------------

def test_loaded_datasets_are_not_empty():
    data = load_data()

    assert not data["accounts"].empty
    assert not data["orders"].empty
    assert not data["tickets"].empty


# ---------------------------------------------------------------------
# Test 26: Required IDs do not contain null values
# ---------------------------------------------------------------------

def test_required_ids_are_not_null():
    data = load_data()

    assert data["accounts"]["account_id"].notna().all()
    assert data["orders"]["order_id"].notna().all()
    assert data["orders"]["account_id"].notna().all()
    assert data["tickets"]["ticket_id"].notna().all()
    assert data["tickets"]["account_id"].notna().all()