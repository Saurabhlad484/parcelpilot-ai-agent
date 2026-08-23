from pathlib import Path
import pandas as pd


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
WORKBOOK_PATH = DOCUMENTS_DIR / "ParcelPilot_Assessment_Data.xlsx"


# ---------------------------------------------------------
# Expected workbook structure
# ---------------------------------------------------------

EXPECTED_SHEETS = {"README", "accounts", "orders", "tickets"}

EXPECTED_COLUMNS = {
    "accounts": {
        "account_id",
        "account_name",
        "plan",
        "status",
        "csm",
        "contract_file",
        "premium_support",
        "notes",
    },
    "orders": {
        "order_id",
        "account_id",
        "carrier",
        "status",
        "booked_at",
        "pickup_window_start",
        "pickup_window_end",
        "pickup_actual_at",
        "shipment_fee_inr",
        "carrier_fault",
        "customer_fault",
        "cancellation_requested_at",
        "notes",
    },
    "tickets": {
        "ticket_id",
        "account_id",
        "created_at",
        "status",
        "subject",
        "description",
        "channel",
        "assigned_to",
        "last_customer_message_at",
        "historical_resolution",
    },
}


def load_data():
    """
    Load the ParcelPilot structured data from the Excel workbook.

    Returns:
        dict containing:
        - readme
        - accounts
        - orders
        - tickets
    """

    if not WORKBOOK_PATH.exists():
        raise FileNotFoundError(
            f"Workbook not found: {WORKBOOK_PATH}"
        )

    excel_file = pd.ExcelFile(WORKBOOK_PATH)

    actual_sheets = set(excel_file.sheet_names)

    missing_sheets = EXPECTED_SHEETS - actual_sheets

    if missing_sheets:
        raise ValueError(
            f"Missing expected sheet(s): {sorted(missing_sheets)}"
        )

    # Load sheets without modifying the Excel workbook
    readme = pd.read_excel(WORKBOOK_PATH, sheet_name="README")
    accounts = pd.read_excel(WORKBOOK_PATH, sheet_name="accounts")
    orders = pd.read_excel(WORKBOOK_PATH, sheet_name="orders")
    tickets = pd.read_excel(WORKBOOK_PATH, sheet_name="tickets")

    # Validate expected columns
    validate_columns("accounts", accounts)
    validate_columns("orders", orders)
    validate_columns("tickets", tickets)

    # Convert timestamp columns
    orders = convert_datetime_columns(
        orders,
        [
            "booked_at",
            "pickup_window_start",
            "pickup_window_end",
            "pickup_actual_at",
            "cancellation_requested_at",
        ],
    )

    tickets = convert_datetime_columns(
        tickets,
        [
            "created_at",
            "last_customer_message_at",
        ],
    )

    # Validate IDs and relationships
    validate_ids(accounts, orders, tickets)

    return {
        "readme": readme,
        "accounts": accounts,
        "orders": orders,
        "tickets": tickets,
    }


def validate_columns(sheet_name, dataframe):
    """
    Check that a sheet contains all required columns.
    """

    expected = EXPECTED_COLUMNS[sheet_name]
    actual = set(dataframe.columns)

    missing_columns = expected - actual

    if missing_columns:
        raise ValueError(
            f"{sheet_name} is missing required column(s): "
            f"{sorted(missing_columns)}"
        )


def convert_datetime_columns(dataframe, columns):
    """
    Convert specified columns to datetime.

    Missing values are allowed and become NaT.
    The original Excel workbook is not changed.
    """

    dataframe = dataframe.copy()

    for column in columns:
        dataframe[column] = pd.to_datetime(
            dataframe[column],
            errors="coerce",
        )

    return dataframe


def validate_ids(accounts, orders, tickets):
    """
    Validate key IDs and relationships.

    Checks:
    - account_id values are unique in accounts
    - order_id values are unique in orders
    - ticket_id values are unique in tickets
    - every order account_id exists in accounts
    - every ticket account_id exists in accounts
    """

    if accounts["account_id"].duplicated().any():
        raise ValueError("Duplicate account_id found in accounts.")

    if orders["order_id"].duplicated().any():
        raise ValueError("Duplicate order_id found in orders.")

    if tickets["ticket_id"].duplicated().any():
        raise ValueError("Duplicate ticket_id found in tickets.")

    account_ids = set(accounts["account_id"])

    invalid_order_accounts = set(
        orders["account_id"]
    ) - account_ids

    if invalid_order_accounts:
        raise ValueError(
            "Orders contain unknown account_id values: "
            f"{sorted(invalid_order_accounts)}"
        )

    invalid_ticket_accounts = set(
        tickets["account_id"]
    ) - account_ids

    if invalid_ticket_accounts:
        raise ValueError(
            "Tickets contain unknown account_id values: "
            f"{sorted(invalid_ticket_accounts)}"
        )


def print_summary(data):
    """
    Print a simple summary for testing.
    """

    readme = data["readme"]
    accounts = data["accounts"]
    orders = data["orders"]
    tickets = data["tickets"]

    print("=" * 70)
    print("PARCELPILOT STRUCTURED DATA LOADED SUCCESSFULLY")
    print("=" * 70)

    print(f"\nWorkbook: {WORKBOOK_PATH.name}")
    print(f"Accounts: {len(accounts)}")
    print(f"Orders: {len(orders)}")
    print(f"Tickets: {len(tickets)}")

    print("\nDATA TYPES AFTER DATETIME CONVERSION")

    print("\nOrders:")
    print(
        orders[
            [
                "booked_at",
                "pickup_window_start",
                "pickup_window_end",
                "pickup_actual_at",
                "cancellation_requested_at",
            ]
        ].dtypes
    )

    print("\nTickets:")
    print(
        tickets[
            [
                "created_at",
                "last_customer_message_at",
            ]
        ].dtypes
    )

    print("\nKEY RELATIONSHIP CHECK")
    print("All order account_ids exist in accounts: PASS")
    print("All ticket account_ids exist in accounts: PASS")

    print("\nREADME DATA")
    print(readme.to_string(index=False))


def main():
    data = load_data()
    print_summary(data)


if __name__ == "__main__":
    main()