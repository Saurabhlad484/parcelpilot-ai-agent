"""
Mock access control for the ParcelPilot AI agent.

The project uses a simple role-based access control (RBAC) model.
Access checks happen before structured-data or document tools are used.
"""

from __future__ import annotations


# Mock authenticated users for the project demo.
MOCK_USERS = {
    "support_agent": {
        "username": "support_agent",
        "role": "support_agent",
        "allowed_accounts": "all",
    },
    "customer_acct_001": {
        "username": "customer_acct_001",
        "role": "customer",
        "allowed_accounts": {"ACCT-001"},
    },
    "customer_acct_002": {
        "username": "customer_acct_002",
        "role": "customer",
        "allowed_accounts": {"ACCT-002"},
    },
}


def get_mock_user(username: str = "support_agent") -> dict:
    """
    Return the mock authenticated user.

    In a production system, this would come from a real authentication
    system. For this project, authentication is mocked locally.
    """

    user = MOCK_USERS.get(username)

    if user is None:
        raise PermissionError(
            f"Unknown or unauthorised user: {username}"
        )

    return user


def can_access_account(
    user: dict,
    account_id: str | None,
) -> bool:
    """
    Check whether the authenticated user can access an account.

    Support agents can access all accounts.
    Customers can only access their own account.
    """

    if account_id is None:
        return True

    if user["allowed_accounts"] == "all":
        return True

    return account_id.upper() in user["allowed_accounts"]


def require_account_access(
    user: dict,
    account_id: str | None,
) -> None:
    """
    Enforce account-level access control.

    Raises PermissionError when access is not authorised.
    """

    if not can_access_account(user, account_id):
        raise PermissionError(
            f"Access denied: you are not authorised to access "
            f"account `{account_id.upper()}`."
        )