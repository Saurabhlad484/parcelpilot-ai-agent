import pytest

from src.access_control import (
    get_mock_user,
    can_access_account,
    require_account_access,
)


# ---------------------------------------------------------------------
# Test 1: Default user is support agent
# ---------------------------------------------------------------------

def test_get_default_user():
    """
    Calling get_mock_user() without a username should return
    the default support agent.
    """

    user = get_mock_user()

    assert user["username"] == "support_agent"
    assert user["role"] == "support_agent"
    assert user["allowed_accounts"] == "all"


# ---------------------------------------------------------------------
# Test 2: Get customer for ACCT-001
# ---------------------------------------------------------------------

def test_get_customer_acct_001():
    """
    The customer_acct_001 user should only belong to ACCT-001.
    """

    user = get_mock_user("customer_acct_001")

    assert user["username"] == "customer_acct_001"
    assert user["role"] == "customer"
    assert user["allowed_accounts"] == {"ACCT-001"}


# ---------------------------------------------------------------------
# Test 3: Get customer for ACCT-002
# ---------------------------------------------------------------------

def test_get_customer_acct_002():
    """
    The customer_acct_002 user should only belong to ACCT-002.
    """

    user = get_mock_user("customer_acct_002")

    assert user["username"] == "customer_acct_002"
    assert user["role"] == "customer"
    assert user["allowed_accounts"] == {"ACCT-002"}


# ---------------------------------------------------------------------
# Test 4: Unknown user raises PermissionError
# ---------------------------------------------------------------------

def test_unknown_user_raises_permission_error():
    """
    An unknown user must not be authenticated.
    """

    with pytest.raises(PermissionError) as exc_info:
        get_mock_user("unknown_user")

    assert "Unknown or unauthorised user" in str(exc_info.value)


# ---------------------------------------------------------------------
# Test 5: Support agent can access ACCT-001
# ---------------------------------------------------------------------

def test_support_agent_can_access_acct_001():
    """
    Support agents have access to all accounts.
    """

    user = get_mock_user("support_agent")

    assert can_access_account(user, "ACCT-001") is True


# ---------------------------------------------------------------------
# Test 6: Support agent can access ACCT-002
# ---------------------------------------------------------------------

def test_support_agent_can_access_acct_002():
    """
    Support agents should also be able to access ACCT-002.
    """

    user = get_mock_user("support_agent")

    assert can_access_account(user, "ACCT-002") is True


# ---------------------------------------------------------------------
# Test 7: Customer ACCT-001 can access own account
# ---------------------------------------------------------------------

def test_customer_001_can_access_own_account():
    """
    customer_acct_001 should be authorised for ACCT-001.
    """

    user = get_mock_user("customer_acct_001")

    assert can_access_account(user, "ACCT-001") is True


# ---------------------------------------------------------------------
# Test 8: Customer ACCT-001 cannot access ACCT-002
# ---------------------------------------------------------------------

def test_customer_001_cannot_access_other_account():
    """
    customer_acct_001 must not access ACCT-002.
    """

    user = get_mock_user("customer_acct_001")

    assert can_access_account(user, "ACCT-002") is False


# ---------------------------------------------------------------------
# Test 9: Customer ACCT-002 can access own account
# ---------------------------------------------------------------------

def test_customer_002_can_access_own_account():
    """
    customer_acct_002 should be authorised for ACCT-002.
    """

    user = get_mock_user("customer_acct_002")

    assert can_access_account(user, "ACCT-002") is True


# ---------------------------------------------------------------------
# Test 10: Customer ACCT-002 cannot access ACCT-001
# ---------------------------------------------------------------------

def test_customer_002_cannot_access_other_account():
    """
    customer_acct_002 must not access ACCT-001.
    """

    user = get_mock_user("customer_acct_002")

    assert can_access_account(user, "ACCT-001") is False


# ---------------------------------------------------------------------
# Test 11: Account access is case-insensitive
# ---------------------------------------------------------------------

def test_account_access_is_case_insensitive():
    """
    Account IDs should work regardless of letter case.
    """

    user = get_mock_user("customer_acct_001")

    assert can_access_account(user, "acct-001") is True
    assert can_access_account(user, "AcCt-001") is True


# ---------------------------------------------------------------------
# Test 12: None account ID is allowed
# ---------------------------------------------------------------------

def test_none_account_id_is_allowed():
    """
    No account-specific resource is being accessed when account_id
    is None, so the function should return True.
    """

    user = get_mock_user("customer_acct_001")

    assert can_access_account(user, None) is True


# ---------------------------------------------------------------------
# Test 13: require_account_access allows authorised access
# ---------------------------------------------------------------------

def test_require_account_access_allows_authorised_user():
    """
    require_account_access should not raise an exception when
    the user is authorised.
    """

    user = get_mock_user("customer_acct_001")

    require_account_access(user, "ACCT-001")


# ---------------------------------------------------------------------
# Test 14: require_account_access allows support agent
# ---------------------------------------------------------------------

def test_require_account_access_allows_support_agent():
    """
    Support agents should pass account access enforcement for
    any account.
    """

    user = get_mock_user("support_agent")

    require_account_access(user, "ACCT-001")
    require_account_access(user, "ACCT-002")


# ---------------------------------------------------------------------
# Test 15: require_account_access denies unauthorised access
# ---------------------------------------------------------------------

def test_require_account_access_denies_unauthorised_user():
    """
    An unauthorised customer should receive a PermissionError.
    """

    user = get_mock_user("customer_acct_001")

    with pytest.raises(PermissionError) as exc_info:
        require_account_access(user, "ACCT-002")

    error_message = str(exc_info.value)

    assert "Access denied" in error_message
    assert "ACCT-002" in error_message


# ---------------------------------------------------------------------
# Test 16: require_account_access is case-insensitive
# ---------------------------------------------------------------------

def test_require_account_access_is_case_insensitive():
    """
    Access enforcement should also handle lowercase account IDs.
    """

    user = get_mock_user("customer_acct_001")

    require_account_access(user, "acct-001")


# ---------------------------------------------------------------------
# Test 17: require_account_access allows None
# ---------------------------------------------------------------------

def test_require_account_access_allows_none_account():
    """
    None should not trigger an account-level access denial.
    """

    user = get_mock_user("customer_acct_001")

    require_account_access(user, None)