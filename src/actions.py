"""
Mock state-changing actions for the ParcelPilot AI agent.

This module simulates actions that would normally call an external
support system. For this project, escalations are stored locally.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from src.data_loader import PROJECT_ROOT


ESCALATIONS_FILE = PROJECT_ROOT / "data" / "escalations.csv"


def _ensure_escalations_file() -> None:
    """Create the local escalation store if it does not exist."""

    ESCALATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)

    if not ESCALATIONS_FILE.exists():
        df = pd.DataFrame(
            columns=[
                "escalation_id",
                "ticket_id",
                "account_id",
                "reason",
                "status",
                "created_at",
            ]
        )

        df.to_csv(ESCALATIONS_FILE, index=False)


def _generate_escalation_id(df: pd.DataFrame) -> str:
    """Generate the next escalation ID."""

    if df.empty:
        next_number = 1
    else:
        numbers = []

        for escalation_id in df["escalation_id"].dropna():
            try:
                number = int(str(escalation_id).split("-")[1])
                numbers.append(number)
            except (IndexError, ValueError):
                continue

        next_number = max(numbers, default=0) + 1

    return f"ESC-{next_number:03d}"


def create_escalation(
    ticket_id: str,
    account_id: str,
    reason: str = "Escalation requested through ParcelPilot AI Agent.",
) -> dict[str, str]:
    """
    Create a mocked escalation.

    This is a state-changing action. It should only be called after
    explicit user confirmation.
    """

    _ensure_escalations_file()

    df = pd.read_csv(ESCALATIONS_FILE)

    escalation_id = _generate_escalation_id(df)

    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    new_escalation = pd.DataFrame(
        [
            {
                "escalation_id": escalation_id,
                "ticket_id": ticket_id.upper(),
                "account_id": account_id.upper(),
                "reason": reason,
                "status": "OPEN",
                "created_at": created_at,
            }
        ]
    )

    updated_df = pd.concat(
        [df, new_escalation],
        ignore_index=True,
    )

    updated_df.to_csv(ESCALATIONS_FILE, index=False)

    return {
        "escalation_id": escalation_id,
        "ticket_id": ticket_id.upper(),
        "account_id": account_id.upper(),
        "status": "OPEN",
        "created_at": created_at,
    }