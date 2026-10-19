"""Shared primitives used across every domain model."""

import re
from datetime import date, datetime
from typing import Annotated

from pydantic import BeforeValidator, Field

# Tickers are 1-10 characters: letters, digits, dot and dash (BRK.B, RDS-A, 7203.T).
_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")


def normalize_ticker(raw: str) -> str:
    """Upper-case and validate a ticker symbol at the trust boundary.

    Rejecting malformed symbols here means no provider ever receives an arbitrary string
    from the LLM, which is both a correctness and an injection concern.
    """
    if not isinstance(raw, str):
        raise ValueError(f"Ticker must be a string, got {type(raw).__name__}.")
    cleaned = raw.strip().upper()
    if not cleaned:
        raise ValueError("Ticker must not be empty.")
    if not _TICKER_RE.match(cleaned):
        raise ValueError(
            f"'{raw}' is not a valid ticker symbol. Expected 1-10 characters using "
            "letters, digits, '.' or '-', starting with a letter (e.g. NVDA, BRK.B)."
        )
    return cleaned


Ticker = Annotated[
    str,
    BeforeValidator(normalize_ticker),
    Field(description="Stock ticker symbol, e.g. 'NVDA'. Case-insensitive."),
]


def parse_date(value: str | date | datetime) -> date:
    """Accept an ISO date string or date object; reject anything else with a clear message."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value).strip())
    except ValueError as exc:
        raise ValueError(
            f"'{value}' is not a valid date. Use ISO format YYYY-MM-DD (e.g. 2025-01-31)."
        ) from exc
