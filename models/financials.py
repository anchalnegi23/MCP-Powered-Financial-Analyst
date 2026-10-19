"""Domain models for financial statements.

Providers ship statements as wide, sparsely-populated tables whose row labels drift
between vendors and even between companies. These models pin down a small set of line
items we actually use, and record every one that was missing rather than defaulting it
to zero — a zero and an unknown lead to very different analysis.
"""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from models.common import Ticker

Period = Literal["annual", "quarterly"]


class StatementPeriod(BaseModel):
    """One reporting period's worth of line items from a single statement."""

    period_ending: date = Field(description="Last day of the reporting period.")
    fiscal_label: str = Field(description="Human label, e.g. 'FY2024' or 'Q3 2024'.")
    line_items: dict[str, float] = Field(
        description="Normalised line item name -> value, in the reporting currency. "
        "Only items the provider actually supplied appear here."
    )
    missing_items: list[str] = Field(
        default_factory=list,
        description="Expected line items this provider did not report for the period. "
        "Treat these as unknown, never as zero.",
    )


class FinancialStatement(BaseModel):
    """A normalised financial statement across several reporting periods."""

    ticker: Ticker
    company_name: str | None = None
    statement_type: Literal["income_statement", "balance_sheet", "cash_flow"]
    period: Period = Field(description="Whether the periods are annual or quarterly.")
    currency: str = "USD"
    periods: list[StatementPeriod] = Field(
        description="Reporting periods, most recent first."
    )
    source: str
    notes: list[str] = Field(
        default_factory=list,
        description="Caveats about completeness or comparability of this statement.",
    )


class FinancialRatios(BaseModel):
    """Ratios derived from a company's own reported statements.

    Every ratio here is computed in Python from reported line items, never estimated.
    A ratio whose inputs were missing is null and named in `missing_ratios`.
    """

    ticker: Ticker
    company_name: str | None = None
    period: Period
    period_ending: date
    fiscal_label: str
    currency: str = "USD"

    gross_margin_percent: float | None = None
    operating_margin_percent: float | None = None
    net_margin_percent: float | None = None
    return_on_equity_percent: float | None = None
    return_on_assets_percent: float | None = None
    debt_to_equity: float | None = None
    current_ratio: float | None = Field(
        default=None, description="Current assets / current liabilities: short-term liquidity."
    )
    missing_ratios: list[str] = Field(
        default_factory=list,
        description="Ratios that could not be computed because inputs were unavailable.",
    )
    inputs_used: dict[str, float] = Field(
        default_factory=dict,
        description="The reported line items the ratios were computed from, for audit.",
    )
    source: str


class EarningsPeriod(BaseModel):
    """Reported earnings for one period, with the analyst estimate where available."""

    fiscal_label: str
    period_ending: date | None = None
    revenue: float | None = None
    earnings: float | None = Field(default=None, description="Net income for the period.")
    eps_actual: float | None = None
    eps_estimate: float | None = None
    surprise_percent: float | None = Field(
        default=None, description="(actual - estimate) / |estimate| * 100, when both are known."
    )


class EarningsData(BaseModel):
    """Historical earnings, plus the next scheduled reporting date when published."""

    ticker: Ticker
    company_name: str | None = None
    period: Period
    currency: str = "USD"
    periods: list[EarningsPeriod] = Field(description="Most recent first.")
    next_earnings_date: date | None = None
    source: str
    notes: list[str] = Field(default_factory=list)
