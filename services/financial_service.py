"""Financial statements service: provider tables in, normalised statements out.

Provider statement frames are wide tables whose row labels vary by vendor and company
("Total Revenue", "TotalRevenue", "Revenues"). This module owns that mess: it maps the
labels it recognises onto a fixed vocabulary and reports everything it could not find,
so no layer above ever guesses at a line item.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import structlog

from models.common import normalize_ticker
from models.financials import (
    EarningsData,
    EarningsPeriod,
    FinancialRatios,
    FinancialStatement,
    Period,
    StatementPeriod,
)
from services.cache import cached
from services.config import get_settings
from services.errors import DataUnavailableError
from services.providers import yfinance_provider as provider

logger = structlog.get_logger("service.financial")

MAX_PERIODS = 5

# Our vocabulary -> the provider row labels that may carry it, in preference order.
INCOME_ITEMS: dict[str, tuple[str, ...]] = {
    "revenue": ("Total Revenue", "TotalRevenue", "Operating Revenue"),
    "cost_of_revenue": ("Cost Of Revenue", "CostOfRevenue"),
    "gross_profit": ("Gross Profit", "GrossProfit"),
    "operating_expenses": ("Operating Expense", "Total Operating Expenses"),
    "operating_income": ("Operating Income", "OperatingIncome", "EBIT"),
    "pretax_income": ("Pretax Income", "Income Before Tax"),
    "tax_provision": ("Tax Provision", "Income Tax Expense"),
    "net_income": ("Net Income", "NetIncome", "Net Income Common Stockholders"),
    "eps_basic": ("Basic EPS", "BasicEPS"),
    "eps_diluted": ("Diluted EPS", "DilutedEPS"),
    "ebitda": ("EBITDA", "Normalized EBITDA"),
    "research_and_development": ("Research And Development",),
}

BALANCE_ITEMS: dict[str, tuple[str, ...]] = {
    "total_assets": ("Total Assets", "TotalAssets"),
    "current_assets": ("Current Assets", "Total Current Assets"),
    "cash_and_equivalents": (
        "Cash And Cash Equivalents",
        "Cash Cash Equivalents And Short Term Investments",
    ),
    "inventory": ("Inventory",),
    "total_liabilities": (
        "Total Liabilities Net Minority Interest",
        "Total Liabilities",
    ),
    "current_liabilities": ("Current Liabilities", "Total Current Liabilities"),
    "total_debt": ("Total Debt", "TotalDebt"),
    "long_term_debt": ("Long Term Debt", "LongTermDebt"),
    "total_equity": (
        "Stockholders Equity",
        "Total Equity Gross Minority Interest",
        "StockholdersEquity",
    ),
    "retained_earnings": ("Retained Earnings",),
}

CASHFLOW_ITEMS: dict[str, tuple[str, ...]] = {
    "operating_cash_flow": ("Operating Cash Flow", "Total Cash From Operating Activities"),
    "investing_cash_flow": ("Investing Cash Flow", "Total Cashflows From Investing Activities"),
    "financing_cash_flow": ("Financing Cash Flow", "Total Cash From Financing Activities"),
    "capital_expenditure": ("Capital Expenditure", "CapitalExpenditure"),
    "free_cash_flow": ("Free Cash Flow", "FreeCashFlow"),
    "net_change_in_cash": ("Changes In Cash", "Change In Cash"),
    "dividends_paid": ("Cash Dividends Paid", "Common Stock Dividend Paid"),
    "stock_repurchase": ("Repurchase Of Capital Stock",),
}

_STATEMENTS: dict[str, tuple[str, dict[str, tuple[str, ...]]]] = {
    "income_statement": ("income", INCOME_ITEMS),
    "balance_sheet": ("balance", BALANCE_ITEMS),
    "cash_flow": ("cashflow", CASHFLOW_ITEMS),
}


def validate_period(period: str) -> Period:
    """Reject anything that is not 'annual' or 'quarterly', with a usable message."""
    normalised = str(period).strip().lower()
    if normalised in ("annual", "yearly", "year", "fy"):
        return "annual"
    if normalised in ("quarterly", "quarter", "q"):
        return "quarterly"
    raise ValueError(
        f"Invalid period '{period}'. Use 'annual' or 'quarterly'."
    )


def _float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result and abs(result) != float("inf") else None


def _fiscal_label(period_end: date, period: Period) -> str:
    if period == "annual":
        return f"FY{period_end.year}"
    return f"Q{(period_end.month - 1) // 3 + 1} {period_end.year}"


def _as_date(value: Any) -> date | None:
    if isinstance(value, date):
        return value
    if hasattr(value, "date"):
        return value.date()
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _extract_periods(
    frame: Any, vocabulary: dict[str, tuple[str, ...]], period: Period
) -> list[StatementPeriod]:
    """Turn a provider's wide table into our per-period line items.

    Columns are reporting dates, rows are provider line-item labels.
    """
    labels = {str(index).strip(): index for index in frame.index}
    periods: list[StatementPeriod] = []

    for column in list(frame.columns)[:MAX_PERIODS]:
        period_end = _as_date(column)
        if period_end is None:
            continue

        items: dict[str, float] = {}
        missing: list[str] = []
        for name, candidates in vocabulary.items():
            value = None
            for candidate in candidates:
                if candidate in labels:
                    value = _float(frame.at[labels[candidate], column])
                    if value is not None:
                        break
            if value is None:
                missing.append(name)
            else:
                items[name] = value

        periods.append(
            StatementPeriod(
                period_ending=period_end,
                fiscal_label=_fiscal_label(period_end, period),
                line_items=items,
                missing_items=missing,
            )
        )

    periods.sort(key=lambda p: p.period_ending, reverse=True)
    return periods


async def _company_name(symbol: str) -> str | None:
    """Best-effort company name. A failure here must not fail the statement request."""
    try:
        info = await provider.fetch_info(symbol, timeout=get_settings().http_timeout_seconds)
    except Exception:  # noqa: BLE001 - descriptive only; the statement is what matters
        return None
    return info.get("longName") or info.get("shortName")


@cached("financial.statement", "statement")
async def get_statement(
    ticker: str, statement_type: str, period: str = "annual"
) -> FinancialStatement:
    """Fetch and normalise one financial statement."""
    symbol = normalize_ticker(ticker)
    resolved_period = validate_period(period)
    if statement_type not in _STATEMENTS:
        raise ValueError(
            f"Unknown statement type '{statement_type}'. "
            f"Choose one of: {', '.join(_STATEMENTS)}."
        )

    provider_key, vocabulary = _STATEMENTS[statement_type]
    frame = await provider.fetch_financials(
        symbol,
        statement=provider_key,
        quarterly=resolved_period == "quarterly",
        timeout=get_settings().http_timeout_seconds,
    )

    periods = _extract_periods(frame, vocabulary, resolved_period)
    if not periods:
        raise DataUnavailableError(
            f"The {statement_type.replace('_', ' ')} for '{symbol}' contained no readable "
            "reporting periods."
        )

    notes: list[str] = []
    always_missing = set(vocabulary) - {k for p in periods for k in p.line_items}
    if always_missing:
        notes.append(
            "Not reported by this provider for any period: "
            + ", ".join(sorted(always_missing))
            + ". Treat these as unknown, not zero."
        )

    return FinancialStatement(
        ticker=symbol,
        company_name=await _company_name(symbol),
        statement_type=statement_type,  # type: ignore[arg-type]
        period=resolved_period,
        periods=periods,
        source=provider.SOURCE,
        notes=notes,
    )


async def get_income_statement(ticker: str, period: str = "annual") -> FinancialStatement:
    return await get_statement(ticker, "income_statement", period)


async def get_balance_sheet(ticker: str, period: str = "annual") -> FinancialStatement:
    return await get_statement(ticker, "balance_sheet", period)


async def get_cash_flow(ticker: str, period: str = "annual") -> FinancialStatement:
    return await get_statement(ticker, "cash_flow", period)


def _ratio(numerator: float | None, denominator: float | None, *, scale: float = 1.0):
    """Compute a ratio, or return None when an input is missing or the denominator is zero."""
    if numerator is None or denominator is None or denominator == 0:
        return None
    return round(numerator / denominator * scale, 4)


@cached("financial.ratios", "ratios")
async def get_financial_ratios(ticker: str, period: str = "annual") -> FinancialRatios:
    """Compute ratios from the company's own reported statements.

    Deliberately derived from reported line items rather than taken from a provider's
    pre-computed ratio endpoint, so `inputs_used` can prove where each number came from.
    """
    symbol = normalize_ticker(ticker)
    resolved_period = validate_period(period)

    income = await get_statement(symbol, "income_statement", resolved_period)
    balance = await get_statement(symbol, "balance_sheet", resolved_period)

    latest_income = income.periods[0]
    # Align the balance sheet to the same period end where possible; a mismatch would
    # silently mix, say, FY2024 income with FY2023 equity.
    latest_balance = next(
        (p for p in balance.periods if p.period_ending == latest_income.period_ending),
        balance.periods[0],
    )

    items = {**latest_balance.line_items, **latest_income.line_items}
    get = items.get

    ratios = {
        "gross_margin_percent": _ratio(get("gross_profit"), get("revenue"), scale=100),
        "operating_margin_percent": _ratio(get("operating_income"), get("revenue"), scale=100),
        "net_margin_percent": _ratio(get("net_income"), get("revenue"), scale=100),
        "return_on_equity_percent": _ratio(get("net_income"), get("total_equity"), scale=100),
        "return_on_assets_percent": _ratio(get("net_income"), get("total_assets"), scale=100),
        "debt_to_equity": _ratio(get("total_debt"), get("total_equity")),
        "current_ratio": _ratio(get("current_assets"), get("current_liabilities")),
    }

    return FinancialRatios(
        ticker=symbol,
        company_name=income.company_name,
        period=resolved_period,
        period_ending=latest_income.period_ending,
        fiscal_label=latest_income.fiscal_label,
        missing_ratios=sorted(name for name, value in ratios.items() if value is None),
        inputs_used=items,
        source=provider.SOURCE,
        **ratios,
    )


@cached("financial.earnings", "earnings")
async def get_earnings_data(ticker: str, period: str = "annual") -> EarningsData:
    """Assemble reported revenue, net income and EPS per period from the income statement."""
    symbol = normalize_ticker(ticker)
    resolved_period = validate_period(period)
    income = await get_statement(symbol, "income_statement", resolved_period)

    periods = [
        EarningsPeriod(
            fiscal_label=p.fiscal_label,
            period_ending=p.period_ending,
            revenue=p.line_items.get("revenue"),
            earnings=p.line_items.get("net_income"),
            eps_actual=p.line_items.get("eps_diluted") or p.line_items.get("eps_basic"),
        )
        for p in income.periods
    ]

    notes = [
        "Analyst estimates and earnings surprises are not available from this provider; "
        "`eps_estimate` and `surprise_percent` are therefore null."
    ]
    return EarningsData(
        ticker=symbol,
        company_name=income.company_name,
        period=resolved_period,
        periods=periods,
        source=provider.SOURCE,
        notes=notes,
    )
