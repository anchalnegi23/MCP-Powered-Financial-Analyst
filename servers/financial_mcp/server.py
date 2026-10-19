"""Financial Statements MCP server.

Exposes normalised income statements, balance sheets, cash-flow statements, derived
ratios and earnings history. Statement figures are *reported accounting data*: use this
server for anything a company published in its filings, and the market-data server for
anything about what the stock trades at.

    FastMCP tool -> financial_service -> provider adapter -> external API
"""

from __future__ import annotations

import os
import time
from typing import Annotated

import structlog
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import Field

from models.financials import EarningsData, FinancialRatios, FinancialStatement
from services import financial_service
from services.errors import ProviderError

logger = structlog.get_logger("financial_mcp")

mcp = FastMCP(
    name="financial-statements",
    instructions=(
        "Reported financial statements: income statement, balance sheet, cash flow, "
        "derived ratios and earnings history, annual or quarterly. Use this server to get "
        "the raw accounting figures a company published, then pass those figures to the "
        "financial-calculator server to compute growth rates and margins. Never compute "
        "them yourself."
    ),
)

TickerArg = Annotated[
    str,
    Field(
        description="Stock ticker symbol, e.g. 'NVDA'. Case-insensitive.",
        min_length=1,
        max_length=10,
    ),
]

PeriodArg = Annotated[
    str,
    Field(
        description="Reporting frequency: 'annual' for fiscal years, 'quarterly' for "
        "fiscal quarters. Anything else is rejected."
    ),
]


async def _run(tool: str, coro_factory, **context):
    """Invoke a service call, converting provider and validation errors into ToolError."""
    started = time.perf_counter()
    try:
        result = await coro_factory()
    except ProviderError as exc:
        logger.warning("tool_failed", tool=tool, error_type=type(exc).__name__, **context)
        raise ToolError(str(exc)) from exc
    except ValueError as exc:
        logger.warning("tool_rejected", tool=tool, reason=str(exc), **context)
        raise ToolError(str(exc)) from exc
    logger.info(
        "tool_completed",
        tool=tool,
        duration_ms=round((time.perf_counter() - started) * 1000, 2),
        **context,
    )
    return result


@mcp.tool
async def get_income_statement(
    ticker: TickerArg, period: PeriodArg = "annual"
) -> FinancialStatement:
    """Get a company's reported income statement across recent periods.

    Use this for revenue, gross profit, operating income, net income, EPS and R&D spend —
    that is, any question about what a company earned. To compute revenue growth or a
    margin from these figures, pass them to the calculator server rather than working the
    arithmetic out yourself.

    Returns a FinancialStatement with up to 5 periods, most recent first. Each period has
    `line_items` (only what the provider actually reported) and `missing_items` (expected
    items it did not). A missing item means unknown, never zero.

    Errors: raises for a malformed or unknown ticker, an invalid period, a provider
    timeout or rate limit, or when the company has no statement of this type — common for
    recent IPOs and for non-US listings with thin coverage.

    Limitation: fiscal years do not align across companies. NVIDIA's FY2024 ends in
    January 2024; AMD's ends in December. Compare `period_ending`, not just the label.
    """
    return await _run(
        "get_income_statement",
        lambda: financial_service.get_income_statement(ticker, period),
        ticker=ticker,
        period=period,
    )


@mcp.tool
async def get_balance_sheet(ticker: TickerArg, period: PeriodArg = "annual") -> FinancialStatement:
    """Get a company's reported balance sheet across recent periods.

    Use this for assets, liabilities, debt, equity, cash and inventory — anything about
    financial position and solvency. It supplies the inputs for debt-to-equity, ROE and
    current-ratio calculations.

    Returns a FinancialStatement with up to 5 periods, most recent first, in the same
    shape as the income statement.

    Errors: raises for a malformed or unknown ticker, an invalid period, a provider
    timeout or rate limit, or when no balance sheet is available.

    Limitation: "total debt" is a provider-assembled figure whose treatment of operating
    leases varies. Check `missing_items` before drawing conclusions from an absent value.
    """
    return await _run(
        "get_balance_sheet",
        lambda: financial_service.get_balance_sheet(ticker, period),
        ticker=ticker,
        period=period,
    )


@mcp.tool
async def get_cash_flow(ticker: TickerArg, period: PeriodArg = "annual") -> FinancialStatement:
    """Get a company's reported cash-flow statement across recent periods.

    Use this to check whether accounting profit converts into actual cash, and for
    questions about capital expenditure, buybacks, dividends or free cash flow. Cash flow
    is the standard cross-check on earnings quality.

    Returns a FinancialStatement with operating, investing and financing cash flow, capital
    expenditure, free cash flow, dividends paid and stock repurchases where reported.

    Errors: raises for a malformed or unknown ticker, an invalid period, a provider
    timeout or rate limit, or when no cash-flow statement is available.

    Limitation: `free_cash_flow` is a provider-computed convenience figure. When it is
    missing, derive it as operating_cash_flow + capital_expenditure (capex is reported
    negative) using the calculator server.
    """
    return await _run(
        "get_cash_flow",
        lambda: financial_service.get_cash_flow(ticker, period),
        ticker=ticker,
        period=period,
    )


@mcp.tool
async def get_financial_ratios(ticker: TickerArg, period: PeriodArg = "annual") -> FinancialRatios:
    """Get profitability, leverage and liquidity ratios computed from reported statements.

    Use this as a fast overview of a company's financial health when you need several
    ratios at once, instead of fetching statements and computing each separately. Every
    ratio here is derived in Python from reported line items, and `inputs_used` shows
    exactly which figures produced them.

    Returns FinancialRatios: gross/operating/net margin, ROE, ROA, debt-to-equity and
    current ratio for the most recent period, plus `missing_ratios` naming any that could
    not be computed because an input was unavailable.

    Errors: raises for a malformed or unknown ticker, an invalid period, a provider
    timeout or rate limit, or when the underlying statements are unavailable.

    Limitation: ratios reflect the latest reported period only. For trends, fetch the
    statements and compute across periods with the calculator server.
    """
    return await _run(
        "get_financial_ratios",
        lambda: financial_service.get_financial_ratios(ticker, period),
        ticker=ticker,
        period=period,
    )


@mcp.tool
async def get_earnings_data(ticker: TickerArg, period: PeriodArg = "annual") -> EarningsData:
    """Get historical revenue, net income and EPS per reporting period.

    Use this when the question is about earnings history or trajectory — "has earnings
    grown", "what did they earn last quarter" — and you want the series rather than the
    full income statement.

    Returns EarningsData with one entry per period, most recent first.

    Errors: raises for a malformed or unknown ticker, an invalid period, a provider
    timeout or rate limit, or when no earnings history exists.

    Limitation: this provider supplies no analyst estimates, so `eps_estimate` and
    `surprise_percent` are always null and the `notes` field says so. Do not describe an
    earnings beat or miss from this data.
    """
    return await _run(
        "get_earnings_data",
        lambda: financial_service.get_earnings_data(ticker, period),
        ticker=ticker,
        period=period,
    )


def main() -> None:
    """Console entrypoint. Defaults to stdio; set MCP_TRANSPORT=http for a networked server."""
    if os.getenv("MCP_TRANSPORT", "stdio") == "stdio":
        mcp.run()
    else:
        mcp.run(
            transport="http",
            host=os.getenv("MCP_HOST", "0.0.0.0"),
            port=int(os.getenv("MCP_PORT", "8003")),
        )


if __name__ == "__main__":
    main()
