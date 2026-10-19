"""Market Data MCP server.

Exposes live and historical market data as MCP tools. The tools themselves contain no
provider logic: each validates input, delegates to :mod:`services.market_service`, and
converts domain errors into MCP tool errors.

    FastMCP tool -> service -> provider adapter -> external API
                 <- domain model <- normalised payload <-
"""

from __future__ import annotations

import os
import time
from typing import Annotated

import structlog
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import Field

from models.market import (
    CompanyProfile,
    HistoricalPrices,
    MarketCap,
    StockQuote,
    StockStatistics,
)
from services import market_service
from services.errors import ProviderError

logger = structlog.get_logger("market_mcp")

mcp = FastMCP(
    name="market-data",
    instructions=(
        "Live and historical stock market data: prices, price history, company profiles, "
        "market capitalisation and valuation statistics. Use this server for anything about "
        "what a stock is worth *right now* or how it has traded. For accounting figures "
        "(revenue, income, balance sheet) use the financial-statements server instead."
    ),
)

TickerArg = Annotated[
    str,
    Field(
        description="Stock ticker symbol, e.g. 'NVDA'. Case-insensitive. Must be 1-10 "
        "characters of letters, digits, '.' or '-'.",
        min_length=1,
        max_length=10,
    ),
]


async def _run(tool: str, coro_factory, **context):
    """Invoke a service call, converting provider and validation errors into ToolError.

    Every failure mode the provider can produce — unknown ticker, timeout, rate limit,
    missing data — reaches the agent as a readable message rather than a stack trace,
    and the server stays available.
    """
    started = time.perf_counter()
    try:
        result = await coro_factory()
    except ProviderError as exc:
        logger.warning(
            "tool_failed",
            tool=tool,
            error_type=type(exc).__name__,
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
            **context,
        )
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
async def get_stock_price(ticker: TickerArg) -> StockQuote:
    """Get the current share price and today's move for one company.

    Use this whenever the question asks what a stock costs now, how it moved today, or
    needs a live price as an input to a valuation calculation (for example, feeding
    `calculate_pe_ratio` on the calculator server).

    Returns a StockQuote: price, currency, previous close, absolute and percentage change,
    day high/low, volume, market state, and the UTC timestamp of retrieval. Fields the
    provider omits are null rather than zero.

    Errors: raises if the ticker is malformed, unknown or delisted, if the provider times
    out, or if it rate-limits the request. Each case returns a distinct message.

    Limitation: prices may be delayed by up to 15 minutes and are last-close outside
    market hours. This is not a real-time trading feed.
    """
    return await _run(
        "get_stock_price", lambda: market_service.get_stock_price(ticker), ticker=ticker
    )


@mcp.tool
async def get_historical_prices(
    ticker: TickerArg,
    period: Annotated[
        str,
        Field(
            description="Look-back window. One of: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, "
            "10y, ytd, max."
        ),
    ] = "1y",
    interval: Annotated[str, Field(description="Bar size. One of: 1d, 5d, 1wk, 1mo, 3mo.")] = "1d",
) -> HistoricalPrices:
    """Get historical OHLCV price bars and the return over the window.

    Use this for questions about performance over time, volatility, drawdowns, or to get
    the endpoints for a share-price CAGR calculation. Prefer a coarser `interval` for long
    windows: '1wk' or '1mo' for 5y and 10y.

    Returns HistoricalPrices: the bars oldest-first, plus period high, period low, the
    close-to-close return as a percentage, and the actual date range covered.

    Long series are down-sampled to at most 400 evenly spaced bars (the latest bar is
    always kept) and `truncated` is set to true, so responses stay a usable size.

    Errors: raises for an unknown ticker, an invalid period or interval, a provider
    timeout or rate limit, or when the symbol has no history for that window — a newly
    listed company asked for '10y', for example.
    """
    return await _run(
        "get_historical_prices",
        lambda: market_service.get_historical_prices(ticker, period=period, interval=interval),
        ticker=ticker,
        period=period,
        interval=interval,
    )


@mcp.tool
async def get_company_profile(ticker: TickerArg) -> CompanyProfile:
    """Get descriptive information about a company: sector, industry, size and business summary.

    Use this to establish what a company actually does before analysing it, to identify
    the right peer group for a comparison, or when the user asks "what is this company".

    Returns a CompanyProfile: company name, sector, industry, country, exchange, website,
    employee count and a business summary truncated to roughly 1500 characters.

    Errors: raises for a malformed or unknown ticker, a provider timeout, or a rate limit.
    Individual descriptive fields may be null when the provider does not cover them —
    common for smaller and non-US listings.
    """
    return await _run(
        "get_company_profile", lambda: market_service.get_company_profile(ticker), ticker=ticker
    )


@mcp.tool
async def get_market_cap(ticker: TickerArg) -> MarketCap:
    """Get market capitalisation, enterprise value and shares outstanding.

    Use this for size comparisons between companies, and as the input for valuation
    questions that need company-level rather than per-share figures.

    Returns a MarketCap in the company's reporting currency. `enterprise_value` may be
    null when the provider does not compute it.

    Errors: raises for an unknown ticker, a provider timeout or rate limit, or when no
    market capitalisation is published — common for funds, indices and some foreign listings.

    Limitation: market caps from different currencies are not directly comparable; check
    the `currency` field before ranking companies.
    """
    return await _run(
        "get_market_cap", lambda: market_service.get_market_cap(ticker), ticker=ticker
    )


@mcp.tool
async def get_stock_statistics(ticker: TickerArg) -> StockStatistics:
    """Get valuation and trading statistics: P/E, price-to-book, EPS, beta, 52-week range.

    Use this for valuation questions and peer comparison. It is the fastest way to get a
    trailing or forward P/E without separately fetching price and EPS. When you need a
    P/E computed from specific figures you already hold, use the calculator server instead
    so the arithmetic is auditable.

    Returns StockStatistics. Crucially, it also returns `missing_fields`: the names of any
    statistics the provider did not supply. Report those as unavailable — do not substitute
    an estimate.

    Errors: raises for an unknown ticker, a provider timeout, or a rate limit.

    Limitation: trailing P/E is null for loss-making companies, and forward P/E depends on
    analyst estimates that the provider may not carry.
    """
    return await _run(
        "get_stock_statistics", lambda: market_service.get_stock_statistics(ticker), ticker=ticker
    )


def main() -> None:
    """Console entrypoint. Defaults to stdio; set MCP_TRANSPORT=http for a networked server."""
    if os.getenv("MCP_TRANSPORT", "stdio") == "stdio":
        mcp.run()
    else:
        mcp.run(
            transport="http",
            host=os.getenv("MCP_HOST", "0.0.0.0"),
            port=int(os.getenv("MCP_PORT", "8002")),
        )


if __name__ == "__main__":
    main()
