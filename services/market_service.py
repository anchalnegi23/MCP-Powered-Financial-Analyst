"""Market data service: provider payloads in, domain models out.

Sits between the Market Data MCP tools and the provider adapter. All normalisation lives
here, so the MCP layer contains no vendor field names and no data-shaping logic.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import structlog

from models.common import normalize_ticker
from models.market import (
    CompanyProfile,
    HistoricalPrices,
    MarketCap,
    PricePoint,
    StockQuote,
    StockStatistics,
)
from services.cache import cached
from services.config import get_settings
from services.errors import DataUnavailableError
from services.providers import yfinance_provider as provider

logger = structlog.get_logger("service.market")

# Cap the number of bars returned over MCP. A 10-year daily series is ~2500 points, which
# would dominate the agent's context for no analytical gain.
MAX_PRICE_POINTS = 400

VALID_PERIODS = ("1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max")
VALID_INTERVALS = ("1d", "5d", "1wk", "1mo", "3mo")


def _first(info: dict[str, Any], *keys: str) -> Any:
    """Return the first present, non-null value among vendor keys.

    Providers rename fields between endpoints and versions; this keeps that churn out of
    the domain models.
    """
    for key in keys:
        value = info.get(key)
        if value is not None:
            return value
    return None


def _float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result and abs(result) != float("inf") else None


def _int(value: Any) -> int | None:
    number = _float(value)
    return int(number) if number is not None else None


def _timeout() -> float:
    return get_settings().http_timeout_seconds


@cached("market.quote", "quote")
async def get_stock_price(ticker: str) -> StockQuote:
    """Fetch the latest price snapshot for a symbol."""
    symbol = normalize_ticker(ticker)
    info = await provider.fetch_info(symbol, timeout=_timeout())

    price = _float(_first(info, "currentPrice", "regularMarketPrice", "previousClose"))
    if price is None:
        raise DataUnavailableError(f"No usable price is available for '{symbol}'.")

    previous_close = _float(_first(info, "regularMarketPreviousClose", "previousClose"))
    change = price - previous_close if previous_close is not None else None
    change_percent = (
        round(change / previous_close * 100, 4)
        if change is not None and previous_close
        else None
    )

    return StockQuote(
        ticker=symbol,
        company_name=_first(info, "longName", "shortName"),
        price=price,
        currency=_first(info, "currency") or "USD",
        previous_close=previous_close,
        change=round(change, 4) if change is not None else None,
        change_percent=change_percent,
        day_high=_float(_first(info, "dayHigh", "regularMarketDayHigh")),
        day_low=_float(_first(info, "dayLow", "regularMarketDayLow")),
        volume=_int(_first(info, "volume", "regularMarketVolume")),
        market_state=_first(info, "marketState"),
        as_of=datetime.now(UTC),
        source=provider.SOURCE,
    )


@cached("market.historical", "historical")
async def get_historical_prices(
    ticker: str, period: str = "1y", interval: str = "1d"
) -> HistoricalPrices:
    """Fetch a daily OHLCV series, down-sampled if it would otherwise be enormous."""
    symbol = normalize_ticker(ticker)
    if period not in VALID_PERIODS:
        raise ValueError(f"Invalid period '{period}'. Choose one of: {', '.join(VALID_PERIODS)}.")
    if interval not in VALID_INTERVALS:
        raise ValueError(
            f"Invalid interval '{interval}'. Choose one of: {', '.join(VALID_INTERVALS)}."
        )

    frame = await provider.fetch_history(
        symbol, period=period, interval=interval, timeout=_timeout()
    )

    records = frame.reset_index().to_dict("records")
    points: list[PricePoint] = []
    for row in records:
        raw_date = row.get("Date") or row.get("Datetime") or row.get("index")
        close = _float(row.get("Close"))
        if raw_date is None or close is None:
            continue  # provider gaps (holidays, halts) are dropped, not guessed at
        points.append(
            PricePoint(
                date=raw_date.date() if hasattr(raw_date, "date") else raw_date,
                open=_float(row.get("Open")) or close,
                high=_float(row.get("High")) or close,
                low=_float(row.get("Low")) or close,
                close=close,
                volume=_int(row.get("Volume")) or 0,
            )
        )

    if not points:
        raise DataUnavailableError(f"Price history for '{symbol}' contained no usable rows.")

    truncated = len(points) > MAX_PRICE_POINTS
    if truncated:
        # Keep evenly spaced samples plus the final bar, so the shape and the latest
        # close both survive.
        step = len(points) // MAX_PRICE_POINTS + 1
        sampled = points[::step]
        if sampled[-1] is not points[-1]:
            sampled.append(points[-1])
        points = sampled

    closes = [p.close for p in points]
    return HistoricalPrices(
        ticker=symbol,
        period=period,
        interval=interval,
        start_date=points[0].date,
        end_date=points[-1].date,
        point_count=len(points),
        prices=points,
        period_high=max(p.high for p in points),
        period_low=min(p.low for p in points),
        period_return_percent=round((closes[-1] - closes[0]) / closes[0] * 100, 4)
        if closes[0]
        else 0.0,
        truncated=truncated,
        source=provider.SOURCE,
    )


@cached("market.profile", "profile")
async def get_company_profile(ticker: str) -> CompanyProfile:
    """Fetch descriptive company information: sector, industry, size, business summary."""
    symbol = normalize_ticker(ticker)
    info = await provider.fetch_info(symbol, timeout=_timeout())

    description = _first(info, "longBusinessSummary")
    if isinstance(description, str) and len(description) > 1500:
        description = description[:1500].rsplit(" ", 1)[0] + "…"

    return CompanyProfile(
        ticker=symbol,
        company_name=_first(info, "longName", "shortName") or symbol,
        sector=_first(info, "sector"),
        industry=_first(info, "industry"),
        country=_first(info, "country"),
        website=_first(info, "website"),
        employees=_int(_first(info, "fullTimeEmployees")),
        exchange=_first(info, "exchange", "fullExchangeName"),
        currency=_first(info, "currency") or "USD",
        description=description,
        source=provider.SOURCE,
    )


@cached("market.cap", "profile")
async def get_market_cap(ticker: str) -> MarketCap:
    """Fetch market capitalisation, enterprise value and shares outstanding."""
    symbol = normalize_ticker(ticker)
    info = await provider.fetch_info(symbol, timeout=_timeout())

    market_cap = _float(_first(info, "marketCap"))
    if market_cap is None:
        raise DataUnavailableError(
            f"The provider did not supply a market capitalisation for '{symbol}'. This is "
            "common for funds, indices and some foreign listings."
        )

    return MarketCap(
        ticker=symbol,
        company_name=_first(info, "longName", "shortName"),
        market_cap=market_cap,
        enterprise_value=_float(_first(info, "enterpriseValue")),
        shares_outstanding=_float(_first(info, "sharesOutstanding")),
        currency=_first(info, "currency") or "USD",
        as_of=datetime.now(UTC),
        source=provider.SOURCE,
    )


# Domain field -> the vendor keys that may carry it, in order of preference.
_STATISTIC_KEYS: dict[str, tuple[str, ...]] = {
    "pe_ratio_trailing": ("trailingPE",),
    "pe_ratio_forward": ("forwardPE",),
    "price_to_book": ("priceToBook",),
    "price_to_sales": ("priceToSalesTrailing12Months",),
    "eps_trailing": ("trailingEps",),
    "beta": ("beta",),
    "fifty_two_week_high": ("fiftyTwoWeekHigh",),
    "fifty_two_week_low": ("fiftyTwoWeekLow",),
    "fifty_day_average": ("fiftyDayAverage",),
    "two_hundred_day_average": ("twoHundredDayAverage",),
}


@cached("market.statistics", "statistics")
async def get_stock_statistics(ticker: str) -> StockStatistics:
    """Fetch valuation and trading statistics, naming anything the provider omitted.

    Missing fields are reported explicitly rather than defaulted, so the agent can say
    "not available" instead of inventing a plausible number.
    """
    symbol = normalize_ticker(ticker)
    info = await provider.fetch_info(symbol, timeout=_timeout())

    values: dict[str, float | None] = {
        field: _float(_first(info, *keys)) for field, keys in _STATISTIC_KEYS.items()
    }

    raw_yield = _float(_first(info, "dividendYield"))
    # Providers disagree on whether dividend yield is a fraction or a percentage.
    dividend_yield = (
        None if raw_yield is None else round(raw_yield * 100 if raw_yield < 1 else raw_yield, 4)
    )

    average_volume = _int(_first(info, "averageVolume", "averageDailyVolume10Day"))
    missing = [field for field, value in values.items() if value is None]
    if dividend_yield is None:
        missing.append("dividend_yield_percent")
    if average_volume is None:
        missing.append("average_volume")

    return StockStatistics(
        ticker=symbol,
        company_name=_first(info, "longName", "shortName"),
        dividend_yield_percent=dividend_yield,
        average_volume=average_volume,
        currency=_first(info, "currency") or "USD",
        as_of=datetime.now(UTC),
        missing_fields=sorted(missing),
        source=provider.SOURCE,
        **values,
    )
