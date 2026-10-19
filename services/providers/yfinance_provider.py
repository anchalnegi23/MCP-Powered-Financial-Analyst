"""Thin adapter around yfinance.

This is the *only* module in the codebase that knows yfinance exists. It does two jobs
and nothing else:

1. Run yfinance's blocking calls off the event loop, with a timeout.
2. Translate every vendor exception into one of our :mod:`services.errors` types.

It deliberately returns raw provider dicts/frames — normalisation into domain models is
the service layer's job, one level up. Splitting it this way means a future provider
swap touches this file plus a mapping function, not the MCP tools.
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog
import yfinance as yf
from yfinance.exceptions import YFRateLimitError

from services.errors import (
    DataUnavailableError,
    ProviderError,
    ProviderTimeoutError,
    RateLimitError,
    TickerNotFoundError,
)
from services.redaction import scrub

logger = structlog.get_logger("provider.yfinance")

SOURCE = "yfinance"

# yfinance populates .info even for nonsense symbols; a real listing always has at least
# one of these keys with a usable value.
_LIVENESS_KEYS = ("regularMarketPrice", "currentPrice", "previousClose", "longName", "shortName")


def _classify(exc: Exception, symbol: str) -> ProviderError:
    """Map a vendor exception onto our closed error set.

    Matching on message text as well as type is deliberate: yfinance surfaces rate limits
    and 404s inconsistently depending on which endpoint answered.
    """
    if isinstance(exc, YFRateLimitError):
        return RateLimitError(
            f"The market data provider rate-limited the request for {symbol}. "
            "Wait before retrying, or supply a provider API key with a higher quota."
        )
    if isinstance(exc, TimeoutError):
        return ProviderTimeoutError(f"The market data provider timed out for {symbol}.")

    text = str(exc).lower()
    if "too many requests" in text or "rate limit" in text or "429" in text:
        return RateLimitError(f"The market data provider rate-limited the request for {symbol}.")
    if "timed out" in text or "timeout" in text:
        return ProviderTimeoutError(f"The market data provider timed out for {symbol}.")
    if "not found" in text or "no data found" in text or "delisted" in text or "404" in text:
        return TickerNotFoundError(f"No market data found for '{symbol}'.")
    # The vendor message is scrubbed and truncated: it can echo back the failing URL,
    # which may carry an API key, and it would otherwise reach the LLM verbatim.
    return ProviderError(
        f"The market data provider failed for {symbol} ({type(exc).__name__}): "
        f"{scrub(str(exc))[:200]}"
    )


async def _call(fn, *args, symbol: str, timeout: float, **kwargs) -> Any:
    """Run a blocking yfinance call in a worker thread under a hard timeout."""
    try:
        return await asyncio.wait_for(asyncio.to_thread(fn, *args, **kwargs), timeout=timeout)
    except TimeoutError as exc:
        logger.warning("provider_timeout", symbol=symbol, timeout=timeout)
        raise ProviderTimeoutError(
            f"The market data provider did not respond within {timeout}s for {symbol}."
        ) from exc
    except Exception as exc:  # noqa: BLE001 - deliberately funnelled through _classify
        error = _classify(exc, symbol)
        logger.warning(
            "provider_error",
            symbol=symbol,
            error_type=type(error).__name__,
            detail=scrub(str(exc))[:200],
        )
        raise error from exc


async def fetch_info(symbol: str, *, timeout: float) -> dict[str, Any]:
    """Return the provider's combined quote/profile/statistics payload for a symbol.

    Raises TickerNotFoundError when the symbol has no usable data, which is how an
    invalid ticker is detected: yfinance returns an almost-empty dict rather than failing.
    """
    info = await _call(lambda: yf.Ticker(symbol).info, symbol=symbol, timeout=timeout)
    if not isinstance(info, dict) or not any(info.get(k) is not None for k in _LIVENESS_KEYS):
        raise TickerNotFoundError(
            f"'{symbol}' returned no market data. It may be delisted, misspelled, or not "
            "covered by this provider."
        )
    return info


async def fetch_history(symbol: str, *, period: str, interval: str, timeout: float) -> Any:
    """Return the provider's OHLCV frame. Raises DataUnavailableError when it is empty."""
    frame = await _call(
        lambda: yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=True),
        symbol=symbol,
        timeout=timeout,
    )
    if frame is None or getattr(frame, "empty", True):
        raise DataUnavailableError(
            f"No price history for '{symbol}' over period '{period}' at interval '{interval}'. "
            "The symbol may be too recently listed, or the period/interval pair unsupported."
        )
    return frame


async def fetch_financials(symbol: str, *, statement: str, quarterly: bool, timeout: float) -> Any:
    """Return an income statement, balance sheet or cash-flow frame.

    `statement` is one of 'income', 'balance', 'cashflow'.
    """
    attribute = {
        "income": ("quarterly_income_stmt", "income_stmt"),
        "balance": ("quarterly_balance_sheet", "balance_sheet"),
        "cashflow": ("quarterly_cashflow", "cashflow"),
    }[statement][0 if quarterly else 1]

    frame = await _call(
        lambda: getattr(yf.Ticker(symbol), attribute), symbol=symbol, timeout=timeout
    )
    if frame is None or getattr(frame, "empty", True):
        raise DataUnavailableError(
            f"No {'quarterly' if quarterly else 'annual'} {statement} statement available "
            f"for '{symbol}'."
        )
    return frame


async def fetch_news(symbol: str, *, count: int, timeout: float) -> list[dict[str, Any]]:
    """Return the provider's raw news items for a symbol. An empty list is a valid answer."""
    items = await _call(
        lambda: yf.Ticker(symbol).news, symbol=symbol, timeout=timeout
    )
    return list(items or [])[:count]
