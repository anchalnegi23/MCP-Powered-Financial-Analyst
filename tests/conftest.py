"""Shared fixtures. Nothing here touches a live external API."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd
import pytest

NVDA_INFO: dict[str, Any] = {
    "symbol": "NVDA",
    "longName": "NVIDIA Corporation",
    "shortName": "NVIDIA",
    "currentPrice": 120.5,
    "regularMarketPrice": 120.5,
    "previousClose": 118.0,
    "regularMarketPreviousClose": 118.0,
    "dayHigh": 121.9,
    "dayLow": 117.4,
    "volume": 285_000_000,
    "marketState": "REGULAR",
    "currency": "USD",
    "sector": "Technology",
    "industry": "Semiconductors",
    "country": "United States",
    "website": "https://www.nvidia.com",
    "fullTimeEmployees": 29_600,
    "exchange": "NMS",
    "longBusinessSummary": "NVIDIA Corporation provides graphics and compute platforms. " * 30,
    "marketCap": 2_960_000_000_000,
    "enterpriseValue": 2_930_000_000_000,
    "sharesOutstanding": 24_600_000_000,
    "trailingPE": 51.2,
    "forwardPE": 33.8,
    "priceToBook": 48.9,
    "priceToSalesTrailing12Months": 27.1,
    "trailingEps": 2.35,
    "dividendYield": 0.0003,
    "beta": 1.68,
    "fiftyTwoWeekHigh": 153.1,
    "fiftyTwoWeekLow": 75.6,
    "fiftyDayAverage": 130.2,
    "twoHundredDayAverage": 118.7,
    "averageVolume": 310_000_000,
}

# A thinly covered listing: valid symbol, but the provider omits most statistics.
SPARSE_INFO: dict[str, Any] = {
    "symbol": "SPRS",
    "shortName": "Sparse Holdings",
    "previousClose": 9.5,
    "regularMarketPrice": 9.5,
    "currency": "EUR",
}


def make_history(days: int = 30, start_close: float = 100.0) -> pd.DataFrame:
    """Build a deterministic OHLCV frame shaped like yfinance's."""
    index = pd.to_datetime([date(2025, 1, 1) + timedelta(days=i) for i in range(days)])
    closes = [start_close + i for i in range(days)]
    return pd.DataFrame(
        {
            "Open": [c - 0.5 for c in closes],
            "High": [c + 1.0 for c in closes],
            "Low": [c - 1.0 for c in closes],
            "Close": closes,
            "Volume": [1_000_000 + i * 1000 for i in range(days)],
        },
        index=pd.Index(index, name="Date"),
    )


class FakeTicker:
    """Stand-in for yfinance.Ticker.

    Each attribute either returns canned data or raises the exception the test supplied,
    which is how provider failures are simulated without any network access.
    """

    def __init__(
        self,
        info: dict[str, Any] | None = None,
        history: pd.DataFrame | None = None,
        news: list[dict[str, Any]] | None = None,
        financials: dict[str, pd.DataFrame] | None = None,
        raises: Exception | None = None,
    ):
        self._info = info if info is not None else {}
        self._history = history
        self._news = news if news is not None else []
        self._financials = financials or {}
        self._raises = raises

    def _guard(self) -> None:
        if self._raises is not None:
            raise self._raises

    @property
    def info(self) -> dict[str, Any]:
        self._guard()
        return self._info

    @property
    def news(self) -> list[dict[str, Any]]:
        self._guard()
        return self._news

    def history(self, **_kwargs: Any) -> pd.DataFrame:
        self._guard()
        return self._history if self._history is not None else pd.DataFrame()

    def __getattr__(self, name: str) -> Any:
        # Statement frames: income_stmt, balance_sheet, cashflow and quarterly_*.
        if self._raises is not None:
            raise self._raises
        return self._financials.get(name, pd.DataFrame())


@pytest.fixture(autouse=True)
def reset_cache():
    """Start every test with an empty cache.

    Without this, a test that mocks one provider response would be served a value cached
    by an earlier test — which is exactly the bug caching introduces if you forget.
    """
    import asyncio

    from services.cache import MemoryCache, cache

    cache.backend = MemoryCache()
    cache.hits = cache.misses = 0
    cache.enabled = True
    yield
    asyncio.get_event_loop_policy()  # no-op; the fresh backend above is the reset


@pytest.fixture
def patch_ticker(monkeypatch):
    """Replace yfinance.Ticker everywhere it is used, for the duration of one test."""

    def _install(fake_or_factory):
        factory = (
            fake_or_factory
            if callable(fake_or_factory) and not isinstance(fake_or_factory, FakeTicker)
            else (lambda _symbol: fake_or_factory)
        )
        monkeypatch.setattr(
            "services.providers.yfinance_provider.yf.Ticker",
            lambda symbol, *a, **k: factory(symbol),
        )

    return _install


def make_statement(
    rows: dict[str, list[float | None]], columns: list[date] | None = None
) -> pd.DataFrame:
    """Build a provider-shaped statement frame: rows are line-item labels, columns dates."""
    columns = columns or [date(2024, 1, 28), date(2023, 1, 29), date(2022, 1, 30)]
    return pd.DataFrame(rows, index=pd.to_datetime(columns)).T


# NVIDIA-shaped annual income statement (FY2024, FY2023, FY2022), values in USD.
NVDA_INCOME = make_statement(
    {
        "Total Revenue": [60_922_000_000, 26_974_000_000, 26_914_000_000],
        "Cost Of Revenue": [16_621_000_000, 11_618_000_000, 9_439_000_000],
        "Gross Profit": [44_301_000_000, 15_356_000_000, 17_475_000_000],
        "Operating Expense": [11_329_000_000, 11_132_000_000, 7_434_000_000],
        "Operating Income": [32_972_000_000, 4_224_000_000, 10_041_000_000],
        "Pretax Income": [33_818_000_000, 4_181_000_000, 9_941_000_000],
        "Tax Provision": [4_058_000_000, -187_000_000, 189_000_000],
        "Net Income": [29_760_000_000, 4_368_000_000, 9_752_000_000],
        "Basic EPS": [12.05, 1.76, 3.91],
        "Diluted EPS": [11.93, 1.74, 3.85],
        "Research And Development": [8_675_000_000, 7_339_000_000, 5_268_000_000],
    }
)

NVDA_BALANCE = make_statement(
    {
        "Total Assets": [65_728_000_000, 41_182_000_000, 44_187_000_000],
        "Current Assets": [44_345_000_000, 23_073_000_000, 28_829_000_000],
        "Current Liabilities": [10_631_000_000, 6_563_000_000, 4_335_000_000],
        "Total Liabilities Net Minority Interest": [22_750_000_000, 19_081_000_000, 17_575_000_000],
        "Total Debt": [11_056_000_000, 12_031_000_000, 11_831_000_000],
        "Stockholders Equity": [42_978_000_000, 22_101_000_000, 26_612_000_000],
    }
)

NVDA_CASHFLOW = make_statement(
    {
        "Operating Cash Flow": [28_090_000_000, 5_641_000_000, 9_108_000_000],
        "Investing Cash Flow": [-10_566_000_000, 7_375_000_000, -9_830_000_000],
        "Financing Cash Flow": [-13_633_000_000, -11_617_000_000, 1_865_000_000],
        "Capital Expenditure": [-1_069_000_000, -1_833_000_000, -976_000_000],
        "Free Cash Flow": [27_021_000_000, 3_808_000_000, 8_132_000_000],
    }
)

# A statement missing the equity and debt rows entirely: forces the "unknown, not zero" path.
INCOMPLETE_BALANCE = make_statement(
    {"Total Assets": [1_000_000, 900_000], "Current Assets": [400_000, 380_000]},
    columns=[date(2024, 12, 31), date(2023, 12, 31)],
)

NVDA_FINANCIALS = {
    "income_stmt": NVDA_INCOME,
    "balance_sheet": NVDA_BALANCE,
    "cashflow": NVDA_CASHFLOW,
    "quarterly_income_stmt": NVDA_INCOME,
    "quarterly_balance_sheet": NVDA_BALANCE,
    "quarterly_cashflow": NVDA_CASHFLOW,
}


NVDA_NEWS_RAW = [
    {
        "content": {
            "title": "Nvidia unveils next-generation data centre GPU",
            "canonicalUrl": {"url": "https://example.com/nvidia-gpu-launch"},
            "provider": {"displayName": "Reuters"},
            "pubDate": "2025-03-18T14:30:00Z",
            "summary": "The chipmaker announced a new accelerator aimed at AI training.",
        }
    },
    {
        "content": {
            "title": "Analysts weigh Nvidia export restrictions",
            "canonicalUrl": {"url": "https://example.com/nvidia-export-rules"},
            "provider": {"displayName": "Bloomberg"},
            "pubDate": "2025-03-15T09:00:00Z",
            "summary": "New rules could affect China revenue.",
        }
    },
    {
        # Legacy flat shape, and no publication timestamp at all.
        "title": "Nvidia supplier commentary",
        "link": "https://example.com/nvidia-supplier",
        "publisher": "MarketWatch",
    },
    {
        # Unusable: no URL, so it cannot be cited and must be dropped.
        "content": {"title": "Headline with no link", "provider": {"displayName": "Nowhere"}}
    },
]
