"""Market Data MCP: service normalisation, error mapping and the MCP tool surface.

Every test mocks yfinance. Nothing here reaches the network; the single live test is
marked `integration` and deselected by default.
"""

from __future__ import annotations

import sys

import pytest
from fastmcp import Client
from fastmcp.client.transports import StdioTransport
from fastmcp.exceptions import ToolError
from yfinance.exceptions import YFRateLimitError

from servers.market_mcp.server import mcp
from services import market_service
from services.errors import (
    DataUnavailableError,
    ProviderTimeoutError,
    RateLimitError,
    TickerNotFoundError,
)
from tests.conftest import NVDA_INFO, SPARSE_INFO, FakeTicker, make_history

EXPECTED_TOOLS = {
    "get_stock_price",
    "get_historical_prices",
    "get_company_profile",
    "get_market_cap",
    "get_stock_statistics",
}


@pytest.fixture
async def client():
    async with Client(mcp) as c:
        yield c


# ── Discovery ─────────────────────────────────────────────────────────────────


async def test_all_market_tools_are_discoverable(client):
    assert {t.name for t in await client.list_tools()} == EXPECTED_TOOLS


async def test_every_tool_documents_usage_returns_and_errors(client):
    for tool in await client.list_tools():
        description = tool.description or ""
        assert "Use this" in description, f"{tool.name} does not say when to use it"
        assert "Returns" in description, f"{tool.name} does not document its return value"
        assert "Errors" in description, f"{tool.name} does not document failure modes"
        assert tool.output_schema, f"{tool.name} has no structured output schema"


# ── Happy path: normalisation into domain models ──────────────────────────────


async def test_stock_price_is_normalised(patch_ticker):
    patch_ticker(FakeTicker(info=NVDA_INFO))
    quote = await market_service.get_stock_price("nvda")

    assert quote.ticker == "NVDA"  # lower-case input normalised
    assert quote.company_name == "NVIDIA Corporation"
    assert quote.price == 120.5
    assert quote.previous_close == 118.0
    assert quote.change == 2.5
    assert quote.change_percent == pytest.approx(2.1186, abs=1e-4)
    assert quote.source == "yfinance"


async def test_historical_prices_compute_window_statistics(patch_ticker):
    patch_ticker(FakeTicker(history=make_history(days=30, start_close=100.0)))
    series = await market_service.get_historical_prices("NVDA", period="1mo")

    assert series.point_count == 30
    assert series.prices[0].close == 100.0
    assert series.prices[-1].close == 129.0
    assert series.period_high == 130.0
    assert series.period_low == 99.0
    assert series.period_return_percent == 29.0
    assert series.truncated is False


async def test_long_series_is_downsampled_and_flagged(patch_ticker):
    patch_ticker(FakeTicker(history=make_history(days=1200)))
    series = await market_service.get_historical_prices("NVDA", period="5y")

    assert series.truncated is True
    assert series.point_count <= market_service.MAX_PRICE_POINTS + 1
    assert series.prices[-1].close == 1299.0  # the latest bar always survives


async def test_company_profile_truncates_a_long_summary(patch_ticker):
    patch_ticker(FakeTicker(info=NVDA_INFO))
    profile = await market_service.get_company_profile("NVDA")

    assert profile.sector == "Technology"
    assert profile.employees == 29_600
    assert profile.description is not None
    assert len(profile.description) <= 1501
    assert profile.description.endswith("…")


async def test_market_cap_is_normalised(patch_ticker):
    patch_ticker(FakeTicker(info=NVDA_INFO))
    cap = await market_service.get_market_cap("NVDA")

    assert cap.market_cap == 2_960_000_000_000
    assert cap.enterprise_value == 2_930_000_000_000
    assert cap.currency == "USD"


async def test_statistics_normalise_dividend_yield_from_a_fraction(patch_ticker):
    patch_ticker(FakeTicker(info=NVDA_INFO))
    stats = await market_service.get_stock_statistics("NVDA")

    assert stats.pe_ratio_trailing == 51.2
    assert stats.dividend_yield_percent == 0.03  # 0.0003 fraction -> percent
    assert stats.missing_fields == []


async def test_missing_statistics_are_named_not_defaulted(patch_ticker):
    """The agent must be able to say 'not available' rather than invent a number."""
    patch_ticker(FakeTicker(info=SPARSE_INFO))
    stats = await market_service.get_stock_statistics("SPRS")

    assert stats.pe_ratio_trailing is None
    assert "pe_ratio_trailing" in stats.missing_fields
    assert "beta" in stats.missing_fields
    assert stats.currency == "EUR"


# ── Failure handling ──────────────────────────────────────────────────────────


async def test_unknown_ticker_raises_a_controlled_error(patch_ticker):
    patch_ticker(FakeTicker(info={"symbol": "ZZZZ", "trailingPegRatio": None}))
    with pytest.raises(TickerNotFoundError, match="no market data"):
        await market_service.get_stock_price("ZZZZ")


async def test_provider_timeout_is_mapped(patch_ticker):
    patch_ticker(FakeTicker(raises=TimeoutError("connection timed out")))
    with pytest.raises(ProviderTimeoutError):
        await market_service.get_stock_price("NVDA")


async def test_rate_limit_is_mapped(patch_ticker):
    patch_ticker(FakeTicker(raises=YFRateLimitError()))
    with pytest.raises(RateLimitError, match="rate-limited"):
        await market_service.get_stock_price("NVDA")


async def test_rate_limit_is_detected_from_message_text(patch_ticker):
    """yfinance reports 429s inconsistently, so message matching is a needed fallback."""
    patch_ticker(FakeTicker(raises=RuntimeError("HTTP 429 Too Many Requests")))
    with pytest.raises(RateLimitError):
        await market_service.get_stock_price("NVDA")


async def test_empty_history_is_data_unavailable(patch_ticker):
    patch_ticker(FakeTicker(history=None))
    with pytest.raises(DataUnavailableError, match="No price history"):
        await market_service.get_historical_prices("NVDA", period="10y")


@pytest.mark.parametrize("bad", ["", "   ", "not a ticker", "TOOLONGTICKER", "123", "a@b"])
async def test_malformed_tickers_are_rejected_before_any_provider_call(bad, patch_ticker):
    patch_ticker(FakeTicker(raises=AssertionError("provider must not be called")))
    with pytest.raises(ValueError):
        await market_service.get_stock_price(bad)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [({"period": "7y"}, "Invalid period"), ({"interval": "1m"}, "Invalid interval")],
)
async def test_invalid_period_and_interval_are_rejected(kwargs, match, patch_ticker):
    patch_ticker(FakeTicker(raises=AssertionError("provider must not be called")))
    with pytest.raises(ValueError, match=match):
        await market_service.get_historical_prices("NVDA", **kwargs)


# ── Boundary: no vendor schema leaks past the service layer ───────────────────


async def test_vendor_field_names_never_reach_the_mcp_response(client, patch_ticker):
    patch_ticker(FakeTicker(info=NVDA_INFO))
    result = await client.call_tool("get_stock_price", {"ticker": "NVDA"})

    payload = result.structured_content
    assert set(payload) == {
        "ticker", "company_name", "price", "currency", "previous_close", "change",
        "change_percent", "day_high", "day_low", "volume", "market_state", "as_of", "source",
    }
    for vendor_key in ("regularMarketPrice", "longName", "trailingPE", "marketCap"):
        assert vendor_key not in payload


async def test_provider_exceptions_do_not_leak_stack_traces(client, patch_ticker):
    patch_ticker(FakeTicker(raises=RuntimeError("psycopg://user:hunter2@db internal detail")))
    with pytest.raises(ToolError) as excinfo:
        await client.call_tool("get_stock_price", {"ticker": "NVDA"})
    assert "Traceback" not in str(excinfo.value)


# ── MCP tool surface ──────────────────────────────────────────────────────────


async def test_price_tool_returns_structured_data_over_mcp(client, patch_ticker):
    patch_ticker(FakeTicker(info=NVDA_INFO))
    result = await client.call_tool("get_stock_price", {"ticker": "NVDA"})
    assert result.data.price == 120.5
    assert result.data.ticker == "NVDA"


async def test_profile_and_statistics_tools_over_mcp(client, patch_ticker):
    patch_ticker(FakeTicker(info=NVDA_INFO))
    profile = await client.call_tool("get_company_profile", {"ticker": "NVDA"})
    stats = await client.call_tool("get_stock_statistics", {"ticker": "NVDA"})
    cap = await client.call_tool("get_market_cap", {"ticker": "NVDA"})

    assert profile.data.industry == "Semiconductors"
    assert stats.data.pe_ratio_forward == 33.8
    assert cap.data.shares_outstanding == 24_600_000_000


async def test_history_tool_over_mcp(client, patch_ticker):
    patch_ticker(FakeTicker(history=make_history(days=10)))
    result = await client.call_tool(
        "get_historical_prices", {"ticker": "NVDA", "period": "1mo", "interval": "1d"}
    )
    assert result.data.point_count == 10


@pytest.mark.parametrize(
    ("bad", "match"),
    [
        ("123", "not a valid ticker"),          # caught by our domain validator
        ("N@T", "not a valid ticker"),
        ("WAY-TOO-LONG-SYMBOL", "at most 10"),  # caught earlier, by the MCP input schema
        ("", "at least 1"),
    ],
)
async def test_invalid_ticker_over_mcp_is_a_tool_error(client, patch_ticker, bad, match):
    """Malformed symbols are rejected either by the tool schema or by our validator."""
    patch_ticker(FakeTicker(raises=AssertionError("provider must not be called")))
    with pytest.raises(ToolError, match=match):
        await client.call_tool("get_stock_price", {"ticker": bad})


async def test_server_survives_a_failed_call(client, patch_ticker):
    patch_ticker(FakeTicker(raises=YFRateLimitError()))
    with pytest.raises(ToolError):
        await client.call_tool("get_stock_price", {"ticker": "NVDA"})

    patch_ticker(FakeTicker(info=NVDA_INFO))
    result = await client.call_tool("get_stock_price", {"ticker": "NVDA"})
    assert result.data.price == 120.5


async def test_market_server_runs_as_an_independent_process():
    transport = StdioTransport(command=sys.executable, args=["-m", "servers.market_mcp.server"])
    async with Client(transport) as client:
        assert {t.name for t in await client.list_tools()} == EXPECTED_TOOLS


# ── Optional live check ───────────────────────────────────────────────────────


@pytest.mark.integration
async def test_live_provider_returns_a_real_quote():
    """Opt-in only: `uv run pytest -m integration`. Requires network access."""
    quote = await market_service.get_stock_price("AAPL")
    assert quote.price > 0
    assert quote.ticker == "AAPL"


async def test_logs_never_contain_credentials(client, patch_ticker, monkeypatch, capsys):
    """Secrets must not reach the logs, even when a provider error mentions them."""
    monkeypatch.setenv("FMP_API_KEY", "sk-live-SUPERSECRET")
    patch_ticker(FakeTicker(raises=RuntimeError("auth failed for key sk-live-SUPERSECRET")))

    with pytest.raises(ToolError):
        await client.call_tool("get_stock_price", {"ticker": "NVDA"})

    captured = capsys.readouterr()
    assert "SUPERSECRET" not in captured.out + captured.err


@pytest.mark.parametrize(
    ("raw", "leaked"),
    [
        ("auth failed for key sk-live-SUPERSECRET", "SUPERSECRET"),
        ("GET https://api.x.com/v1?apikey=abc123XYZ", "abc123XYZ"),
        ("psycopg://user:hunter2@db/financial", "hunter2"),
        ("Authorization: Bearer eyJhbGciOiJIUzI1", "eyJhbGciOiJIUzI1"),
    ],
)
async def test_credentials_in_provider_errors_never_reach_the_agent(
    client, patch_ticker, raw, leaked
):
    """A vendor exception can echo the failing URL or header. It must be scrubbed."""
    patch_ticker(FakeTicker(raises=RuntimeError(raw)))
    with pytest.raises(ToolError) as excinfo:
        await client.call_tool("get_stock_price", {"ticker": "NVDA"})
    assert leaked not in str(excinfo.value)
    assert "REDACTED" in str(excinfo.value)


@pytest.mark.parametrize(
    ("raw", "leaked"),
    [
        ('{"password": "hunter2"}', "hunter2"),
        ('{"api_key":"abc123def"}', "abc123def"),
    ],
)
def test_scrubber_covers_json_shaped_credentials(raw, leaked):
    from services.redaction import scrub

    assert leaked not in scrub(raw)


def test_scrubber_leaves_ordinary_text_alone():
    from services.redaction import scrub

    text = "NVDA traded at 120.5 with revenue growth of 20% and a P/E of 51.2"
    assert scrub(text) == text
