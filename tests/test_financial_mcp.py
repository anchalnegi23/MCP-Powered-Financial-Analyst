"""Financial Statements MCP: normalisation, missing-data policy and the tool surface."""

from __future__ import annotations

import sys
from datetime import date

import pytest
from fastmcp import Client
from fastmcp.client.transports import StdioTransport
from fastmcp.exceptions import ToolError

from servers.financial_mcp.server import mcp
from services import financial_service
from services.errors import DataUnavailableError, ProviderTimeoutError
from tests.conftest import (
    INCOMPLETE_BALANCE,
    NVDA_FINANCIALS,
    NVDA_INFO,
    FakeTicker,
)

EXPECTED_TOOLS = {
    "get_income_statement",
    "get_balance_sheet",
    "get_cash_flow",
    "get_financial_ratios",
    "get_earnings_data",
}


@pytest.fixture
async def client():
    async with Client(mcp) as c:
        yield c


@pytest.fixture
def nvda(patch_ticker):
    patch_ticker(FakeTicker(info=NVDA_INFO, financials=NVDA_FINANCIALS))


# ── Discovery ─────────────────────────────────────────────────────────────────


async def test_all_statement_tools_are_discoverable(client):
    assert {t.name for t in await client.list_tools()} == EXPECTED_TOOLS


async def test_every_tool_documents_usage_returns_errors_and_limits(client):
    for tool in await client.list_tools():
        description = tool.description or ""
        assert "Use this" in description
        assert "Returns" in description
        assert "Errors" in description
        assert "Limitation" in description
        assert tool.output_schema


# ── Income statement ──────────────────────────────────────────────────────────


async def test_income_statement_is_normalised(nvda):
    statement = await financial_service.get_income_statement("NVDA")

    assert statement.statement_type == "income_statement"
    assert statement.period == "annual"
    assert len(statement.periods) == 3

    latest = statement.periods[0]
    assert latest.period_ending == date(2024, 1, 28)
    assert latest.fiscal_label == "FY2024"
    assert latest.line_items["revenue"] == 60_922_000_000
    assert latest.line_items["net_income"] == 29_760_000_000
    assert latest.line_items["eps_diluted"] == 11.93


async def test_periods_are_ordered_most_recent_first(nvda):
    statement = await financial_service.get_income_statement("NVDA")
    endings = [p.period_ending for p in statement.periods]
    assert endings == sorted(endings, reverse=True)


async def test_provider_row_labels_do_not_leak_into_the_response(nvda):
    """'Total Revenue' is a vendor label; the domain vocabulary is 'revenue'."""
    statement = await financial_service.get_income_statement("NVDA")
    keys = set(statement.periods[0].line_items)
    assert "revenue" in keys
    assert "Total Revenue" not in keys
    assert not any(" " in key for key in keys)


# ── Balance sheet and cash flow ───────────────────────────────────────────────


async def test_balance_sheet_is_normalised(nvda):
    statement = await financial_service.get_balance_sheet("NVDA")
    latest = statement.periods[0]
    assert latest.line_items["total_assets"] == 65_728_000_000
    assert latest.line_items["total_equity"] == 42_978_000_000


async def test_cash_flow_is_normalised(nvda):
    statement = await financial_service.get_cash_flow("NVDA")
    latest = statement.periods[0]
    assert latest.line_items["operating_cash_flow"] == 28_090_000_000
    assert latest.line_items["capital_expenditure"] == -1_069_000_000


# ── Missing data: unknown, never zero ─────────────────────────────────────────


async def test_missing_line_items_are_named_not_defaulted_to_zero(patch_ticker):
    patch_ticker(
        FakeTicker(info=NVDA_INFO, financials={"balance_sheet": INCOMPLETE_BALANCE})
    )
    statement = await financial_service.get_balance_sheet("SPRS")

    latest = statement.periods[0]
    assert "total_equity" not in latest.line_items  # absent, not 0.0
    assert "total_equity" in latest.missing_items
    assert statement.notes and "not zero" in statement.notes[0]


async def test_ratios_that_cannot_be_computed_are_null_and_named(patch_ticker):
    patch_ticker(
        FakeTicker(
            info=NVDA_INFO,
            financials={
                "income_stmt": NVDA_FINANCIALS["income_stmt"],
                "balance_sheet": INCOMPLETE_BALANCE,
            },
        )
    )
    ratios = await financial_service.get_financial_ratios("NVDA")

    assert ratios.debt_to_equity is None
    assert "debt_to_equity" in ratios.missing_ratios
    assert "return_on_equity_percent" in ratios.missing_ratios
    assert ratios.gross_margin_percent is not None  # income-only ratios still work


# ── Ratios ────────────────────────────────────────────────────────────────────


async def test_ratios_are_computed_from_reported_line_items(nvda):
    ratios = await financial_service.get_financial_ratios("NVDA")

    assert ratios.gross_margin_percent == pytest.approx(72.72, abs=0.01)
    assert ratios.operating_margin_percent == pytest.approx(54.12, abs=0.01)
    assert ratios.net_margin_percent == pytest.approx(48.85, abs=0.01)
    assert ratios.return_on_equity_percent == pytest.approx(69.24, abs=0.01)
    assert ratios.debt_to_equity == pytest.approx(0.2573, abs=0.001)
    assert ratios.current_ratio == pytest.approx(4.1713, abs=0.001)
    assert ratios.missing_ratios == []


async def test_ratio_inputs_are_traceable(nvda):
    ratios = await financial_service.get_financial_ratios("NVDA")
    assert ratios.inputs_used["revenue"] == 60_922_000_000
    assert ratios.inputs_used["total_equity"] == 42_978_000_000


async def test_ratios_align_income_and_balance_to_the_same_period_end(nvda):
    """Mixing FY2024 income with FY2023 equity would silently corrupt ROE."""
    ratios = await financial_service.get_financial_ratios("NVDA")
    assert ratios.period_ending == date(2024, 1, 28)


# ── Earnings ──────────────────────────────────────────────────────────────────


async def test_earnings_series_is_built_from_the_income_statement(nvda):
    earnings = await financial_service.get_earnings_data("NVDA")

    assert [p.fiscal_label for p in earnings.periods] == ["FY2024", "FY2023", "FY2022"]
    assert earnings.periods[0].revenue == 60_922_000_000
    assert earnings.periods[0].eps_actual == 11.93


async def test_unavailable_estimates_are_disclosed_not_invented(nvda):
    earnings = await financial_service.get_earnings_data("NVDA")
    assert earnings.periods[0].eps_estimate is None
    assert earnings.periods[0].surprise_percent is None
    assert any("not available" in note for note in earnings.notes)


# ── Period validation ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("supplied", "expected"),
    [("annual", "annual"), ("ANNUAL", "annual"), ("quarterly", "quarterly"), ("Q", "quarterly")],
)
def test_valid_periods_are_accepted(supplied, expected):
    assert financial_service.validate_period(supplied) == expected


@pytest.mark.parametrize("bad", ["monthly", "ttm", "", "2024", "yearly-ish"])
def test_invalid_periods_are_rejected(bad):
    with pytest.raises(ValueError, match="Invalid period"):
        financial_service.validate_period(bad)


async def test_quarterly_uses_the_quarterly_provider_frame(nvda):
    statement = await financial_service.get_income_statement("NVDA", period="quarterly")
    assert statement.period == "quarterly"
    assert statement.periods[0].fiscal_label.startswith("Q")


# ── Failure handling ──────────────────────────────────────────────────────────


async def test_invalid_ticker_is_handled(patch_ticker):
    patch_ticker(FakeTicker(info={}, financials={}))
    with pytest.raises(DataUnavailableError):
        await financial_service.get_income_statement("ZZZZ")


async def test_malformed_ticker_is_rejected_before_the_provider(patch_ticker):
    patch_ticker(FakeTicker(raises=AssertionError("provider must not be called")))
    with pytest.raises(ValueError, match="not a valid ticker"):
        await financial_service.get_income_statement("not a ticker")


async def test_provider_timeout_becomes_a_meaningful_error(patch_ticker):
    patch_ticker(FakeTicker(raises=TimeoutError("timed out")))
    with pytest.raises(ProviderTimeoutError):
        await financial_service.get_income_statement("NVDA")


async def test_empty_statement_frame_is_data_unavailable(patch_ticker):
    patch_ticker(FakeTicker(info=NVDA_INFO, financials={}))
    with pytest.raises(DataUnavailableError, match="No annual income statement"):
        await financial_service.get_income_statement("NVDA")


async def test_company_name_lookup_failure_does_not_fail_the_statement(patch_ticker):
    """A descriptive lookup is best-effort; the statement is the actual deliverable."""

    class NameFails(FakeTicker):
        @property
        def info(self):
            raise RuntimeError("profile endpoint down")

    patch_ticker(NameFails(financials=NVDA_FINANCIALS))
    statement = await financial_service.get_income_statement("NVDA")
    assert statement.company_name is None
    assert statement.periods[0].line_items["revenue"] == 60_922_000_000


async def test_unknown_statement_type_is_rejected():
    with pytest.raises(ValueError, match="Unknown statement type"):
        await financial_service.get_statement("NVDA", "equity_statement")


# ── MCP tool surface ──────────────────────────────────────────────────────────


async def test_statement_tools_over_mcp(client, nvda):
    income = await client.call_tool("get_income_statement", {"ticker": "NVDA"})
    balance = await client.call_tool("get_balance_sheet", {"ticker": "NVDA"})
    cash = await client.call_tool("get_cash_flow", {"ticker": "NVDA"})

    assert income.data.periods[0].line_items["revenue"] == 60_922_000_000
    assert balance.data.periods[0].line_items["total_assets"] == 65_728_000_000
    assert cash.data.periods[0].line_items["free_cash_flow"] == 27_021_000_000


async def test_ratios_and_earnings_tools_over_mcp(client, nvda):
    ratios = await client.call_tool("get_financial_ratios", {"ticker": "NVDA"})
    earnings = await client.call_tool("get_earnings_data", {"ticker": "NVDA"})

    assert ratios.data.net_margin_percent == pytest.approx(48.85, abs=0.01)
    assert len(earnings.data.periods) == 3


async def test_invalid_period_over_mcp_is_a_tool_error(client, nvda):
    with pytest.raises(ToolError, match="Invalid period"):
        await client.call_tool("get_income_statement", {"ticker": "NVDA", "period": "monthly"})


async def test_server_survives_a_failed_call(client, patch_ticker):
    patch_ticker(FakeTicker(raises=TimeoutError("timed out")))
    with pytest.raises(ToolError):
        await client.call_tool("get_income_statement", {"ticker": "NVDA"})

    patch_ticker(FakeTicker(info=NVDA_INFO, financials=NVDA_FINANCIALS))
    result = await client.call_tool("get_income_statement", {"ticker": "NVDA"})
    assert result.data.periods[0].line_items["revenue"] == 60_922_000_000


async def test_financial_server_runs_as_an_independent_process():
    transport = StdioTransport(command=sys.executable, args=["-m", "servers.financial_mcp.server"])
    async with Client(transport) as client:
        assert {t.name for t in await client.list_tools()} == EXPECTED_TOOLS


@pytest.mark.integration
async def test_live_provider_returns_a_real_income_statement():
    statement = await financial_service.get_income_statement("AAPL")
    assert statement.periods[0].line_items["revenue"] > 0
