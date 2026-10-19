"""MCP client: multi-server connection, discovery, invocation and failure handling.

These are integration tests in the honest sense — they start the real FastMCP servers as
subprocesses and speak the real protocol — but they touch no external API. The market and
financial servers are exercised only for discovery, not for live data.
"""

from __future__ import annotations

import json

import pytest
import pytest_asyncio

from agent.mcp_client import MCPServerSpec, MCPToolkit

CALCULATOR = MCPServerSpec(
    name="calculator", module="servers.calculator_mcp.server", description="Maths."
)
MARKET = MCPServerSpec(
    name="market", module="servers.market_mcp.server", description="Market data."
)
FINANCIAL = MCPServerSpec(
    name="financial", module="servers.financial_mcp.server", description="Statements."
)
NEWS = MCPServerSpec(name="news", module="servers.news_mcp.server", description="News.")

# Every test in this module shares one event loop so the module-scoped toolkit can hold
# its subprocess servers open across tests. Other test modules keep the default
# per-test loop.
pytestmark = pytest.mark.asyncio(loop_scope="module")

MISSING = MCPServerSpec(
    name="ghost", module="servers.does_not_exist.server", description="Never starts."
)


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def toolkit():
    """One toolkit across the module: starting four subprocesses per test is wasteful.

    The loop scope must match the fixture scope, or the servers would be started on an
    event loop that closes after the first test.
    """
    async with MCPToolkit([CALCULATOR, MARKET, FINANCIAL, NEWS]) as tk:
        yield tk


# ── Connection ────────────────────────────────────────────────────────────────


async def test_connects_to_the_calculator_server():
    async with MCPToolkit([CALCULATOR]) as tk:
        assert tk.statuses["calculator"].connected is True


async def test_connects_to_the_market_server():
    async with MCPToolkit([MARKET]) as tk:
        assert tk.statuses["market"].connected is True


async def test_connects_to_multiple_servers_at_once(toolkit):
    assert set(toolkit.connected_servers) == {"calculator", "market", "financial", "news"}
    assert toolkit.failed_servers == {}


# ── Discovery ─────────────────────────────────────────────────────────────────


async def test_discovers_calculator_tools(toolkit):
    names = set(toolkit.statuses["calculator"].tool_names)
    assert "calculate_revenue_growth" in names
    assert len(names) == 10


async def test_discovers_tools_from_every_server(toolkit):
    discovered = {s["server"] for s in toolkit.tool_schemas()}
    assert discovered == {"calculator", "market", "financial", "news"}
    assert len(toolkit.tool_schemas()) == 10 + 5 + 5 + 3


async def test_tool_schemas_are_preserved_verbatim(toolkit):
    """The model must see the server's own parameter names, types and descriptions."""
    schema = next(
        s for s in toolkit.tool_schemas() if s["name"] == "calculate_revenue_growth"
    )
    properties = schema["input_schema"]["properties"]

    assert set(properties) == {"previous_revenue", "current_revenue"}
    assert properties["previous_revenue"]["type"] == "number"
    assert "greater than zero" in properties["previous_revenue"]["description"]
    assert set(schema["input_schema"]["required"]) == {"previous_revenue", "current_revenue"}
    assert schema["output_schema"] is not None
    assert "Use this" in schema["description"]


async def test_can_locate_which_server_owns_a_tool(toolkit):
    assert toolkit.server_for_tool("get_stock_price") == "market"
    assert toolkit.server_for_tool("calculate_roe") == "calculator"
    assert toolkit.server_for_tool("no_such_tool") is None


# ── Invocation ────────────────────────────────────────────────────────────────


async def test_invokes_a_calculator_tool(toolkit):
    invocation = await toolkit.call_tool(
        "calculator", "calculate_revenue_growth", {"previous_revenue": 100, "current_revenue": 120}
    )
    assert invocation.ok is True
    assert invocation.result["value"] == 20.0
    assert invocation.duration_ms >= 0


async def test_invocation_returns_structured_data_not_prose(toolkit):
    invocation = await toolkit.call_tool(
        "calculator", "calculate_pe_ratio", {"price_per_share": 120, "earnings_per_share": 4}
    )
    assert isinstance(invocation.result, dict)
    assert invocation.result["unit"] == "times"
    assert invocation.result["formula"] == "price_per_share / earnings_per_share"


async def test_every_invocation_is_recorded_for_tracing(toolkit):
    before = len(toolkit.invocations)
    await toolkit.call_tool("calculator", "calculate_roa", {"net_income": 15, "total_assets": 300})
    assert len(toolkit.invocations) == before + 1
    assert toolkit.invocations[-1].tool == "calculate_roa"


# ── Failure handling ──────────────────────────────────────────────────────────


async def test_a_server_that_cannot_start_is_recorded_not_raised():
    """One dead server must not take the session down."""
    async with MCPToolkit([CALCULATOR, MISSING]) as tk:
        assert tk.connected_servers == ["calculator"]
        assert "ghost" in tk.failed_servers
        assert tk.statuses["ghost"].connected is False


async def test_tools_from_surviving_servers_still_work_after_a_failure():
    async with MCPToolkit([MISSING, CALCULATOR]) as tk:
        invocation = await tk.call_tool(
            "calculator", "calculate_profit_margin", {"net_income": 25, "revenue": 100}
        )
        assert invocation.ok is True
        assert invocation.result["value"] == 25.0


async def test_calling_a_disconnected_server_is_a_recorded_failure():
    async with MCPToolkit([CALCULATOR]) as tk:
        invocation = await tk.call_tool("news", "get_latest_company_news", {"ticker": "NVDA"})
        assert invocation.ok is False
        assert "not connected" in invocation.error


async def test_tool_execution_failure_is_captured_not_raised(toolkit):
    invocation = await toolkit.call_tool(
        "calculator", "calculate_profit_margin", {"net_income": 25, "revenue": 0}
    )
    assert invocation.ok is False
    assert "undefined" in invocation.error
    assert invocation.result is None


async def test_unknown_tool_is_a_recorded_failure(toolkit):
    invocation = await toolkit.call_tool("calculator", "calculate_vibes", {})
    assert invocation.ok is False


async def test_toolkit_survives_a_failed_call(toolkit):
    await toolkit.call_tool(
        "calculator", "calculate_roe", {"net_income": 1, "shareholders_equity": 0}
    )
    invocation = await toolkit.call_tool(
        "calculator", "calculate_roe", {"net_income": 20, "shareholders_equity": 100}
    )
    assert invocation.ok is True


# ── LangChain adaptation ──────────────────────────────────────────────────────


async def test_mcp_tools_become_langchain_tools(toolkit):
    tools = toolkit.langchain_tools()
    assert len(tools) == 23
    by_name = {t.name: t for t in tools}
    assert "calculate_revenue_growth" in by_name
    assert "get_stock_price" in by_name
    assert "get_income_statement" in by_name
    assert "get_latest_company_news" in by_name


async def test_langchain_tool_carries_its_server_identity(toolkit):
    tool = next(t for t in toolkit.langchain_tools() if t.name == "get_stock_price")
    assert tool.metadata["mcp_server"] == "market"
    assert tool.description.startswith("[market MCP]")


async def test_langchain_tool_preserves_the_mcp_argument_schema(toolkit):
    tool = next(t for t in toolkit.langchain_tools() if t.name == "calculate_cagr")
    properties = tool.tool_call_schema["properties"]
    assert set(properties) >= {"beginning_value", "ending_value", "years"}
    assert "fractional values such as 2.5" in properties["years"]["description"]


async def test_langchain_tool_invocation_returns_json(toolkit):
    tool = next(t for t in toolkit.langchain_tools() if t.name == "calculate_gross_margin")
    payload = json.loads(await tool.ainvoke({"gross_profit": 75, "revenue": 100}))
    assert payload["value"] == 75.0


async def test_langchain_tool_failure_is_returned_to_the_model_with_guidance(toolkit):
    """A failure the model can read beats an exception that aborts the graph."""
    tool = next(t for t in toolkit.langchain_tools() if t.name == "calculate_gross_margin")
    payload = json.loads(await tool.ainvoke({"gross_profit": 75, "revenue": 0}))

    assert "error" in payload
    assert "undefined" in payload["error"]
    assert "Do not invent" in payload["guidance"]
