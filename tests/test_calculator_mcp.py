"""MCP-level tests for the Calculator server.

These exercise the real MCP protocol surface — tool discovery, schemas, invocation and
error propagation — rather than calling the Python functions directly.
"""

import sys

import pytest
from fastmcp import Client
from fastmcp.client.transports import StdioTransport
from fastmcp.exceptions import ToolError

from servers.calculator_mcp.server import mcp

EXPECTED_TOOLS = {
    "calculate_revenue_growth",
    "calculate_profit_margin",
    "calculate_gross_margin",
    "calculate_operating_margin",
    "calculate_pe_ratio",
    "calculate_debt_to_equity",
    "calculate_roe",
    "calculate_roa",
    "calculate_cagr",
    "compare_financial_metrics",
}


@pytest.fixture
async def client():
    """In-memory client speaking the MCP protocol to the server object."""
    async with Client(mcp) as c:
        yield c


# ── Discovery ─────────────────────────────────────────────────────────────────


async def test_all_ten_tools_are_discoverable(client):
    tools = await client.list_tools()
    assert {t.name for t in tools} == EXPECTED_TOOLS


async def test_every_tool_description_is_useful_to_an_llm(client):
    """A description must say what it returns and what can go wrong, not just what it is."""
    for tool in await client.list_tools():
        description = tool.description or ""
        assert len(description) > 200, f"{tool.name} description is too thin"
        assert "Use this" in description, f"{tool.name} does not say when to use it"
        assert "Returns" in description, f"{tool.name} does not document its return value"
        assert "Errors" in description, f"{tool.name} does not document its failure modes"


async def test_input_schemas_document_every_parameter(client):
    for tool in await client.list_tools():
        properties = tool.input_schema["properties"]
        assert properties, f"{tool.name} exposes no parameters"
        for name, spec in properties.items():
            assert spec.get("description"), f"{tool.name}.{name} has no description"


async def test_tools_declare_structured_output_schemas(client):
    for tool in await client.list_tools():
        assert tool.output_schema, f"{tool.name} has no output schema"


# ── Invocation ────────────────────────────────────────────────────────────────


async def test_revenue_growth_over_mcp_matches_worked_example(client):
    result = await client.call_tool(
        "calculate_revenue_growth", {"previous_revenue": 100, "current_revenue": 120}
    )
    assert result.data.value == 20.0
    assert result.data.unit == "percent"


async def test_pe_ratio_over_mcp(client):
    result = await client.call_tool(
        "calculate_pe_ratio", {"price_per_share": 120, "earnings_per_share": 4}
    )
    assert result.data.value == 30.0
    assert result.data.unit == "times"


async def test_comparison_over_mcp_accepts_nested_models(client):
    result = await client.call_tool(
        "compare_financial_metrics",
        {
            "metric": "operating_margin",
            "companies": [
                {"company": "NVDA", "value": 54.1},
                {"company": "AMD", "value": 7.4},
            ],
            "higher_is_better": True,
        },
    )
    assert result.data.best == "NVDA"
    assert [r.rank for r in result.data.ranked] == [1, 2]


async def test_tool_results_are_structured_not_prose(client):
    """The agent receives typed fields it can index, not a sentence to re-parse."""
    result = await client.call_tool(
        "calculate_profit_margin", {"net_income": 25, "revenue": 100}
    )
    assert set(result.structured_content) >= {
        "metric",
        "label",
        "value",
        "unit",
        "formula",
        "inputs",
        "interpretation",
        "warnings",
    }
    assert result.structured_content["value"] == 25.0


# ── Failure handling ──────────────────────────────────────────────────────────


async def test_division_by_zero_returns_a_tool_error(client):
    with pytest.raises(ToolError, match="undefined"):
        await client.call_tool("calculate_profit_margin", {"net_income": 25, "revenue": 0})


async def test_missing_argument_is_rejected_by_the_schema(client):
    with pytest.raises(ToolError):
        await client.call_tool("calculate_profit_margin", {"net_income": 25})


async def test_wrong_type_is_rejected_by_the_schema(client):
    with pytest.raises(ToolError):
        await client.call_tool(
            "calculate_profit_margin", {"net_income": "twenty-five", "revenue": 100}
        )


async def test_unknown_tool_is_rejected(client):
    with pytest.raises(ToolError):
        await client.call_tool("calculate_alpha", {})


async def test_server_survives_a_failed_call_and_answers_the_next_one(client):
    """A bad call must not poison the session: the server keeps serving."""
    with pytest.raises(ToolError):
        await client.call_tool("calculate_roe", {"net_income": 20, "shareholders_equity": 0})

    result = await client.call_tool("calculate_roe", {"net_income": 20, "shareholders_equity": 100})
    assert result.data.value == 20.0


# ── Standalone process ────────────────────────────────────────────────────────


async def test_server_runs_as_an_independent_process():
    """Start the server as a real subprocess over stdio, with nothing else running.

    This is the proof that the Calculator MCP has no dependency on the agent, the API
    or any external service.
    """
    transport = StdioTransport(
        command=sys.executable, args=["-m", "servers.calculator_mcp.server"]
    )
    async with Client(transport) as client:
        assert {t.name for t in await client.list_tools()} == EXPECTED_TOOLS

        growth = await client.call_tool(
            "calculate_revenue_growth", {"previous_revenue": 100, "current_revenue": 120}
        )
        margin = await client.call_tool(
            "calculate_gross_margin", {"gross_profit": 75, "revenue": 100}
        )
        cagr = await client.call_tool(
            "calculate_cagr", {"beginning_value": 100, "ending_value": 200, "years": 2}
        )

    assert growth.data.value == 20.0
    assert margin.data.value == 75.0
    assert cagr.data.value == 41.42


ROUND_TRIP_CASES = [
    ("calculate_revenue_growth", {"previous_revenue": 100, "current_revenue": 120}, 20.0),
    ("calculate_profit_margin", {"net_income": 25, "revenue": 100}, 25.0),
    ("calculate_gross_margin", {"gross_profit": 75, "revenue": 100}, 75.0),
    ("calculate_operating_margin", {"operating_income": 33, "revenue": 100}, 33.0),
    ("calculate_pe_ratio", {"price_per_share": 120, "earnings_per_share": 4}, 30.0),
    ("calculate_debt_to_equity", {"total_debt": 50, "total_equity": 200}, 0.25),
    ("calculate_roe", {"net_income": 20, "shareholders_equity": 100}, 20.0),
    ("calculate_roa", {"net_income": 15, "total_assets": 300}, 5.0),
    ("calculate_cagr", {"beginning_value": 100, "ending_value": 200, "years": 2}, 41.42),
]


@pytest.mark.parametrize(("tool", "args", "expected"), ROUND_TRIP_CASES)
async def test_every_metric_tool_round_trips_over_mcp(client, tool, args, expected):
    """Every tool is reachable over the protocol and returns the same value as pure Python."""
    result = await client.call_tool(tool, args)
    assert result.data.value == expected
    assert result.data.inputs == args
