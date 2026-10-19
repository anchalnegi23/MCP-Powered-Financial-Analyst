"""Phase 18: the six end-to-end scenarios, run against the real system.

Real graph, real MCP client, real FastMCP servers, real vector store, real FastAPI, real
PostgreSQL checkpointing. Only two things are substituted:

- the **LLM**, replaced by a scripted model, so the scenarios are deterministic and cost
  nothing. The model's *choices* are scripted; every tool it chooses actually executes.
- the **market data provider**, replaced by fixtures, so results do not depend on the
  market being open or on a rate limit.

What is genuinely verified is the thing under test: that a question routes to the right
MCP servers, that calculations are deterministic, that nothing is fabricated, and that
conversation context survives.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from langchain_core.messages import AIMessage

from agent.graph import FinancialAnalystAgent
from agent.mcp_client import MCPServerSpec, MCPToolkit
from api import dependencies
from api.main import app
from models.research import DocumentMetadata
from servers.calculator_mcp.server import mcp as calculator_server
from servers.financial_mcp.server import mcp as financial_server
from servers.market_mcp.server import mcp as market_server
from servers.news_mcp.server import mcp as news_server
from servers.research_mcp import server as research_module
from services.checkpointer import build_checkpointer
from services.database import Database
from services.research_service import ResearchStore, make_document_id
from tests.conftest import NVDA_FINANCIALS, NVDA_INFO, NVDA_NEWS_RAW, FakeTicker
from tests.fakes import REPORT_WITH_ALL_SECTIONS, ScriptedChatModel, tool_call

pytestmark = pytest.mark.asyncio(loop_scope="module")

TEST_DSN = (
    "postgresql://analyst:local-dev-only-not-a-real-secret@localhost:5433/financial_analyst"
)

AMD_INFO = {
    **NVDA_INFO,
    "symbol": "AMD",
    "longName": "Advanced Micro Devices, Inc.",
    "shortName": "AMD",
    "currentPrice": 162.4,
    "previousClose": 158.0,
    "regularMarketPreviousClose": 158.0,
    "marketCap": 262_000_000_000,
    "trailingPE": 46.1,
    "trailingEps": 3.52,
}


@pytest.fixture(scope="module")
def research_store():
    """A real vector store holding both sample 10-K extracts."""
    store = ResearchStore(in_memory=True)
    for path, company, ticker in (
        ("data/sample_documents/nvda_10k_2024.txt", "NVIDIA Corporation", "NVDA"),
        ("data/sample_documents/amd_10k_2024.txt", "Advanced Micro Devices, Inc.", "AMD"),
    ):
        store.index_document(
            Path(path).read_text(),
            DocumentMetadata(
                document_id=make_document_id(ticker, "10-K", 2024),
                company=company,
                ticker=ticker,
                document_type="10-K",
                year=2024,
                title=f"{company} Form 10-K 2024",
                source=path,
            ),
        )
    return store


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def toolkit(research_store):
    research_module.get_store.cache_clear()
    original = research_module.get_store
    research_module.get_store = lambda: research_store  # type: ignore[assignment]

    servers = [
        MCPServerSpec("calculator", "servers.calculator_mcp.server", "Maths.", calculator_server),
        MCPServerSpec("market", "servers.market_mcp.server", "Market.", market_server),
        MCPServerSpec(
            "financial", "servers.financial_mcp.server", "Statements.", financial_server
        ),
        MCPServerSpec("news", "servers.news_mcp.server", "News.", news_server),
        MCPServerSpec("research", "servers.research_mcp.server", "RAG.", research_module.mcp),
    ]
    async with MCPToolkit(servers) as tk:
        yield tk
    research_module.get_store = original  # type: ignore[assignment]


@pytest.fixture
def providers(patch_ticker):
    """Deterministic market, statement and news data for both companies."""
    payloads = {
        "NVDA": FakeTicker(info=NVDA_INFO, financials=NVDA_FINANCIALS, news=NVDA_NEWS_RAW),
        "AMD": FakeTicker(info=AMD_INFO, financials=NVDA_FINANCIALS, news=NVDA_NEWS_RAW),
    }
    patch_ticker(lambda symbol: payloads.get(symbol, payloads["NVDA"]))


def understanding(companies, restated, *, comparison=False, documents=False) -> AIMessage:
    return AIMessage(
        content=json.dumps(
            {
                "companies": companies,
                "intents": ["analysis"],
                "comparison": comparison,
                "needs_documents": documents,
                "restated_question": restated,
            }
        )
    )


PLAN = AIMessage(
    content=json.dumps({"steps": ["gather"], "data_needed": ["figures"], "calculations_needed": []})
)
NVDA_ONLY = [{"name": "NVIDIA", "ticker": "NVDA"}]
BOTH = [{"name": "NVIDIA", "ticker": "NVDA"}, {"name": "AMD", "ticker": "AMD"}]
TAIL = [
    AIMessage(content="Analysis of the retrieved data."),
    AIMessage(content="Risks identified from the evidence."),
    AIMessage(content=REPORT_WITH_ALL_SECTIONS),
]


# ── Scenario 1: current stock price -> Market MCP ─────────────────────────────


async def test_scenario_1_current_stock_price_uses_market_mcp(toolkit, providers):
    model = ScriptedChatModel(
        script=[
            understanding(NVDA_ONLY, "What is NVIDIA's current stock price?"),
            PLAN,
            tool_call("get_stock_price", {"ticker": "NVDA"}, "s1"),
            AIMessage(content="Retrieved the current price."),
            AIMessage(content="No calculations required."),
            *TAIL,
        ]
    )
    agent = FinancialAnalystAgent(model, toolkit)
    response = await agent.run("What's NVIDIA's current stock price?")

    assert response.mcp_servers_used == ["market"]
    assert response.tools_used == ["get_stock_price"]
    assert response.tool_calls[0].ok is True
    assert "120.5" in "\n".join(model.prompts_seen)  # the real figure reached the model


# ── Scenario 2: revenue growth comparison -> Financial + Calculator ───────────


async def test_scenario_2_revenue_growth_comparison_chains_financial_into_calculator(
    toolkit, providers
):
    model = ScriptedChatModel(
        script=[
            understanding(BOTH, "Compare NVIDIA and AMD revenue growth.", comparison=True),
            PLAN,
            tool_call("get_income_statement", {"ticker": "NVDA", "period": "annual"}, "d1"),
            tool_call("get_income_statement", {"ticker": "AMD", "period": "annual"}, "d2"),
            AIMessage(content="Retrieved both income statements."),
            tool_call(
                "calculate_revenue_growth",
                {"previous_revenue": 26_974_000_000, "current_revenue": 60_922_000_000},
                "c1",
            ),
            tool_call(
                "compare_financial_metrics",
                {
                    "metric": "revenue_growth",
                    "companies": [
                        {"company": "NVDA", "value": 125.85},
                        {"company": "AMD", "value": 13.69},
                    ],
                    "higher_is_better": True,
                },
                "c2",
            ),
            AIMessage(content="Computed and compared growth."),
            *TAIL,
        ]
    )
    agent = FinancialAnalystAgent(model, toolkit)
    response = await agent.run("Compare NVIDIA and AMD revenue growth.")

    assert set(response.mcp_servers_used) == {"financial", "calculator"}

    growth = next(m for m in response.metrics if m.metric == "revenue_growth")
    assert growth.value == 125.85  # deterministic, computed in Python
    assert growth.inputs["previous_revenue"] == 26_974_000_000
    assert all(call.ok for call in response.tool_calls)


async def test_scenario_2_calculations_are_deterministic_across_runs(toolkit, providers):
    """The same question twice must produce byte-identical numbers."""

    def run_once():
        model = ScriptedChatModel(
            script=[
                understanding(BOTH, "Compare growth.", comparison=True),
                PLAN,
                AIMessage(content="Collected."),
                tool_call(
                    "calculate_revenue_growth",
                    {"previous_revenue": 26_974_000_000, "current_revenue": 60_922_000_000},
                    "c1",
                ),
                AIMessage(content="Computed."),
                *TAIL,
            ]
        )
        return FinancialAnalystAgent(model, toolkit)

    first = await run_once().run("Compare NVIDIA and AMD revenue growth.")
    second = await run_once().run("Compare NVIDIA and AMD revenue growth.")

    assert [m.model_dump() for m in first.metrics] == [m.model_dump() for m in second.metrics]


# ── Scenario 3: recent developments -> News MCP ───────────────────────────────


async def test_scenario_3_recent_developments_uses_news_mcp(toolkit, providers):
    model = ScriptedChatModel(
        script=[
            understanding(NVDA_ONLY, "What are the latest developments affecting NVIDIA?"),
            PLAN,
            tool_call("get_latest_company_news", {"ticker": "NVDA", "limit": 5}, "n1"),
            AIMessage(content="Retrieved recent headlines."),
            AIMessage(content="No calculations required."),
            *TAIL,
        ]
    )
    agent = FinancialAnalystAgent(model, toolkit)
    response = await agent.run("What are the latest developments affecting NVIDIA?")

    assert response.mcp_servers_used == ["news"]
    assert response.sources  # citable URLs came back
    assert all(url.startswith("http") for url in response.sources)


# ── Scenario 4: full comparison -> four MCP servers in one workflow ───────────


async def test_scenario_4_full_comparison_uses_multiple_mcp_servers(toolkit, providers):
    model = ScriptedChatModel(
        script=[
            understanding(
                BOTH,
                "Compare NVIDIA and AMD on revenue growth, profitability, valuation and news.",
                comparison=True,
            ),
            PLAN,
            tool_call("get_income_statement", {"ticker": "NVDA", "period": "annual"}, "d1"),
            tool_call("get_stock_statistics", {"ticker": "NVDA"}, "d2"),
            tool_call("get_latest_company_news", {"ticker": "NVDA", "limit": 3}, "d3"),
            AIMessage(content="Gathered statements, valuation statistics and news."),
            tool_call(
                "calculate_revenue_growth",
                {"previous_revenue": 26_974_000_000, "current_revenue": 60_922_000_000},
                "c1",
            ),
            tool_call(
                "calculate_operating_margin",
                {"operating_income": 32_972_000_000, "revenue": 60_922_000_000},
                "c2",
            ),
            AIMessage(content="Computed growth and margin."),
            *TAIL,
        ]
    )
    agent = FinancialAnalystAgent(model, toolkit)
    response = await agent.run(
        "Compare NVIDIA and AMD based on revenue growth, profitability, valuation and "
        "recent news."
    )

    assert set(response.mcp_servers_used) == {"financial", "market", "news", "calculator"}
    assert len(response.metrics) == 2
    assert {m.metric for m in response.metrics} == {"revenue_growth", "operating_margin"}
    assert next(m for m in response.metrics if m.metric == "operating_margin").value == 54.12


# ── Scenario 5: annual report question -> Research MCP ────────────────────────


async def test_scenario_5_annual_report_question_uses_research_mcp(toolkit, providers):
    model = ScriptedChatModel(
        script=[
            understanding(
                NVDA_ONLY,
                "What major business risks does NVIDIA's annual report identify?",
                documents=True,
            ),
            PLAN,
            tool_call(
                "search_financial_documents",
                {
                    "query": "major business risks, supply concentration, export controls",
                    "ticker": "NVDA",
                    "limit": 3,
                },
                "r1",
            ),
            AIMessage(content="Retrieved risk-factor passages from the 10-K."),
            AIMessage(content="No calculations required."),
            *TAIL,
        ]
    )
    agent = FinancialAnalystAgent(model, toolkit)
    response = await agent.run(
        "According to NVIDIA's annual report, what are its major business risks?"
    )

    assert response.mcp_servers_used == ["research"]

    retrieved = "\n".join(model.prompts_seen)
    assert "foundry" in retrieved.lower() or "concentration" in retrieved.lower()
    assert '"document_type": "10-K"' in retrieved  # citable provenance reached the model
    assert '"ticker": "NVDA"' in retrieved


# ── Scenario 6: follow-up -> conversation context ─────────────────────────────


async def test_scenario_6_follow_up_remembers_the_companies(toolkit, providers):
    """'Which one has the better profit margin?' needs turn one to have an antecedent."""
    handle = await build_checkpointer(TEST_DSN, connect_timeout=2)
    thread = f"e2e-{uuid.uuid4().hex[:10]}"
    try:
        first = ScriptedChatModel(
            script=[
                understanding(BOTH, "Compare NVIDIA and AMD.", comparison=True),
                PLAN,
                tool_call("get_income_statement", {"ticker": "NVDA", "period": "annual"}, "d1"),
                AIMessage(content="Retrieved NVIDIA's income statement."),
                AIMessage(content="No calculations yet."),
                *TAIL,
            ]
        )
        await FinancialAnalystAgent(first, toolkit, checkpointer=handle.saver).run(
            "Compare NVIDIA and AMD.", conversation_id=thread
        )

        # A brand-new model and agent: anything they know came out of the checkpoint.
        second = ScriptedChatModel(
            script=[
                understanding(
                    BOTH,
                    "Which of NVIDIA and AMD has the better profit margin?",
                    comparison=True,
                ),
                PLAN,
                AIMessage(content="Using the statements already retrieved."),
                tool_call(
                    "calculate_profit_margin",
                    {"net_income": 29_760_000_000, "revenue": 60_922_000_000},
                    "c1",
                ),
                AIMessage(content="Computed the margin."),
                *TAIL,
            ]
        )
        response = await FinancialAnalystAgent(
            second, toolkit, checkpointer=handle.saver
        ).run("Which one has the better profit margin?", conversation_id=thread)

        first_prompt = second.prompts_seen[0]
        assert "Compare NVIDIA and AMD." in first_prompt   # turn one survived
        assert "Which one has the better profit margin?" in first_prompt
        assert [c.ticker for c in response.companies] == ["NVDA", "AMD"]
        assert next(m for m in response.metrics if m.metric == "profit_margin").value == 48.85
    finally:
        await handle.close()


# ── Cross-cutting guarantees ──────────────────────────────────────────────────


async def test_no_data_is_fabricated_when_every_tool_fails(toolkit, patch_ticker):
    patch_ticker(FakeTicker(raises=TimeoutError("provider unavailable")))
    model = ScriptedChatModel(
        script=[
            understanding(NVDA_ONLY, "What is NVIDIA's revenue growth?"),
            PLAN,
            tool_call("get_income_statement", {"ticker": "NVDA", "period": "annual"}, "d1"),
            AIMessage(content="The statement could not be retrieved."),
            AIMessage(content="No calculations are possible."),
            *TAIL,
        ]
    )
    agent = FinancialAnalystAgent(model, toolkit)
    response = await agent.run("What is NVIDIA's revenue growth?")

    assert response.mcp_servers_used == []  # nothing succeeded, so nothing is claimed
    assert response.metrics == []
    assert any("failed" in w for w in response.warnings)
    assert "Do not invent" in "\n".join(model.prompts_seen)


async def test_tool_failures_do_not_stop_the_rest_of_the_workflow(toolkit, patch_ticker):
    """One dead provider must degrade the report, not void it."""
    calls = {"n": 0}

    def factory(_symbol):
        calls["n"] += 1
        if calls["n"] == 1:
            return FakeTicker(raises=TimeoutError("provider unavailable"))
        return FakeTicker(info=NVDA_INFO, financials=NVDA_FINANCIALS)

    patch_ticker(factory)
    model = ScriptedChatModel(
        script=[
            understanding(NVDA_ONLY, "Analyse NVIDIA."),
            PLAN,
            tool_call("get_stock_price", {"ticker": "NVDA"}, "d1"),
            tool_call("get_income_statement", {"ticker": "NVDA", "period": "annual"}, "d2"),
            AIMessage(content="Price failed; statements retrieved."),
            tool_call(
                "calculate_revenue_growth",
                {"previous_revenue": 26_974_000_000, "current_revenue": 60_922_000_000},
                "c1",
            ),
            AIMessage(content="Computed growth."),
            *TAIL,
        ]
    )
    agent = FinancialAnalystAgent(model, toolkit)
    response = await agent.run("Analyse NVIDIA.")

    assert any(not c.ok for c in response.tool_calls)
    assert any(c.ok for c in response.tool_calls)
    assert response.metrics  # the analysis still produced a real, sourced number
    assert response.answer


async def test_every_report_is_structured_and_understandable(toolkit, providers):
    from agent.state import REPORT_SECTIONS

    model = ScriptedChatModel(script=[understanding(NVDA_ONLY, "Analyse NVIDIA."), PLAN, *TAIL])
    agent = FinancialAnalystAgent(model, toolkit)
    response = await agent.run("Analyse NVIDIA.")

    for section in REPORT_SECTIONS:
        assert f"## {section}" in response.answer
    assert "not investment advice" in response.answer.lower()


# ── Through the HTTP API ──────────────────────────────────────────────────────


@pytest_asyncio.fixture(loop_scope="module")
async def api_client(toolkit):
    model = ScriptedChatModel(
        script=[
            understanding(BOTH, "Compare NVIDIA and AMD.", comparison=True),
            PLAN,
            tool_call("get_income_statement", {"ticker": "NVDA", "period": "annual"}, "d1"),
            tool_call("get_latest_company_news", {"ticker": "NVDA", "limit": 2}, "d2"),
            AIMessage(content="Collected."),
            tool_call(
                "calculate_revenue_growth",
                {"previous_revenue": 26_974_000_000, "current_revenue": 60_922_000_000},
                "c1",
            ),
            AIMessage(content="Computed."),
            *TAIL,
        ]
    )
    database = Database(TEST_DSN, connect_timeout=2)
    await database.connect()

    dependencies.state.toolkit = toolkit
    dependencies.state.agent = FinancialAnalystAgent(model, toolkit)
    dependencies.state.agent_error = None
    dependencies.state.db = database

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    await database.close()


async def test_the_complete_workflow_runs_through_the_http_api(api_client, providers):
    response = await api_client.post(
        "/chat",
        json={"query": "Compare NVIDIA and AMD on revenue growth and recent news."},
    )
    assert response.status_code == 200
    body = response.json()

    assert "## Executive Summary" in body["answer"]
    assert "## Bull Case" in body["answer"] and "## Bear Case" in body["answer"]
    assert set(body["mcp_servers_used"]) == {"financial", "news", "calculator"}
    assert body["metrics"][0]["value"] == 125.85
    assert body["metrics"][0]["formula"]
    assert body["sources"]
    assert body["request_id"] and body["conversation_id"]


async def test_all_five_servers_are_exposed_through_the_api(api_client):
    body = (await api_client.get("/tools")).json()

    assert set(body["servers"]) == {"calculator", "market", "financial", "news", "research"}
    assert body["total"] == 29
    assert body["unavailable_servers"] == {}
