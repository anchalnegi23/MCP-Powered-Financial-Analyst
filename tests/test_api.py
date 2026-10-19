"""FastAPI: contract, error mapping and the agent integration.

The app is started for real (lifespan included) with in-process MCP servers and a
scripted model, so these exercise the actual routes end to end without an API key.
"""

from __future__ import annotations

import json
import os
import uuid

import httpx
import pytest
import pytest_asyncio
from langchain_core.messages import AIMessage

from agent.graph import FinancialAnalystAgent
from agent.mcp_client import MCPServerSpec, MCPToolkit
from api import dependencies
from api.main import app
from servers.calculator_mcp.server import mcp as calculator_server
from servers.financial_mcp.server import mcp as financial_server
from servers.market_mcp.server import mcp as market_server
from servers.news_mcp.server import mcp as news_server
from services.database import Database
from tests.fakes import REPORT_WITH_ALL_SECTIONS, ScriptedChatModel, tool_call

pytestmark = pytest.mark.asyncio(loop_scope="module")

SERVERS = [
    MCPServerSpec("calculator", "servers.calculator_mcp.server", "Maths.", calculator_server),
    MCPServerSpec("market", "servers.market_mcp.server", "Market.", market_server),
    MCPServerSpec("financial", "servers.financial_mcp.server", "Statements.", financial_server),
    MCPServerSpec("news", "servers.news_mcp.server", "News.", news_server),
]

UNDERSTANDING = AIMessage(
    content=json.dumps(
        {
            "companies": [{"name": "NVIDIA", "ticker": "NVDA"}],
            "intents": ["revenue growth"],
            "comparison": False,
            "needs_documents": False,
            "restated_question": "What is NVIDIA's revenue growth?",
        }
    )
)
PLAN = AIMessage(
    content=json.dumps(
        {"steps": ["fetch"], "data_needed": ["NVDA revenue"], "calculations_needed": ["growth"]}
    )
)


def working_script() -> list:
    return [
        UNDERSTANDING,
        PLAN,
        tool_call("get_income_statement", {"ticker": "NVDA", "period": "annual"}, "d1"),
        AIMessage(content="Collected NVDA revenue."),
        tool_call(
            "calculate_revenue_growth",
            {"previous_revenue": 26974000000, "current_revenue": 60922000000},
            "c1",
        ),
        AIMessage(content="Computed growth."),
        AIMessage(content="NVIDIA grew strongly (calculated)."),
        AIMessage(content="Concentration risk."),
        AIMessage(content=REPORT_WITH_ALL_SECTIONS),
    ]


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def toolkit():
    async with MCPToolkit(SERVERS) as tk:
        yield tk


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def db():
    """A real database when one is reachable; otherwise a disconnected one.

    The API must work either way, so the tests assert against the actual state rather
    than requiring PostgreSQL.
    """
    database = Database(
        os.getenv(
            "TEST_DATABASE_URL",
            "postgresql://analyst:local-dev-only-not-a-real-secret@localhost:5433/"
            "financial_analyst",
        )
    )
    await database.connect()
    yield database
    await database.close()


@pytest_asyncio.fixture(loop_scope="module")
async def client(toolkit, db, monkeypatch):
    """A real ASGI client over the real app, with startup replaced by our fixtures."""
    model = ScriptedChatModel(script=working_script())
    dependencies.state.toolkit = toolkit
    dependencies.state.agent = FinancialAnalystAgent(model, toolkit)
    dependencies.state.agent_error = None
    dependencies.state.db = db
    dependencies.state.checkpointer = None

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        c.model = model  # type: ignore[attr-defined]
        yield c


@pytest_asyncio.fixture(loop_scope="module")
async def offline_agent(client):
    """Simulate a deployment with no LLM configured."""
    dependencies.state.agent = None
    dependencies.state.agent_error = "ANTHROPIC_API_KEY is not configured."
    yield client


@pytest.fixture
def nvda(patch_ticker):
    from tests.conftest import NVDA_FINANCIALS, NVDA_INFO, NVDA_NEWS_RAW, FakeTicker

    patch_ticker(FakeTicker(info=NVDA_INFO, financials=NVDA_FINANCIALS, news=NVDA_NEWS_RAW))


# ── /health ───────────────────────────────────────────────────────────────────


async def test_health_reports_component_status(client, db):
    response = await client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["agent_ready"] is True
    assert body["mcp_servers"] == {
        "calculator": True, "market": True, "financial": True, "news": True
    }
    assert body["failed_servers"] == {}
    assert body["database_connected"] is db.available
    # "ok" requires every dependency; without a database the API still serves, degraded.
    assert body["status"] == ("ok" if db.available else "degraded")


async def test_health_reports_degraded_without_an_agent(offline_agent):
    body = (await offline_agent.get("/health")).json()
    assert body["status"] == "degraded"
    assert body["agent_ready"] is False


# ── /models and /tools ────────────────────────────────────────────────────────


async def test_models_lists_the_active_model(client):
    body = (await client.get("/models")).json()
    assert body["active"]
    assert any(m["provider"] == "anthropic" for m in body["models"])


async def test_tools_exposes_every_mcp_capability(client):
    body = (await client.get("/tools")).json()

    assert body["total"] == 23
    assert set(body["servers"]) == {"calculator", "market", "financial", "news"}
    assert len(body["servers"]["calculator"]) == 10

    growth = next(t for t in body["tools"] if t["name"] == "calculate_revenue_growth")
    assert growth["server"] == "calculator"
    assert "Use this" in growth["description"]
    assert "previous_revenue" in growth["input_schema"]["properties"]


# ── /companies ────────────────────────────────────────────────────────────────


async def test_company_snapshot_needs_no_llm(client, nvda):
    body = (await client.get("/companies/nvda")).json()

    assert body["symbol"] == "NVDA"
    assert body["profile"]["sector"] == "Technology"
    assert body["quote"]["price"] == 120.5
    assert body["statistics"]["pe_ratio_trailing"] == 51.2


async def test_company_returns_422_for_a_malformed_symbol(client):
    response = await client.get("/companies/123")
    assert response.status_code == 422
    assert "not a valid ticker" in response.json()["detail"]


async def test_company_returns_404_for_an_unknown_symbol(client, patch_ticker):
    from tests.conftest import FakeTicker

    patch_ticker(FakeTicker(info={}))
    response = await client.get("/companies/ZZZZ")
    assert response.status_code == 404


async def test_company_reports_partial_data_rather_than_failing(client, patch_ticker):
    """One broken sub-request should degrade the snapshot, not void it."""
    from tests.conftest import NVDA_INFO, FakeTicker

    calls = {"n": 0}

    def factory(_symbol):
        calls["n"] += 1
        if calls["n"] == 3:  # statistics
            return FakeTicker(raises=RuntimeError("rate limit"))
        return FakeTicker(info=NVDA_INFO)

    patch_ticker(factory)
    body = (await client.get("/companies/NVDA")).json()

    assert body["profile"] is not None
    assert body["statistics"] is None
    assert any("statistics unavailable" in w for w in body["warnings"])


# ── /chat ─────────────────────────────────────────────────────────────────────


async def test_chat_runs_the_agent_and_returns_a_structured_response(client, nvda):
    response = await client.post("/chat", json={"query": "What is NVIDIA's revenue growth?"})
    assert response.status_code == 200

    body = response.json()
    assert "## Executive Summary" in body["answer"]
    assert body["companies"][0]["ticker"] == "NVDA"
    assert set(body["mcp_servers_used"]) == {"financial", "calculator"}
    assert "calculate_revenue_growth" in body["tools_used"]
    assert body["metrics"][0]["value"] == 125.85
    assert body["metrics"][0]["formula"]
    assert body["conversation_id"]
    assert body["request_id"]
    assert body["duration_ms"] >= 0


async def test_chat_response_matches_the_declared_schema(client, nvda):
    from api.schemas import ChatResponse

    body = (await client.post("/chat", json={"query": "NVIDIA revenue growth"})).json()
    ChatResponse.model_validate(body)  # raises if the contract drifted


async def test_chat_echoes_a_supplied_conversation_id(client, nvda):
    body = (
        await client.post(
            "/chat", json={"query": "NVIDIA revenue growth", "conversation_id": "conv-123"}
        )
    ).json()
    assert body["conversation_id"] == "conv-123"


@pytest.mark.parametrize(
    "payload",
    [
        {},                                  # no query
        {"query": ""},                       # empty
        {"query": "   "},                    # blank
        {"query": "x" * 2001},               # too long
        {"query": "ok", "conversation_id": "c" * 200},  # id too long
    ],
)
async def test_invalid_chat_requests_return_422(client, payload):
    response = await client.post("/chat", json=payload)
    assert response.status_code == 422


async def test_chat_returns_503_when_no_agent_is_configured(offline_agent):
    response = await offline_agent.post("/chat", json={"query": "NVIDIA revenue growth"})
    assert response.status_code == 503
    assert "ANTHROPIC_API_KEY" in response.json()["detail"]


async def test_agent_failure_becomes_a_controlled_502(client, nvda, monkeypatch):
    async def boom(*args, **kwargs):
        raise RuntimeError("graph exploded")

    monkeypatch.setattr(dependencies.state.agent, "run", boom)
    response = await client.post("/chat", json={"query": "NVIDIA revenue growth"})

    assert response.status_code == 502
    assert "graph exploded" not in response.text  # internals are not leaked
    assert "RuntimeError" in response.json()["detail"]


async def test_tool_failures_reach_the_response_as_warnings(client, patch_ticker):
    from tests.conftest import FakeTicker

    patch_ticker(FakeTicker(raises=TimeoutError("timed out")))
    body = (await client.post("/chat", json={"query": "NVIDIA revenue growth"})).json()

    assert body["warnings"]
    assert any(not call["ok"] for call in body["tool_calls"])


# ── Cross-cutting ─────────────────────────────────────────────────────────────


async def test_every_response_carries_a_correlation_id(client):
    response = await client.get("/health")
    assert response.headers["X-Request-ID"]


async def test_a_supplied_correlation_id_is_honoured(client, nvda):
    response = await client.post(
        "/chat", json={"query": "NVIDIA revenue growth"}, headers={"X-Request-ID": "trace-abc"}
    )
    assert response.headers["X-Request-ID"] == "trace-abc"
    assert response.json()["request_id"] == "trace-abc"


async def test_unknown_route_returns_404(client):
    assert (await client.get("/nope")).status_code == 404


async def test_openapi_documentation_is_generated(client):
    schema = (await client.get("/openapi.json")).json()

    assert schema["info"]["title"] == "MCP-Powered Financial Analyst"
    for path in ("/chat", "/health", "/models", "/tools", "/companies/{symbol}"):
        assert path in schema["paths"]
    assert "ChatResponse" in schema["components"]["schemas"]


async def test_docs_page_is_served(client):
    assert (await client.get("/docs")).status_code == 200


# ── Persistence through the API ───────────────────────────────────────────────


async def test_chat_persists_the_conversation_and_it_can_be_restored(client, db, nvda):
    if not db.available:
        pytest.skip("PostgreSQL not reachable; run: docker compose up -d postgres")

    conversation_id = f"api-{uuid.uuid4().hex[:10]}"
    try:
        await client.post(
            "/chat",
            json={"query": "What is NVIDIA's revenue growth?", "conversation_id": conversation_id},
        )
        restored = (await client.get(f"/conversations/{conversation_id}")).json()

        assert restored["conversation_id"] == conversation_id
        assert [m["role"] for m in restored["messages"]] == ["user", "assistant"]
        assert restored["messages"][0]["content"] == "What is NVIDIA's revenue growth?"
        assert "## Executive Summary" in restored["messages"][1]["content"]
        assert [c["ticker"] for c in restored["companies"]] == ["NVDA"]

        analysis = restored["analyses"][0]
        assert "calculate_revenue_growth" in analysis["tools_used"]
        assert set(analysis["mcp_servers_used"]) == {"financial", "calculator"}
    finally:
        await db.delete_conversation(conversation_id)


async def test_restoring_an_unknown_conversation_returns_404(client, db):
    if not db.available:
        pytest.skip("PostgreSQL not reachable")
    assert (await client.get("/conversations/does-not-exist")).status_code == 404


async def test_health_reports_cache_status(client):
    body = (await client.get("/health")).json()
    assert body["cache"]["backend"] in ("memory", "redis")
    assert "hit_rate" in body["cache"]
