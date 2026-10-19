"""PostgreSQL persistence and conversation memory.

These run against a real PostgreSQL, started with `docker compose up -d postgres`. When
it is not reachable the tests skip rather than fail, and a separate group asserts that
the application degrades correctly when the database is down.
"""

from __future__ import annotations

import json
import os
import uuid

import pytest
import pytest_asyncio
from langchain_core.messages import AIMessage

from agent.graph import FinancialAnalystAgent
from agent.mcp_client import MCPServerSpec, MCPToolkit
from agent.state import AnalysisResponse, CompanyRef, MetricRecord
from servers.calculator_mcp.server import mcp as calculator_server
from servers.financial_mcp.server import mcp as financial_server
from services.checkpointer import build_checkpointer
from services.database import Database
from tests.fakes import REPORT_WITH_ALL_SECTIONS, ScriptedChatModel

pytestmark = pytest.mark.asyncio(loop_scope="module")

TEST_DSN = os.getenv(
    "TEST_DATABASE_URL",
    "postgresql://analyst:local-dev-only-not-a-real-secret@localhost:5433/financial_analyst",
)

UNREACHABLE = "postgresql://nobody:nothing@127.0.0.1:1/none"

SERVERS = [
    MCPServerSpec("calculator", "servers.calculator_mcp.server", "Maths.", calculator_server),
    MCPServerSpec("financial", "servers.financial_mcp.server", "Statements.", financial_server),
]


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def db():
    database = Database(TEST_DSN)
    if not await database.connect():
        pytest.skip(
            f"PostgreSQL not reachable ({database.error}). "
            "Run: docker compose up -d postgres"
        )
    yield database
    await database.close()


@pytest_asyncio.fixture(loop_scope="module")
async def conversation_id(db):
    """A unique conversation per test, cleaned up afterwards."""
    cid = f"test-{uuid.uuid4().hex[:12]}"
    yield cid
    await db.delete_conversation(cid)


# ── Schema and basic persistence ──────────────────────────────────────────────


async def test_schema_is_created(db):
    rows = await db._execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'",
        fetch="all",
    )
    tables = {row["table_name"] for row in rows}
    assert {"conversations", "messages", "company_context", "analyses"} <= tables


async def test_conversations_are_persisted(db, conversation_id):
    await db.ensure_conversation(conversation_id, title="Compare NVIDIA and AMD")
    conversations = await db.list_conversations()
    assert any(c["id"] == conversation_id for c in conversations)


async def test_messages_are_persisted_in_order(db, conversation_id):
    await db.add_message(conversation_id, "user", "Compare NVIDIA and AMD")
    await db.add_message(conversation_id, "assistant", "NVIDIA grew faster.")
    await db.add_message(conversation_id, "user", "What about their margins?")

    messages = await db.get_messages(conversation_id)
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert messages[0]["content"] == "Compare NVIDIA and AMD"
    assert messages[2]["content"] == "What about their margins?"


async def test_invalid_message_role_is_rejected(db, conversation_id):
    with pytest.raises(ValueError, match="Invalid message role"):
        await db.add_message(conversation_id, "system", "should not be stored")


async def test_company_context_is_persisted(db, conversation_id):
    await db.remember_companies(conversation_id, [("NVDA", "NVIDIA"), ("AMD", "AMD Inc")])
    companies = await db.get_companies(conversation_id)
    assert {c["ticker"] for c in companies} == {"NVDA", "AMD"}


async def test_company_context_is_idempotent(db, conversation_id):
    await db.remember_companies(conversation_id, [("NVDA", "NVIDIA")])
    await db.remember_companies(conversation_id, [("NVDA", "NVIDIA Corporation")])

    companies = await db.get_companies(conversation_id)
    assert len(companies) == 1
    assert companies[0]["company_name"] == "NVIDIA Corporation"


async def test_analysis_metadata_is_persisted(db, conversation_id):
    response = AnalysisResponse(
        answer="Report body",
        companies=[CompanyRef(name="NVIDIA", ticker="NVDA")],
        tools_used=["get_income_statement", "calculate_revenue_growth"],
        mcp_servers_used=["financial", "calculator"],
        metrics=[
            MetricRecord(
                metric="revenue_growth",
                label="Revenue Growth",
                value=125.85,
                unit="percent",
                formula="((current_revenue - previous_revenue) / previous_revenue) * 100",
                inputs={"previous_revenue": 26974000000.0, "current_revenue": 60922000000.0},
            )
        ],
        warnings=["market tool failed"],
    )
    await db.record_analysis(conversation_id, "Compare NVIDIA", response, duration_ms=2400.0)

    stored = await db.get_analyses(conversation_id)
    assert len(stored) == 1
    assert stored[0]["query"] == "Compare NVIDIA"
    assert "calculate_revenue_growth" in stored[0]["tools_used"]
    assert stored[0]["metrics"][0]["value"] == 125.85
    assert stored[0]["duration_ms"] == 2400.0


async def test_conversation_state_can_be_restored(db, conversation_id):
    await db.add_message(conversation_id, "user", "Compare NVIDIA and AMD")
    await db.remember_companies(conversation_id, [("NVDA", "NVIDIA"), ("AMD", "AMD Inc")])

    restored = await db.restore_context(conversation_id)
    assert restored["conversation_id"] == conversation_id
    assert restored["messages"][0]["content"] == "Compare NVIDIA and AMD"
    assert {c["ticker"] for c in restored["companies"]} == {"NVDA", "AMD"}


async def test_conversations_are_isolated_from_each_other(db):
    """The most important property: one user's context must never bleed into another's."""
    first, second = f"iso-a-{uuid.uuid4().hex[:8]}", f"iso-b-{uuid.uuid4().hex[:8]}"
    try:
        await db.add_message(first, "user", "Tell me about NVIDIA")
        await db.remember_companies(first, [("NVDA", "NVIDIA")])
        await db.add_message(second, "user", "Tell me about Ford")
        await db.remember_companies(second, [("F", "Ford")])

        assert [m["content"] for m in await db.get_messages(first)] == ["Tell me about NVIDIA"]
        assert [m["content"] for m in await db.get_messages(second)] == ["Tell me about Ford"]
        assert {c["ticker"] for c in await db.get_companies(first)} == {"NVDA"}
        assert {c["ticker"] for c in await db.get_companies(second)} == {"F"}
    finally:
        await db.delete_conversation(first)
        await db.delete_conversation(second)


async def test_deleting_a_conversation_cascades(db):
    cid = f"cascade-{uuid.uuid4().hex[:8]}"
    await db.add_message(cid, "user", "hello")
    await db.remember_companies(cid, [("NVDA", "NVIDIA")])
    await db.delete_conversation(cid)

    assert await db.get_messages(cid) == []
    assert await db.get_companies(cid) == []


# ── Follow-up context through the graph ───────────────────────────────────────


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def toolkit():
    async with MCPToolkit(SERVERS) as tk:
        yield tk


EMPTY_PLAN = AIMessage(
    content=json.dumps({"steps": [], "data_needed": [], "calculations_needed": []})
)


def understanding(companies: list[dict], restated: str) -> AIMessage:
    return AIMessage(
        content=json.dumps(
            {
                "companies": companies,
                "intents": ["profitability"],
                "comparison": len(companies) > 1,
                "needs_documents": False,
                "restated_question": restated,
            }
        )
    )


async def test_follow_up_question_sees_the_earlier_turn(toolkit, patch_ticker):
    """The point of checkpointing: 'their' must have an antecedent on turn two."""
    from tests.conftest import NVDA_FINANCIALS, NVDA_INFO, FakeTicker

    patch_ticker(FakeTicker(info=NVDA_INFO, financials=NVDA_FINANCIALS))

    handle = await build_checkpointer(TEST_DSN)
    thread = f"followup-{uuid.uuid4().hex[:8]}"
    try:
        model = ScriptedChatModel(
            script=[
                understanding(
                    [{"name": "NVIDIA", "ticker": "NVDA"}, {"name": "AMD", "ticker": "AMD"}],
                    "Compare NVIDIA and AMD.",
                ),
                EMPTY_PLAN,
                AIMessage(content="Collected."),
                AIMessage(content="Calculated."),
                AIMessage(content="Analysis."),
                AIMessage(content="Risks."),
                AIMessage(content=REPORT_WITH_ALL_SECTIONS),
            ]
        )
        agent = FinancialAnalystAgent(model, toolkit, checkpointer=handle.saver)
        await agent.run("Compare NVIDIA and AMD.", conversation_id=thread)

        # Turn two: a new model instance, so nothing is carried in Python memory. Anything
        # it knows about NVIDIA and AMD came out of the checkpoint.
        follow_up_model = ScriptedChatModel(
            script=[
                understanding(
                    [{"name": "NVIDIA", "ticker": "NVDA"}, {"name": "AMD", "ticker": "AMD"}],
                    "Which of NVIDIA and AMD has the better profit margin?",
                ),
                EMPTY_PLAN,
                AIMessage(content="Collected."),
                AIMessage(content="Calculated."),
                AIMessage(content="Analysis."),
                AIMessage(content="Risks."),
                AIMessage(content=REPORT_WITH_ALL_SECTIONS),
            ]
        )
        follow_up_agent = FinancialAnalystAgent(
            follow_up_model, toolkit, checkpointer=handle.saver
        )
        await follow_up_agent.run("What about their profit margins?", conversation_id=thread)

        # The understanding step on turn two received the first turn's messages.
        first_prompt = follow_up_model.prompts_seen[0]
        assert "Compare NVIDIA and AMD." in first_prompt
        assert "What about their profit margins?" in first_prompt
    finally:
        await handle.close()


async def test_different_threads_do_not_share_state(toolkit, patch_ticker):
    from tests.conftest import NVDA_INFO, FakeTicker

    patch_ticker(FakeTicker(info=NVDA_INFO))
    handle = await build_checkpointer(TEST_DSN)
    try:
        script = [
            understanding([{"name": "NVIDIA", "ticker": "NVDA"}], "NVIDIA."),
            EMPTY_PLAN,
            AIMessage(content="Collected."),
            AIMessage(content="Calculated."),
            AIMessage(content="Analysis."),
            AIMessage(content="Risks."),
            AIMessage(content=REPORT_WITH_ALL_SECTIONS),
        ]
        agent_a = FinancialAnalystAgent(
            ScriptedChatModel(script=script), toolkit, checkpointer=handle.saver
        )
        await agent_a.run(
            "Tell me about NVIDIA.", conversation_id=f"thread-a-{uuid.uuid4().hex[:8]}"
        )

        model_b = ScriptedChatModel(script=script)
        agent_b = FinancialAnalystAgent(model_b, toolkit, checkpointer=handle.saver)
        await agent_b.run("Tell me about Ford.", conversation_id=f"thread-b-{uuid.uuid4().hex[:8]}")

        assert "Tell me about NVIDIA." not in model_b.prompts_seen[0]
    finally:
        await handle.close()


async def test_checkpointer_uses_postgres_when_available():
    handle = await build_checkpointer(TEST_DSN)
    try:
        assert handle.backend == "postgres"
        assert handle.persistent is True
    finally:
        await handle.close()


# ── Degradation when the database is down ─────────────────────────────────────


async def test_database_connection_failure_is_handled_not_raised():
    unreachable = Database(UNREACHABLE, connect_timeout=1)
    assert await unreachable.connect() is False
    assert unreachable.available is False
    assert unreachable.error


async def test_writes_are_no_ops_when_the_database_is_down():
    """An analysis that ran correctly but could not be recorded is still a useful answer."""
    unreachable = Database(UNREACHABLE, connect_timeout=1)
    await unreachable.connect()

    await unreachable.add_message("c1", "user", "hello")  # must not raise
    await unreachable.remember_companies("c1", [("NVDA", "NVIDIA")])
    assert await unreachable.get_messages("c1") == []
    restored = await unreachable.restore_context("c1")
    assert restored["messages"] == []


async def test_checkpointer_falls_back_to_memory_and_says_so():
    handle = await build_checkpointer(UNREACHABLE, connect_timeout=1)
    assert handle.backend == "memory"
    assert handle.persistent is False
    assert handle.error  # the degradation is announced, not silent
