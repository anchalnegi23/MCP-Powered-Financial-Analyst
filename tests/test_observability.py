"""Observability: structured logs, correlation ids, timing and secret hygiene."""

from __future__ import annotations

import json
import logging

import pytest
import structlog

from services.observability import (
    SENSITIVE_KEYS,
    bind_request_context,
    clear_request_context,
    configure_logging,
    new_request_id,
    redact_processor,
    traced,
)


@pytest.fixture(autouse=True)
def json_logging():
    configure_logging(json_logs=True, level="DEBUG")
    clear_request_context()
    yield
    clear_request_context()


def emitted(capsys) -> list[dict]:
    """Parse the JSON log lines produced so far."""
    captured = capsys.readouterr()
    events = []
    for line in (captured.out + captured.err).splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


# ── Structure ─────────────────────────────────────────────────────────────────


def test_logs_are_structured_json(capsys):
    structlog.get_logger("t").info("analysis_started", query_chars=42)
    events = emitted(capsys)

    assert events
    event = events[-1]
    assert event["event"] == "analysis_started"
    assert event["query_chars"] == 42
    assert event["level"] == "info"
    assert "timestamp" in event


def test_logs_are_readable_in_a_console(capsys):
    configure_logging(json_logs=False, level="INFO")
    structlog.get_logger("t").info("analysis_started", tool="get_stock_price")

    output = capsys.readouterr().out + capsys.readouterr().err
    assert "analysis_started" in output


# ── Correlation ───────────────────────────────────────────────────────────────


def test_request_ids_are_unique():
    assert new_request_id() != new_request_id()
    assert len(new_request_id()) == 16


def test_bound_context_appears_on_every_subsequent_line(capsys):
    """The mechanism that makes a trace possible without threading a parameter everywhere."""
    bind_request_context(request_id="req-abc", conversation_id="conv-1")

    structlog.get_logger("api").info("analysis_started")
    structlog.get_logger("mcp_client").info("mcp_tool_called", tool="calculate_roe")
    structlog.get_logger("service.market").info("tool_completed", ticker="NVDA")

    events = emitted(capsys)
    assert len(events) >= 3
    assert all(e["request_id"] == "req-abc" for e in events)
    assert all(e["conversation_id"] == "conv-1" for e in events)


def test_context_can_be_cleared(capsys):
    bind_request_context(request_id="req-abc")
    clear_request_context("request_id")
    structlog.get_logger("t").info("after_clear")

    assert "request_id" not in emitted(capsys)[-1]


# ── Timing ────────────────────────────────────────────────────────────────────


def test_traced_records_duration(capsys):
    with traced("mcp_tool", structlog.get_logger("t"), server="calculator"):
        pass

    event = emitted(capsys)[-1]
    assert event["event"] == "mcp_tool_completed"
    assert event["server"] == "calculator"
    assert isinstance(event["duration_ms"], float)


def test_traced_logs_failures_and_reraises(capsys):
    with pytest.raises(ValueError), traced("mcp_tool", structlog.get_logger("t"), tool="x"):
        raise ValueError("boom")

    event = emitted(capsys)[-1]
    assert event["event"] == "mcp_tool_failed"
    assert event["error_type"] == "ValueError"
    assert "duration_ms" in event


def test_extra_context_added_inside_the_block_is_logged(capsys):
    with traced("mcp_tool", structlog.get_logger("t")) as extra:
        extra["tool"] = "get_stock_price"

    assert emitted(capsys)[-1]["tool"] == "get_stock_price"


# ── Secret hygiene ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("key", sorted(SENSITIVE_KEYS))
def test_sensitive_keys_are_dropped(key):
    assert redact_processor(None, "", {key: "super-secret-value"})[key] == "[REDACTED]"


def test_credentials_inside_message_text_are_scrubbed():
    """The realistic leak: a key inside a third-party exception message."""
    event = redact_processor(
        None, "", {"detail": "GET https://api.x.com?apikey=abc123XYZ failed"}
    )
    assert "abc123XYZ" not in event["detail"]


def test_credentials_nested_in_a_dict_are_dropped():
    event = redact_processor(None, "", {"config": {"api_key": "sk-live-x", "model": "sonnet"}})
    assert event["config"]["api_key"] == "[REDACTED]"
    assert event["config"]["model"] == "sonnet"


def test_a_real_secret_never_survives_the_pipeline(capsys, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-DO-NOT-LEAK-THIS")
    structlog.get_logger("t").warning(
        "provider_error",
        anthropic_api_key="sk-ant-DO-NOT-LEAK-THIS",
        detail="auth failed for key sk-ant-DO-NOT-LEAK-THIS",
        database_url="postgresql://user:hunter2@db/app",
    )

    raw = capsys.readouterr()
    combined = raw.out + raw.err
    assert "DO-NOT-LEAK-THIS" not in combined
    assert "hunter2" not in combined


def test_ordinary_financial_values_are_not_redacted(capsys):
    structlog.get_logger("t").info(
        "calculation_completed", tool="calculate_revenue_growth", value=125.85, ticker="NVDA"
    )
    event = emitted(capsys)[-1]

    assert event["value"] == 125.85
    assert event["ticker"] == "NVDA"


def test_log_level_is_respected(capsys):
    configure_logging(json_logs=True, level="WARNING")
    structlog.get_logger("t").info("should_not_appear")
    structlog.get_logger("t").warning("should_appear")

    events = [e["event"] for e in emitted(capsys)]
    assert "should_not_appear" not in events
    assert "should_appear" in events
    logging.getLogger().setLevel(logging.INFO)


# ── End-to-end trace ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_one_request_can_be_traced_across_api_agent_mcp_and_tool(capsys, monkeypatch):
    """The whole point of Phase 15: follow a single request through every layer.

    Runs the real API, the real graph and real MCP servers, then asserts that one
    correlation id appears on log lines emitted by each layer.
    """
    import httpx

    from agent.graph import FinancialAnalystAgent
    from agent.mcp_client import MCPServerSpec, MCPToolkit
    from api import dependencies
    from api.main import app
    from servers.calculator_mcp.server import mcp as calculator_server
    from servers.financial_mcp.server import mcp as financial_server
    from tests.conftest import NVDA_FINANCIALS, NVDA_INFO, FakeTicker
    from tests.fakes import ScriptedChatModel
    from tests.test_api import working_script

    monkeypatch.setattr(
        "services.providers.yfinance_provider.yf.Ticker",
        lambda symbol, *a, **k: FakeTicker(info=NVDA_INFO, financials=NVDA_FINANCIALS),
    )

    servers = [
        MCPServerSpec("calculator", "servers.calculator_mcp.server", "Maths.", calculator_server),
        MCPServerSpec(
            "financial", "servers.financial_mcp.server", "Statements.", financial_server
        ),
    ]

    async with MCPToolkit(servers) as toolkit:
        dependencies.state.toolkit = toolkit
        dependencies.state.agent = FinancialAnalystAgent(
            ScriptedChatModel(script=working_script()), toolkit
        )
        dependencies.state.agent_error = None

        capsys.readouterr()  # discard startup noise
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/chat",
                json={"query": "What is NVIDIA's revenue growth?"},
                headers={"X-Request-ID": "trace-e2e-001"},
            )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "trace-e2e-001"

    events = [e for e in emitted(capsys) if e.get("request_id") == "trace-e2e-001"]
    names = {e["event"] for e in events}

    # Every layer logged under the one correlation id.
    assert "analysis_started" in names          # API
    assert "query_understood" in names          # agent
    assert "mcp_tool_called" in names           # MCP client
    assert "tool_completed" in names            # MCP server / service
    assert "analysis_completed" in names        # API
    assert "request_completed" in names         # middleware

    # Tool execution time is recorded, and attributed to a server and a stage.
    tool_calls = [e for e in events if e["event"] == "mcp_tool_called"]
    assert tool_calls
    assert all(isinstance(e["duration_ms"], int | float) for e in tool_calls)
    assert {e["server"] for e in tool_calls} == {"financial", "calculator"}
    assert all("agent_stage" in e for e in tool_calls)

    # The conversation id was bound from /chat onwards.
    assert any(e.get("conversation_id") for e in events)

    # The final API event carries the response time and the provenance summary.
    completed = next(e for e in events if e["event"] == "analysis_completed")
    assert completed["duration_ms"] > 0
    assert "calculate_revenue_growth" in completed["tools_used"]


@pytest.mark.asyncio
async def test_agent_and_api_errors_are_logged(capsys, monkeypatch):
    import httpx

    from api import dependencies
    from api.main import app

    class Boom:
        async def run(self, *args, **kwargs):
            raise RuntimeError("graph exploded")

    dependencies.state.agent = Boom()  # type: ignore[assignment]
    capsys.readouterr()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/chat", json={"query": "anything"}, headers={"X-Request-ID": "trace-err"}
        )

    assert response.status_code == 502
    events = [e for e in emitted(capsys) if e.get("request_id") == "trace-err"]
    assert any(e["event"] == "analysis_failed" for e in events)
    assert any(e.get("error_type") == "RuntimeError" for e in events)
