"""Streamlit UI: rendering, history, error presentation and secret hygiene.

Uses Streamlit's own AppTest harness to run the real script, with the backend intercepted
by respx. No browser, no live API.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "frontend" / "app.py")
API = "http://localhost:8000"

HEALTH = {
    "status": "ok",
    "version": "0.1.0",
    "mcp_servers": {"calculator": True, "market": True, "financial": True, "news": True},
    "failed_servers": {},
    "agent_ready": True,
}

ANALYSIS = {
    "answer": (
        "## Executive Summary\nNVIDIA grew strongly.\n\n## Revenue Growth\n"
        "Revenue growth was 125.85% (calculated).\n\n## Conclusion\nStrong growth."
    ),
    "companies": [{"name": "NVIDIA", "ticker": "NVDA"}],
    "tools_used": ["get_income_statement", "calculate_revenue_growth"],
    "mcp_servers_used": ["financial", "calculator"],
    "metrics": [
        {
            "metric": "revenue_growth",
            "label": "Revenue Growth",
            "value": 125.85,
            "unit": "percent",
            "formula": "((current_revenue - previous_revenue) / previous_revenue) * 100",
            "inputs": {"previous_revenue": 26974000000.0, "current_revenue": 60922000000.0},
            "warnings": [],
            "company": None,
        }
    ],
    "tool_calls": [
        {
            "server": "financial",
            "tool": "get_income_statement",
            "arguments": {"ticker": "NVDA"},
            "ok": True,
            "duration_ms": 42.0,
            "error": None,
        },
        {
            "server": "market",
            "tool": "get_stock_price",
            "arguments": {"ticker": "NVDA"},
            "ok": False,
            "duration_ms": 12.0,
            "error": "ProviderTimeoutError: provider timed out",
        },
    ],
    "sources": ["https://example.com/nvidia-gpu-launch"],
    "warnings": ["Tool 'get_stock_price' on the market server failed: timed out"],
    "conversation_id": "conv-1",
    "request_id": "req-1",
    "duration_ms": 2400.0,
}


@pytest.fixture
def backend():
    """Intercept every backend call. Individual tests override routes as needed."""
    with respx.mock(base_url=API, assert_all_called=False) as router:
        router.get("/health").respond(200, json=HEALTH)
        router.post("/chat").respond(200, json=ANALYSIS)
        yield router


def run_app(backend, timeout: float = 30) -> AppTest:
    app = AppTest.from_file(APP, default_timeout=timeout)
    app.run()
    return app


# ── Startup ───────────────────────────────────────────────────────────────────


def test_app_starts_without_exceptions(backend):
    app = run_app(backend)
    assert not app.exception
    assert any("Financial Analyst" in t.value for t in app.title)


def test_backend_status_is_shown_in_the_sidebar(backend):
    app = run_app(backend)
    text = " ".join(c.value for c in app.caption) + " ".join(s.value for s in app.success)
    assert "calculator" in text
    assert "Backend: ok" in text


def test_example_queries_are_offered(backend):
    app = run_app(backend)
    labels = [b.label for b in app.button]
    assert "What's NVIDIA's current stock price?" in labels
    assert "Compare NVIDIA and AMD revenue growth." in labels
    assert len([label for label in labels if "NVIDIA" in label or "AMD" in label]) >= 3


# ── Submitting a question ─────────────────────────────────────────────────────


def test_submitting_a_question_renders_the_response(backend):
    app = run_app(backend)
    app.chat_input[0].set_value("What is NVIDIA's revenue growth?").run()

    markdown = " ".join(m.value for m in app.markdown)
    assert "Executive Summary" in markdown
    assert "125.85% (calculated)" in markdown


def test_conversation_history_is_preserved_across_turns(backend):
    app = run_app(backend)
    app.chat_input[0].set_value("First question").run()
    app.chat_input[0].set_value("Second question").run()

    history = app.session_state["history"]
    assert [t["role"] for t in history] == ["user", "assistant", "user", "assistant"]
    assert history[0]["content"] == "First question"
    assert history[2]["content"] == "Second question"


def test_the_conversation_id_is_carried_into_follow_up_requests(backend):
    """Follow-ups must reach the same conversation, or memory is pointless."""
    app = run_app(backend)
    app.chat_input[0].set_value("First").run()
    app.chat_input[0].set_value("What about their margins?").run()

    second_request = json.loads(backend.calls[-1].request.content)
    assert second_request["conversation_id"] == "conv-1"


def test_empty_input_is_handled(backend):
    app = run_app(backend)
    app.chat_input[0].set_value("   ").run()

    assert not app.exception
    assert app.session_state["history"] == []
    assert any("enter a question" in w.value.lower() for w in app.warning)


def test_clearing_the_conversation_resets_state(backend):
    app = run_app(backend)
    app.chat_input[0].set_value("First").run()
    assert app.session_state["history"]

    next(b for b in app.button if "Clear" in b.label).click().run()
    assert app.session_state["history"] == []
    assert app.session_state["conversation_id"] is None


# ── Provenance rendering ──────────────────────────────────────────────────────


def test_tools_and_servers_used_are_visible(backend):
    app = run_app(backend)
    app.chat_input[0].set_value("NVIDIA revenue growth").run()

    captions = " ".join(c.value for c in app.caption)
    assert "financial" in captions
    assert "calculate_revenue_growth" in captions


def test_metrics_render_with_their_formula(backend):
    app = run_app(backend)
    app.chat_input[0].set_value("NVIDIA revenue growth").run()

    metrics_tab = next(t for t in app.tabs if "Calculated metrics" in t.label)
    assert metrics_tab.dataframe, "no metric table rendered"
    row = metrics_tab.dataframe[0].value.iloc[0]

    assert row["Metric"] == "Revenue Growth"
    assert row["Value"] == "125.85%"
    assert "previous_revenue" in row["Formula"]
    assert "previous_revenue=26,974,000,000" in row["Inputs"]


def test_sources_are_visible(backend):
    app = run_app(backend)
    app.chat_input[0].set_value("NVIDIA revenue growth").run()

    markdown = " ".join(m.value for m in app.markdown)
    assert "https://example.com/nvidia-gpu-launch" in markdown


def test_warnings_and_failed_tools_are_surfaced(backend):
    """A failed tool must be visible, not buried: it is why a number may be missing."""
    app = run_app(backend)
    app.chat_input[0].set_value("NVIDIA revenue growth").run()

    warnings = " ".join(w.value for w in app.warning)
    assert "get_stock_price" in warnings

    tools_tab = next(t for t in app.tabs if "Tool calls" in t.label)
    table = tools_tab.dataframe[0].value
    assert "ProviderTimeoutError" in " ".join(table["Error"].astype(str))
    assert set(table[""]) == {"✅", "❌"}


# ── Error presentation ────────────────────────────────────────────────────────


def test_api_error_is_presented_clearly(backend):
    backend.post("/chat").respond(503, json={"detail": "The analyst agent is unavailable"})
    app = run_app(backend)
    app.chat_input[0].set_value("NVIDIA revenue growth").run()

    errors = " ".join(e.value for e in app.error)
    assert "503" in errors
    assert "analyst agent is unavailable" in errors
    assert not app.exception


def test_unreachable_backend_gives_actionable_guidance(backend):
    backend.get("/health").mock(side_effect=httpx.ConnectError("refused"))
    backend.post("/chat").mock(side_effect=httpx.ConnectError("refused"))
    app = run_app(backend)
    app.chat_input[0].set_value("NVIDIA revenue growth").run()

    errors = " ".join(e.value for e in app.error)
    assert "uvicorn api.main:app" in errors
    assert not app.exception


def test_timeout_is_reported_not_raised(backend):
    backend.post("/chat").mock(side_effect=httpx.ReadTimeout("slow"))
    app = run_app(backend)
    app.chat_input[0].set_value("NVIDIA revenue growth").run()

    assert any("timed out" in e.value for e in app.error)
    assert not app.exception


# ── Secret hygiene ────────────────────────────────────────────────────────────


def test_the_ui_holds_no_credentials():
    """The frontend talks only to the API, so there is no key in this process to leak."""
    source = Path(APP).read_text()

    for forbidden in ("ANTHROPIC_API_KEY", "NEWSAPI_API_KEY", "DATABASE_URL", "anthropic"):
        assert forbidden not in source, f"frontend references {forbidden}"

    # ...and it never imports the agent, the MCP client or a data provider.
    for module in ("agent", "servers", "yfinance", "langchain"):
        assert f"import {module}" not in source
        assert f"from {module}" not in source
