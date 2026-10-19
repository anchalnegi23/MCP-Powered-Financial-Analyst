"""The full stack through the Streamlit UI.

Streamlit -> a real uvicorn server -> the real FastAPI app -> the real LangGraph agent
-> the real MCP client -> real FastMCP servers.

Nothing is stubbed except the LLM's choices and the market data provider. In particular
the HTTP hop is real: the UI makes an actual network request to a listening socket.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
import threading
from pathlib import Path

import pytest
import uvicorn
from langchain_core.messages import AIMessage
from streamlit.testing.v1 import AppTest

from agent.graph import FinancialAnalystAgent
from agent.mcp_client import MCPServerSpec, MCPToolkit
from api import dependencies
from api.main import app
from servers.calculator_mcp.server import mcp as calculator_server
from servers.financial_mcp.server import mcp as financial_server
from servers.news_mcp.server import mcp as news_server
from services.database import Database
from tests.conftest import NVDA_FINANCIALS, NVDA_INFO, NVDA_NEWS_RAW, FakeTicker
from tests.fakes import REPORT_WITH_ALL_SECTIONS, ScriptedChatModel, tool_call

APP = str(Path(__file__).resolve().parents[1] / "frontend" / "app.py")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def build_script() -> list:
    return [
        AIMessage(
            content=json.dumps(
                {
                    "companies": [
                        {"name": "NVIDIA", "ticker": "NVDA"},
                        {"name": "AMD", "ticker": "AMD"},
                    ],
                    "intents": ["revenue growth", "recent news"],
                    "comparison": True,
                    "needs_documents": False,
                    "restated_question": "Compare NVIDIA and AMD revenue growth and news.",
                }
            )
        ),
        AIMessage(
            content=json.dumps(
                {"steps": ["gather"], "data_needed": ["revenue"], "calculations_needed": ["growth"]}
            )
        ),
        tool_call("get_income_statement", {"ticker": "NVDA", "period": "annual"}, "d1"),
        tool_call("get_latest_company_news", {"ticker": "NVDA", "limit": 2}, "d2"),
        AIMessage(content="Collected statements and news."),
        tool_call(
            "calculate_revenue_growth",
            {"previous_revenue": 26_974_000_000, "current_revenue": 60_922_000_000},
            "c1",
        ),
        AIMessage(content="Computed revenue growth."),
        AIMessage(content="NVIDIA grew far faster (calculated)."),
        AIMessage(content="Concentration and export-control risk."),
        AIMessage(content=REPORT_WITH_ALL_SECTIONS),
    ]


@pytest.fixture(scope="module")
def live_api(request):
    """Run the real API on a real port, in a background event loop."""
    port = free_port()
    ready = threading.Event()
    stop = threading.Event()
    model = ScriptedChatModel(script=build_script())

    def serve() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def main() -> None:
            # Patch the provider inside this loop's thread, then wire up real components.
            import services.providers.yfinance_provider as provider

            provider.yf.Ticker = lambda symbol, *a, **k: FakeTicker(  # type: ignore[assignment]
                info=NVDA_INFO, financials=NVDA_FINANCIALS, news=NVDA_NEWS_RAW
            )

            servers = [
                MCPServerSpec(
                    "calculator", "servers.calculator_mcp.server", "Maths.", calculator_server
                ),
                MCPServerSpec(
                    "financial", "servers.financial_mcp.server", "Statements.", financial_server
                ),
                MCPServerSpec("news", "servers.news_mcp.server", "News.", news_server),
            ]
            async with MCPToolkit(servers) as toolkit:
                dependencies.state.toolkit = toolkit
                dependencies.state.agent = FinancialAnalystAgent(model, toolkit)
                dependencies.state.agent_error = None
                dependencies.state.db = Database(
                    "postgresql://nobody@127.0.0.1:1/none", connect_timeout=1
                )

                config = uvicorn.Config(
                    app, host="127.0.0.1", port=port, log_level="error", lifespan="off"
                )
                server = uvicorn.Server(config)
                task = asyncio.create_task(server.serve())
                while not server.started:
                    await asyncio.sleep(0.05)
                ready.set()
                while not stop.is_set():
                    await asyncio.sleep(0.05)
                server.should_exit = True
                with contextlib.suppress(Exception):
                    await task

        loop.run_until_complete(main())
        loop.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    assert ready.wait(timeout=60), "the API did not start"

    # The model is yielded so each test can rewind the script: the server is module-scoped,
    # so without this the second test would run against an exhausted script.
    yield f"http://127.0.0.1:{port}", model

    stop.set()
    thread.join(timeout=15)


@pytest.fixture
def ui(live_api, monkeypatch):
    url, model = live_api
    monkeypatch.setenv("API_BASE_URL", url)
    model.index = 0  # rewind the script for this test
    return url


def test_the_whole_stack_works_through_the_streamlit_ui(ui):
    """Streamlit -> HTTP -> FastAPI -> LangGraph -> MCP client -> three FastMCP servers."""
    app_test = AppTest.from_file(APP, default_timeout=120)
    app_test.run()
    assert not app_test.exception

    app_test.chat_input[0].set_value(
        "Compare NVIDIA and AMD on revenue growth and recent news."
    ).run()

    assert not app_test.exception

    markdown = " ".join(m.value for m in app_test.markdown)
    assert "Executive Summary" in markdown
    assert "Bull Case" in markdown and "Bear Case" in markdown
    assert "not investment advice" in markdown.lower()


def test_the_ui_shows_which_mcp_servers_and_tools_ran(ui):
    app_test = AppTest.from_file(APP, default_timeout=120)
    app_test.run()
    app_test.chat_input[0].set_value("Compare NVIDIA and AMD revenue growth.").run()

    captions = " ".join(c.value for c in app_test.caption)
    assert "financial" in captions
    assert "calculator" in captions
    assert "calculate_revenue_growth" in captions


def test_the_ui_shows_the_deterministic_metric_with_its_formula(ui):
    app_test = AppTest.from_file(APP, default_timeout=120)
    app_test.run()
    app_test.chat_input[0].set_value("Compare NVIDIA and AMD revenue growth.").run()

    metrics_tab = next(t for t in app_test.tabs if "Calculated metrics" in t.label)
    row = metrics_tab.dataframe[0].value.iloc[0]

    assert row["Value"] == "125.85%"
    assert "previous_revenue" in row["Formula"]
    assert "26,974,000,000" in row["Inputs"]


def test_the_ui_shows_sources_and_the_backend_status(ui):
    app_test = AppTest.from_file(APP, default_timeout=120)
    app_test.run()

    sidebar_text = " ".join(c.value for c in app_test.caption)
    assert "calculator" in sidebar_text  # live /health reached the UI

    app_test.chat_input[0].set_value("What is the latest news on NVIDIA?").run()
    markdown = " ".join(m.value for m in app_test.markdown)
    assert "https://" in markdown


def test_conversation_context_is_carried_across_ui_turns(ui):
    app_test = AppTest.from_file(APP, default_timeout=120)
    app_test.run()
    app_test.chat_input[0].set_value("Compare NVIDIA and AMD.").run()
    conversation_id = app_test.session_state["conversation_id"]
    assert conversation_id

    app_test.chat_input[0].set_value("Which one has the better profit margin?").run()

    assert app_test.session_state["conversation_id"] == conversation_id
    assert len(app_test.session_state["history"]) == 4
