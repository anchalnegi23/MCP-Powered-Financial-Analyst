"""Application startup: model construction and dependency wiring, including degradation."""

from __future__ import annotations

import pytest

from agent.llm import build_model
from agent.mcp_client import MCPServerSpec
from api.dependencies import AppState
from services.config import get_settings
from services.database import Database

pytestmark = pytest.mark.asyncio(loop_scope="module")

from servers.calculator_mcp.server import mcp as calculator_server  # noqa: E402

# In-process: these tests are about wiring, not about subprocess transport (covered in
# tests/test_mcp_client.py), and spawning a process per test is needlessly slow.
CALCULATOR = MCPServerSpec(
    "calculator", "servers.calculator_mcp.server", "Maths.", in_process=calculator_server
)
GHOST = MCPServerSpec("ghost", "servers.no_such_module.server", "Never starts.")

# A port nothing listens on. connect_timeout=1 keeps the degradation tests quick.
UNREACHABLE = "postgresql://nobody@127.0.0.1:1/none"


@pytest.fixture
def no_llm_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def fake_llm_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake-key-for-construction-only")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# ── Model factory ─────────────────────────────────────────────────────────────


async def test_building_a_model_without_a_key_explains_what_to_do(no_llm_key):
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY is not configured"):
        build_model()


async def test_a_model_is_constructed_when_a_key_is_present(fake_llm_key):
    """Construction only: no request is made, so no real key is needed."""
    model = build_model()
    assert model.model == get_settings().llm_model
    assert model.temperature == 0.0  # a non-deterministic analyst is not useful


async def test_temperature_is_configurable(fake_llm_key):
    assert build_model(temperature=0.3).temperature == 0.3


# ── Application state ─────────────────────────────────────────────────────────


async def test_startup_connects_mcp_servers(no_llm_key):
    state = AppState(servers=[CALCULATOR], db=Database(UNREACHABLE, connect_timeout=1))
    try:
        await state.startup()

        assert state.toolkit is not None
        assert state.toolkit.connected_servers == ["calculator"]
        assert state.checkpointer is not None
    finally:
        await state.shutdown()


async def test_startup_degrades_when_no_llm_is_configured(no_llm_key):
    """Missing credentials must disable /chat, not prevent the process from starting."""
    state = AppState(servers=[CALCULATOR], db=Database(UNREACHABLE, connect_timeout=1))
    try:
        await state.startup()

        assert state.agent is None
        assert state.agent_error is not None
        assert "ANTHROPIC_API_KEY" in state.agent_error
        assert state.toolkit.connected_servers == ["calculator"]  # the rest still works
    finally:
        await state.shutdown()


async def test_startup_degrades_when_a_server_cannot_start(no_llm_key):
    state = AppState(
        servers=[CALCULATOR, GHOST], db=Database(UNREACHABLE, connect_timeout=1)
    )
    try:
        await state.startup()

        assert state.toolkit.connected_servers == ["calculator"]
        assert "ghost" in state.toolkit.failed_servers
    finally:
        await state.shutdown()


async def test_startup_degrades_when_the_database_is_unreachable(no_llm_key):
    state = AppState(servers=[CALCULATOR], db=Database(UNREACHABLE, connect_timeout=1))
    try:
        await state.startup()

        assert state.db.available is False
        assert state.checkpointer.backend == "memory"  # follow-ups still work in-process
        assert state.checkpointer.persistent is False
    finally:
        await state.shutdown()


async def test_shutdown_is_idempotent(no_llm_key):
    state = AppState(servers=[CALCULATOR], db=Database(UNREACHABLE, connect_timeout=1))
    await state.startup()
    await state.shutdown()
    await state.shutdown()  # must not raise

    assert state.toolkit is None
    assert state.agent is None
