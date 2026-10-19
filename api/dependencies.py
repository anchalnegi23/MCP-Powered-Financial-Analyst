"""Shared application state: the MCP toolkit and the agent, built once per process.

Starting five MCP subprocesses per HTTP request would be absurd, so the toolkit is
created at startup and shared. The agent is rebuilt lazily because it may be unavailable
(no LLM key) while the rest of the API still works.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog

from agent.graph import FinancialAnalystAgent
from agent.mcp_client import MCPServerSpec, MCPToolkit, default_servers
from services.cache import build_cache
from services.checkpointer import CheckpointerHandle, build_checkpointer
from services.database import Database, database

logger = structlog.get_logger("api")


@dataclass
class AppState:
    """Everything the request handlers need, assembled at startup."""

    toolkit: MCPToolkit | None = None
    agent: FinancialAnalystAgent | None = None
    agent_error: str | None = None
    checkpointer: CheckpointerHandle | None = None
    db: Database = field(default_factory=lambda: database)
    servers: tuple[MCPServerSpec, ...] | list[MCPServerSpec] = field(
        default_factory=default_servers
    )

    async def startup(self) -> None:
        """Connect MCP servers, persistence and the agent, in that order.

        Every dependency degrades independently. No LLM key means /chat is unavailable but
        /health, /tools and /companies still work; no database means analyses still run
        but are not written down and follow-ups only survive within this process.
        """
        self.toolkit = MCPToolkit(self.servers)
        await self.toolkit.connect()

        # The checkpointer uses the same DSN as the transcript store. Reading global
        # settings here instead would let the two point at different databases.
        self.checkpointer = await build_checkpointer(
            self.db.dsn, connect_timeout=self.db.connect_timeout
        )
        await build_cache()
        await self.db.connect()

        try:
            from agent.llm import build_model

            self.agent = FinancialAnalystAgent(
                build_model(),
                self.toolkit,
                checkpointer=self.checkpointer.saver if self.checkpointer else None,
            )
            self.agent_error = None
        except Exception as exc:  # noqa: BLE001 - reported through /health, not fatal
            self.agent = None
            self.agent_error = str(exc)
            logger.warning("agent_unavailable", reason=str(exc)[:200])

    async def shutdown(self) -> None:
        if self.toolkit is not None:
            await self.toolkit.aclose()
        if self.checkpointer is not None:
            await self.checkpointer.close()
        await self.db.close()
        self.toolkit = None
        self.agent = None
        self.checkpointer = None


state = AppState()
