"""MCP client: connect to every FastMCP server, discover their tools, expose them to LangGraph.

This module is the *only* place that knows MCP exists. The agent graph above it sees
nothing but a list of LangChain tools; the servers below it see nothing but MCP protocol
calls. That separation is the point — adding a sixth server changes this file's registry
and nothing in the graph.

Written against `fastmcp.Client` directly rather than `langchain-mcp-adapters`, which is
currently incompatible with the MCP SDK v2 that FastMCP 4 requires. Owning the adapter
also lets us decide exactly how a tool failure is presented to the model, which matters:
a failure must reach the LLM as a readable message it can react to, never as a silent
empty result it might paper over.
"""

from __future__ import annotations

import contextvars
import json
import os
import sys
import time
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any

import structlog
from fastmcp import Client
from fastmcp.client.transports import StdioTransport, StreamableHttpTransport
from langchain_core.tools import BaseTool, StructuredTool

logger = structlog.get_logger("mcp_client")

# Set per agent run so concurrent requests each collect their own tool history. Without
# this, two simultaneous queries would read each other's invocations out of the shared
# toolkit and report the wrong provenance.
current_run_invocations: contextvars.ContextVar[list[ToolInvocation] | None] = (
    contextvars.ContextVar("current_run_invocations", default=None)
)


@dataclass(frozen=True)
class MCPServerSpec:
    """How to start and identify one FastMCP server.

    Three transports, one protocol:

    - `url` — an already-running HTTP MCP service. Used by Docker Compose, where each
      server is a separate container that can be scaled and restarted independently.
    - `module` — launched as a stdio subprocess. The default for local development: no
      orchestration needed, still real process isolation.
    - `in_process` — attaches to an imported FastMCP server object over FastMCP's
      in-memory transport. Used by tests and single-process deployments.
    """

    name: str
    module: str
    description: str
    in_process: Any | None = None
    url: str | None = None

    def transport(self) -> Any:
        if self.in_process is not None:
            return self.in_process
        if self.url:
            # Containerised deployment: each server is its own long-lived HTTP service.
            return StreamableHttpTransport(self.url)
        return StdioTransport(
            command=sys.executable,
            args=["-m", self.module],
            # Children inherit configuration but are never handed the parent's argv.
            env={**os.environ, "MCP_TRANSPORT": "stdio"},
        )


# The registry: the single list of servers the system knows about. Adding a sixth means
# adding one entry here and nothing in the agent graph.
_REGISTRY: tuple[tuple[str, str, str], ...] = (
    ("calculator", "servers.calculator_mcp.server", "Deterministic financial mathematics."),
    ("market", "servers.market_mcp.server", "Live and historical market data."),
    ("financial", "servers.financial_mcp.server", "Reported financial statements and ratios."),
    ("news", "servers.news_mcp.server", "Company and market news search."),
    (
        "research",
        "servers.research_mcp.server",
        "Financial document retrieval (10-K, 10-Q, transcripts).",
    ),
)


def default_servers() -> tuple[MCPServerSpec, ...]:
    """Build the server registry from the environment.

    Setting `<NAME>_MCP_URL` (e.g. `MARKET_MCP_URL=http://market-mcp:8002/mcp`) switches
    that server from a local stdio subprocess to a remote HTTP service. That single
    switch is what lets the identical code run as one process locally and as nine
    containers under Compose.
    """
    return tuple(
        MCPServerSpec(
            name=name,
            module=module,
            description=description,
            url=os.getenv(f"{name.upper()}_MCP_URL") or None,
        )
        for name, module, description in _REGISTRY
    )


DEFAULT_SERVERS: tuple[MCPServerSpec, ...] = default_servers()


@dataclass
class ToolInvocation:
    """The record of one tool call, kept for observability and for the final report.

    Successes and failures are both recorded. A failed call that the agent then ignores
    is exactly the situation where a model is tempted to invent a plausible number, so
    the failure has to stay visible all the way to the response.
    """

    server: str
    tool: str
    arguments: dict[str, Any]
    ok: bool
    duration_ms: float
    result: Any = None
    error: str | None = None


@dataclass
class ServerStatus:
    """Whether one server came up, and what it offers."""

    name: str
    connected: bool
    tool_names: list[str] = field(default_factory=list)
    error: str | None = None


class MCPToolkit:
    """Manages connections to several FastMCP servers and adapts their tools for LangGraph.

    A server that fails to start is recorded and skipped rather than failing the whole
    session: an outage in the news server should not prevent a purely numerical question
    from being answered.

    Usage::

        async with MCPToolkit() as toolkit:
            tools = toolkit.langchain_tools()
    """

    def __init__(
        self,
        servers: tuple[MCPServerSpec, ...] | list[MCPServerSpec] | None = None,
        *,
        tool_timeout: float = 60.0,
    ) -> None:
        # Resolved at construction, not import, so environment changes are picked up.
        self.specs = list(servers if servers is not None else default_servers())
        self.tool_timeout = tool_timeout
        self._clients: dict[str, Client] = {}
        self._schemas: dict[str, list[dict[str, Any]]] = {}
        self.statuses: dict[str, ServerStatus] = {}
        self.invocations: list[ToolInvocation] = []

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def __aenter__(self) -> MCPToolkit:
        await self.connect()
        return self

    async def __aexit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                        tb: TracebackType | None) -> None:
        await self.aclose()

    async def connect(self) -> dict[str, ServerStatus]:
        """Start every configured server and discover its tools.

        Returns a status per server. Never raises for a single server's failure.
        """
        for spec in self.specs:
            started = time.perf_counter()
            try:
                client = Client(spec.transport())
                await client.__aenter__()
                tools = await client.list_tools()
            except Exception as exc:  # noqa: BLE001 - one server must not sink the rest
                logger.warning(
                    "mcp_server_unavailable",
                    server=spec.name,
                    error_type=type(exc).__name__,
                    error=str(exc)[:200],
                )
                self.statuses[spec.name] = ServerStatus(
                    name=spec.name,
                    connected=False,
                    error=f"{type(exc).__name__}: {str(exc)[:200]}",
                )
                continue

            self._clients[spec.name] = client
            self._schemas[spec.name] = [
                {
                    "name": tool.name,
                    "description": tool.description or "",
                    "input_schema": tool.input_schema,
                    "output_schema": tool.output_schema,
                }
                for tool in tools
            ]
            self.statuses[spec.name] = ServerStatus(
                name=spec.name, connected=True, tool_names=[t.name for t in tools]
            )
            logger.info(
                "mcp_server_connected",
                server=spec.name,
                tool_count=len(tools),
                duration_ms=round((time.perf_counter() - started) * 1000, 1),
            )

        return self.statuses

    async def aclose(self) -> None:
        """Shut every server down. A failure closing one must not block the others."""
        for name, client in list(self._clients.items()):
            try:
                await client.__aexit__(None, None, None)
            except Exception as exc:  # noqa: BLE001
                logger.warning("mcp_close_failed", server=name, error=str(exc)[:200])
        self._clients.clear()

    # ── discovery ────────────────────────────────────────────────────────────

    @property
    def connected_servers(self) -> list[str]:
        return [name for name, status in self.statuses.items() if status.connected]

    @property
    def failed_servers(self) -> dict[str, str]:
        return {
            name: status.error or "unknown error"
            for name, status in self.statuses.items()
            if not status.connected
        }

    def tool_schemas(self) -> list[dict[str, Any]]:
        """Every discovered tool, with its server, description and JSON schemas intact."""
        return [
            {"server": server, **schema}
            for server, schemas in self._schemas.items()
            for schema in schemas
        ]

    def server_for_tool(self, tool_name: str) -> str | None:
        for server, schemas in self._schemas.items():
            if any(schema["name"] == tool_name for schema in schemas):
                return server
        return None

    # ── invocation ───────────────────────────────────────────────────────────

    async def call_tool(self, server: str, tool: str, arguments: dict[str, Any]) -> ToolInvocation:
        """Invoke one tool and record the outcome, successful or not."""
        started = time.perf_counter()
        client = self._clients.get(server)
        if client is None:
            invocation = ToolInvocation(
                server=server,
                tool=tool,
                arguments=arguments,
                ok=False,
                duration_ms=0.0,
                error=f"MCP server '{server}' is not connected.",
            )
            self._record(invocation)
            return invocation

        try:
            result = await client.call_tool(tool, arguments, timeout=self.tool_timeout)
            payload = result.structured_content if result.structured_content is not None else (
                result.data if hasattr(result, "data") else None
            )
            invocation = ToolInvocation(
                server=server,
                tool=tool,
                arguments=arguments,
                ok=True,
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
                result=payload,
            )
            logger.info(
                "mcp_tool_called",
                server=server,
                tool=tool,
                ok=True,
                duration_ms=invocation.duration_ms,
                argument_keys=sorted(arguments),  # names only: values may be large
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the model, not swallowed
            invocation = ToolInvocation(
                server=server,
                tool=tool,
                arguments=arguments,
                ok=False,
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
                error=f"{type(exc).__name__}: {str(exc)[:400]}",
            )
            logger.warning(
                "mcp_tool_failed",
                server=server,
                tool=tool,
                error_type=type(exc).__name__,
                duration_ms=invocation.duration_ms,
            )

        self._record(invocation)
        return invocation

    def _record(self, invocation: ToolInvocation) -> None:
        self.invocations.append(invocation)
        run_log = current_run_invocations.get()
        if run_log is not None:
            run_log.append(invocation)

    # ── LangChain adaptation ─────────────────────────────────────────────────

    def langchain_tools(self) -> list[BaseTool]:
        """Adapt every discovered MCP tool into a LangChain tool.

        The MCP input schema is passed through untouched, so the model sees exactly the
        parameter names, types and descriptions the server declared. Tool names collide
        only if two servers declare the same one; that is detected and disambiguated
        rather than silently shadowed.
        """
        seen: dict[str, str] = {}
        tools: list[BaseTool] = []

        for schema in self.tool_schemas():
            server, name = schema["server"], schema["name"]
            exposed = name
            if name in seen:
                exposed = f"{server}__{name}"
                logger.warning(
                    "mcp_tool_name_collision",
                    tool=name,
                    servers=[seen[name], server],
                    exposed_as=exposed,
                )
            seen.setdefault(name, server)
            tools.append(self._as_langchain_tool(server, name, exposed, schema))

        return tools

    def _as_langchain_tool(
        self, server: str, tool_name: str, exposed_name: str, schema: dict[str, Any]
    ) -> BaseTool:
        async def _invoke(**kwargs: Any) -> str:
            invocation = await self.call_tool(server, tool_name, kwargs)
            if invocation.ok:
                return json.dumps(invocation.result, default=str)
            # Returned rather than raised: the model must see the failure and adapt
            # (try another tool, or report the gap) instead of the graph aborting.
            return json.dumps(
                {
                    "error": invocation.error,
                    "tool": tool_name,
                    "server": server,
                    "guidance": "This tool call failed. Do not invent the value it would "
                    "have returned. Either try a different tool or report the data as "
                    "unavailable.",
                }
            )

        return StructuredTool(
            name=exposed_name,
            description=f"[{server} MCP] {schema['description']}",
            args_schema=schema["input_schema"],
            coroutine=_invoke,
            metadata={"mcp_server": server, "mcp_tool": tool_name},
        )
