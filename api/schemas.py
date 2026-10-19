"""Request and response schemas for the HTTP API.

Separate from the agent's internal models: the API contract should be able to stay stable
while the graph's state evolves.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from agent.state import CompanyRef, MetricRecord, ToolRecord

MAX_QUERY_CHARS = 2000


class ChatRequest(BaseModel):
    """A question for the analyst."""

    query: str = Field(
        description="The financial question to analyse.",
        min_length=1,
        max_length=MAX_QUERY_CHARS,
    )
    conversation_id: str | None = Field(
        default=None,
        description="Continue an existing conversation. Omit to start a new one.",
        max_length=128,
    )

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Query must not be blank.")
        return cleaned


class ChatResponse(BaseModel):
    """The analyst's structured answer."""

    answer: str = Field(description="The final report, in markdown.")
    companies: list[CompanyRef] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    mcp_servers_used: list[str] = Field(default_factory=list)
    metrics: list[MetricRecord] = Field(default_factory=list)
    tool_calls: list[ToolRecord] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    conversation_id: str
    request_id: str = Field(description="Correlation id; matches the X-Request-ID header.")
    duration_ms: float


class HealthResponse(BaseModel):
    """Liveness and dependency status."""

    status: str = Field(description="'ok' when the API can serve, 'degraded' otherwise.")
    version: str
    mcp_servers: dict[str, bool] = Field(
        default_factory=dict, description="Server name -> connected."
    )
    failed_servers: dict[str, str] = Field(default_factory=dict)
    agent_ready: bool = Field(description="False when no LLM is configured.")
    database_connected: bool = Field(
        default=False, description="False when conversations cannot be persisted."
    )
    checkpointer_backend: str = Field(
        default="none",
        description="'postgres' when follow-ups survive a restart, 'memory' when they do not.",
    )
    cache: dict[str, Any] = Field(
        default_factory=dict, description="Cache backend and hit statistics."
    )


class ModelInfo(BaseModel):
    id: str
    provider: str
    configured: bool = Field(description="True when credentials for it are present.")


class ModelsResponse(BaseModel):
    models: list[ModelInfo]
    active: str


class ToolInfo(BaseModel):
    """One MCP tool, as advertised to the agent."""

    name: str
    server: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)


class ToolsResponse(BaseModel):
    servers: dict[str, list[str]] = Field(description="Server name -> its tool names.")
    tools: list[ToolInfo]
    total: int
    unavailable_servers: dict[str, str] = Field(default_factory=dict)


class CompanyResponse(BaseModel):
    """A quick company snapshot, assembled without invoking the LLM."""

    symbol: str
    profile: dict[str, Any] | None = None
    quote: dict[str, Any] | None = None
    statistics: dict[str, Any] | None = None
    warnings: list[str] = Field(default_factory=list)


class ConversationResponse(BaseModel):
    """A restored conversation: transcript, remembered companies and past analyses."""

    conversation_id: str
    messages: list[dict[str, Any]] = Field(default_factory=list)
    companies: list[dict[str, Any]] = Field(
        default_factory=list, description="Companies this conversation has discussed."
    )
    analyses: list[dict[str, Any]] = Field(
        default_factory=list, description="Provenance of each completed analysis."
    )


class ErrorResponse(BaseModel):
    """Every error the API returns has this shape."""

    error: str = Field(description="Short machine-readable error kind.")
    detail: str = Field(description="Human-readable explanation.")
    request_id: str
