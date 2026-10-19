"""FastAPI application.

Thin by design: it validates input, assigns a correlation id, calls the agent, and maps
exceptions onto HTTP status codes. No financial logic lives here.
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from agent.state import AnalysisResponse
from api.dependencies import state
from api.schemas import (
    ChatRequest,
    ChatResponse,
    CompanyResponse,
    ConversationResponse,
    ErrorResponse,
    HealthResponse,
    ModelInfo,
    ModelsResponse,
    ToolInfo,
    ToolsResponse,
)
from models.common import normalize_ticker
from services import market_service
from services.cache import cache
from services.config import get_settings
from services.errors import ProviderError, TickerNotFoundError
from services.observability import bind_request_context, clear_request_context, configure_logging

logger = structlog.get_logger("api")

VERSION = "0.1.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Connect the MCP servers once, at startup, and close them cleanly on shutdown."""
    configure_logging()
    await state.startup()
    logger.info(
        "api_started",
        connected=state.toolkit.connected_servers if state.toolkit else [],
        agent_ready=state.agent is not None,
    )
    yield
    await state.shutdown()


app = FastAPI(
    title="MCP-Powered Financial Analyst",
    version=VERSION,
    description=(
        "An agentic financial research API. A LangGraph agent plans a research workflow "
        "and dynamically selects tools from five independent FastMCP servers.\n\n"
        "**This service produces research summaries, not investment advice.**"
    ),
    lifespan=lifespan,
)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Attach a correlation id to every request, and bind it to the log context.

    The same id appears in the response body, the X-Request-ID header and every log line
    the request produces, which is what makes an end-to-end trace possible.
    """
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
    request.state.request_id = request_id

    bind_request_context(request_id=request_id, path=request.url.path, method=request.method)
    started = time.perf_counter()
    try:
        response = await call_next(request)
    finally:
        clear_request_context("request_id", "path", "method")

    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request_completed",
        request_id=request_id,
        path=request.url.path,
        duration_ms=round((time.perf_counter() - started) * 1000, 2),
    )
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a structured error instead of leaking a stack trace to the caller."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.exception("unhandled_error", request_id=request_id, error_type=type(exc).__name__)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="internal_error",
            detail="The request could not be completed. See server logs for the request id.",
            request_id=request_id,
        ).model_dump(),
    )


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    """Liveness and dependency status. Never fails: reporting 'degraded' is the point."""
    toolkit = state.toolkit
    connected = {name: s.connected for name, s in (toolkit.statuses if toolkit else {}).items()}
    failed = toolkit.failed_servers if toolkit else {}

    healthy = bool(connected) and not failed and state.agent is not None and state.db.available
    return HealthResponse(
        status="ok" if healthy else "degraded",
        version=VERSION,
        mcp_servers=connected,
        failed_servers=failed,
        agent_ready=state.agent is not None,
        database_connected=state.db.available,
        checkpointer_backend=state.checkpointer.backend if state.checkpointer else "none",
        cache=cache.stats(),
    )


@app.get("/models", response_model=ModelsResponse, tags=["ops"])
async def models() -> ModelsResponse:
    """The models this deployment can use, and whether credentials are present."""
    settings = get_settings()
    catalogue = [
        "claude-opus-5",
        "claude-sonnet-5",
        "claude-haiku-4-5-20251001",
        "gpt-4o",
        "gpt-4o-mini",
    ]
    if settings.llm_model not in catalogue:
        catalogue.insert(0, settings.llm_model)

    def provider_of(name: str) -> str:
        return "anthropic" if name.startswith("claude") else "openai"

    keys = {
        "anthropic": bool(settings.anthropic_api_key),
        "openai": bool(settings.openai_api_key),
    }
    return ModelsResponse(
        models=[
            ModelInfo(
                id=name,
                provider=provider_of(name),
                configured=keys[provider_of(name)],
            )
            for name in catalogue
        ],
        active=settings.llm_model,
    )


@app.get("/tools", response_model=ToolsResponse, tags=["mcp"])
async def tools() -> ToolsResponse:
    """Every MCP capability currently available, grouped by server.

    This is the discovery surface: it shows what the agent can actually do right now,
    including which servers are down.
    """
    toolkit = state.toolkit
    if toolkit is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MCP servers are not connected yet.",
        )

    schemas = toolkit.tool_schemas()
    return ToolsResponse(
        servers={
            name: status_.tool_names
            for name, status_ in toolkit.statuses.items()
            if status_.connected
        },
        tools=[
            ToolInfo(
                name=s["name"],
                server=s["server"],
                description=s["description"],
                input_schema=s["input_schema"],
            )
            for s in schemas
        ],
        total=len(schemas),
        unavailable_servers=toolkit.failed_servers,
    )


@app.get("/companies/{symbol}", response_model=CompanyResponse, tags=["data"])
async def company(symbol: str) -> CompanyResponse:
    """A company snapshot straight from the market service, with no LLM involved.

    Useful for populating a UI and for checking data availability before asking a
    question about a company.
    """
    try:
        ticker = normalize_ticker(symbol)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc

    warnings: list[str] = []
    results: dict[str, Any] = {}
    for field_name, fetch in (
        ("profile", market_service.get_company_profile),
        ("quote", market_service.get_stock_price),
        ("statistics", market_service.get_stock_statistics),
    ):
        try:
            results[field_name] = (await fetch(ticker)).model_dump(mode="json")
        except TickerNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        except ProviderError as exc:
            # Partial data is more useful than none, as long as the gap is stated.
            results[field_name] = None
            warnings.append(f"{field_name} unavailable: {exc}")

    return CompanyResponse(symbol=ticker, warnings=warnings, **results)


@app.post("/chat", response_model=ChatResponse, tags=["agent"])
async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    """Run a full financial analysis and return the structured result.

    The response separates what was retrieved, what was calculated and what the model
    concluded, and lists every tool call including the ones that failed.
    """
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex[:16])
    if state.agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"The analyst agent is unavailable: {state.agent_error or 'not initialised'}",
        )

    conversation_id = payload.conversation_id or uuid.uuid4().hex
    # Bound for the rest of this request, so every downstream log line — agent stage,
    # MCP client, MCP server, tool — carries the conversation id without being passed it.
    bind_request_context(conversation_id=conversation_id)
    started = time.perf_counter()
    logger.info("analysis_started", query_chars=len(payload.query))

    # Written before the analysis runs, so a crashed request still leaves a record of
    # what was asked.
    await state.db.add_message(conversation_id, "user", payload.query)

    try:
        result: AnalysisResponse = await state.agent.run(
            payload.query, conversation_id=conversation_id
        )
    except Exception as exc:  # noqa: BLE001 - mapped to a controlled 502
        logger.exception("analysis_failed", request_id=request_id, error_type=type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"The analysis could not be completed: {type(exc).__name__}",
        ) from exc

    duration_ms = round((time.perf_counter() - started) * 1000, 2)
    logger.info(
        "analysis_completed",
        request_id=request_id,
        conversation_id=conversation_id,
        tools_used=result.tools_used,
        servers=result.mcp_servers_used,
        warnings=len(result.warnings),
        duration_ms=duration_ms,
    )

    await state.db.add_message(conversation_id, "assistant", result.answer)
    await state.db.remember_companies(
        conversation_id, [(c.ticker, c.name) for c in result.companies if c.ticker]
    )
    await state.db.record_analysis(
        conversation_id, payload.query, result, request_id=request_id, duration_ms=duration_ms
    )

    return ChatResponse(
        **result.model_dump(exclude={"conversation_id"}),
        conversation_id=conversation_id,
        request_id=request_id,
        duration_ms=duration_ms,
    )


@app.get("/conversations/{conversation_id}", response_model=ConversationResponse, tags=["agent"])
async def conversation(conversation_id: str) -> ConversationResponse:
    """Restore a stored conversation: its transcript, its companies and its analyses.

    This is the persisted record, which is deliberately separate from LangGraph's
    checkpointed state — a transcript should be readable without deserialising graph
    internals.
    """
    if not state.db.available:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Conversation history is unavailable: the database is not connected.",
        )

    context = await state.db.restore_context(conversation_id)
    analyses = await state.db.get_analyses(conversation_id)
    if not context["messages"] and not analyses:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No conversation found with id '{conversation_id}'.",
        )

    return ConversationResponse(
        conversation_id=conversation_id,
        messages=[
            {"role": m["role"], "content": m["content"], "created_at": str(m["created_at"])}
            for m in context["messages"]
        ],
        companies=[
            {"ticker": c["ticker"], "company_name": c["company_name"]}
            for c in context["companies"]
        ],
        analyses=[
            {
                "id": a["id"],
                "query": a["query"],
                "tools_used": a["tools_used"],
                "mcp_servers_used": a["mcp_servers_used"],
                "duration_ms": a["duration_ms"],
                "created_at": str(a["created_at"]),
            }
            for a in analyses
        ],
    )
