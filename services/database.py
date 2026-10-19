"""PostgreSQL persistence for conversations, messages and analysis metadata.

Two separate concerns share the database:

1. **This module** stores the human-readable record — what was asked, what was answered,
   which tools ran, what was calculated. It is what a UI reads and what an operator
   inspects.
2. **LangGraph's checkpointer** (see `services.checkpointer`) stores graph state so a
   follow-up question resumes the same thread. That is machine state, not a transcript.

Keeping them apart matters: the transcript must survive a change to the graph's internal
state shape, and it must be readable without deserialising LangGraph internals.

Connection failures degrade the service rather than killing it. An analysis that ran
correctly but could not be written down is still a useful answer.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

import psycopg
import structlog
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from services.config import get_settings

logger = structlog.get_logger("database")

# Generous enough for a cold container, short enough that an outage fails fast.
CONNECT_TIMEOUT_SECONDS = 5

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id            TEXT PRIMARY KEY,
    title         TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS messages (
    id              BIGSERIAL PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content         TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS messages_conversation_idx
    ON messages (conversation_id, created_at);

-- Companies seen in a conversation. This is what lets a follow-up like "what about
-- their margins" resolve without re-reading the whole transcript.
CREATE TABLE IF NOT EXISTS company_context (
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    ticker          TEXT NOT NULL,
    company_name    TEXT,
    last_seen_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (conversation_id, ticker)
);

-- One row per completed analysis: the provenance of an answer, queryable.
CREATE TABLE IF NOT EXISTS analyses (
    id               TEXT PRIMARY KEY,
    conversation_id  TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    request_id       TEXT,
    query            TEXT NOT NULL,
    tools_used       JSONB NOT NULL DEFAULT '[]'::jsonb,
    mcp_servers_used JSONB NOT NULL DEFAULT '[]'::jsonb,
    metrics          JSONB NOT NULL DEFAULT '[]'::jsonb,
    warnings         JSONB NOT NULL DEFAULT '[]'::jsonb,
    duration_ms      DOUBLE PRECISION,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS analyses_conversation_idx
    ON analyses (conversation_id, created_at DESC);
"""


class Database:
    """An async connection pool plus the queries this application needs.

    Every method is safe to call when the database is down: it logs and returns a neutral
    value rather than raising into a request handler.
    """

    def __init__(self, dsn: str | None = None, *, connect_timeout: float | None = None) -> None:
        self.dsn = dsn or get_settings().database_url
        self.connect_timeout = connect_timeout or CONNECT_TIMEOUT_SECONDS
        self.pool: AsyncConnectionPool | None = None
        self.available = False
        self.error: str | None = None

    async def connect(self, *, create_schema: bool = True) -> bool:
        """Open the pool and ensure the schema exists. Returns whether it succeeded."""
        try:
            self.pool = AsyncConnectionPool(
                self.dsn, min_size=1, max_size=10, open=False, kwargs={"row_factory": dict_row}
            )
            await self.pool.open(wait=True, timeout=self.connect_timeout)
            if create_schema:
                async with self.pool.connection() as connection:
                    await connection.execute(SCHEMA)
            self.available = True
            self.error = None
            logger.info("database_connected")
        except Exception as exc:  # noqa: BLE001 - a missing database must not be fatal
            self.available = False
            self.error = f"{type(exc).__name__}: {str(exc)[:200]}"
            logger.warning("database_unavailable", error=self.error)
            self.pool = None
        return self.available

    async def close(self) -> None:
        if self.pool is not None:
            await self.pool.close()
        self.pool = None
        self.available = False

    async def _execute(self, query: str, params: tuple = (), *, fetch: str | None = None):
        """Run one statement, tolerating an unavailable database."""
        if not self.available or self.pool is None:
            return [] if fetch == "all" else None
        try:
            async with self.pool.connection() as connection, connection.cursor() as cursor:
                await cursor.execute(query, params)  # type: ignore[arg-type]
                if fetch == "all":
                    return await cursor.fetchall()
                if fetch == "one":
                    return await cursor.fetchone()
                return None
        except psycopg.Error as exc:
            logger.warning("query_failed", error_type=type(exc).__name__, detail=str(exc)[:200])
            return [] if fetch == "all" else None

    # ── conversations ────────────────────────────────────────────────────────

    async def ensure_conversation(self, conversation_id: str, title: str | None = None) -> str:
        await self._execute(
            """
            INSERT INTO conversations (id, title) VALUES (%s, %s)
            ON CONFLICT (id) DO UPDATE SET updated_at = now()
            """,
            (conversation_id, (title or "")[:200] or None),
        )
        return conversation_id

    async def list_conversations(self, limit: int = 50) -> list[dict[str, Any]]:
        return await self._execute(
            "SELECT id, title, created_at, updated_at FROM conversations "
            "ORDER BY updated_at DESC LIMIT %s",
            (limit,),
            fetch="all",
        ) or []

    async def delete_conversation(self, conversation_id: str) -> None:
        await self._execute("DELETE FROM conversations WHERE id = %s", (conversation_id,))

    # ── messages ─────────────────────────────────────────────────────────────

    async def add_message(self, conversation_id: str, role: str, content: str) -> None:
        if role not in ("user", "assistant"):
            raise ValueError(f"Invalid message role '{role}'. Use 'user' or 'assistant'.")
        await self.ensure_conversation(conversation_id)
        await self._execute(
            "INSERT INTO messages (conversation_id, role, content) VALUES (%s, %s, %s)",
            (conversation_id, role, content),
        )

    async def get_messages(self, conversation_id: str, limit: int = 100) -> list[dict[str, Any]]:
        return await self._execute(
            "SELECT role, content, created_at FROM messages WHERE conversation_id = %s "
            "ORDER BY created_at, id LIMIT %s",
            (conversation_id, limit),
            fetch="all",
        ) or []

    # ── company context ──────────────────────────────────────────────────────

    async def remember_companies(
        self, conversation_id: str, companies: list[tuple[str, str | None]]
    ) -> None:
        """Record which companies a conversation is about, for pronoun resolution."""
        await self.ensure_conversation(conversation_id)
        for ticker, name in companies:
            if not ticker:
                continue
            await self._execute(
                """
                INSERT INTO company_context (conversation_id, ticker, company_name)
                VALUES (%s, %s, %s)
                ON CONFLICT (conversation_id, ticker)
                DO UPDATE SET last_seen_at = now(), company_name = EXCLUDED.company_name
                """,
                (conversation_id, ticker.upper(), name),
            )

    async def get_companies(self, conversation_id: str) -> list[dict[str, Any]]:
        return await self._execute(
            "SELECT ticker, company_name, last_seen_at FROM company_context "
            "WHERE conversation_id = %s ORDER BY last_seen_at DESC",
            (conversation_id,),
            fetch="all",
        ) or []

    # ── analyses ─────────────────────────────────────────────────────────────

    async def record_analysis(
        self,
        conversation_id: str,
        query: str,
        response: Any,
        *,
        request_id: str | None = None,
        duration_ms: float | None = None,
    ) -> str:
        """Store the provenance of one completed analysis."""
        analysis_id = uuid.uuid4().hex
        await self.ensure_conversation(conversation_id, title=query)
        await self._execute(
            """
            INSERT INTO analyses (id, conversation_id, request_id, query, tools_used,
                                  mcp_servers_used, metrics, warnings, duration_ms)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                analysis_id,
                conversation_id,
                request_id,
                query,
                json.dumps(response.tools_used),
                json.dumps(response.mcp_servers_used),
                json.dumps([m.model_dump() for m in response.metrics]),
                json.dumps(response.warnings),
                duration_ms,
            ),
        )
        return analysis_id

    async def get_analyses(self, conversation_id: str) -> list[dict[str, Any]]:
        return await self._execute(
            "SELECT id, query, tools_used, mcp_servers_used, metrics, warnings, duration_ms, "
            "created_at FROM analyses WHERE conversation_id = %s ORDER BY created_at DESC",
            (conversation_id,),
            fetch="all",
        ) or []

    # ── restoring a conversation ─────────────────────────────────────────────

    async def restore_context(self, conversation_id: str) -> dict[str, Any]:
        """Everything needed to continue a conversation: transcript plus known companies."""
        return {
            "conversation_id": conversation_id,
            "messages": await self.get_messages(conversation_id),
            "companies": await self.get_companies(conversation_id),
            "restored_at": datetime.now(UTC).isoformat(),
        }


database = Database()
