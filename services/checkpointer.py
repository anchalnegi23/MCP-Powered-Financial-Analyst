"""LangGraph checkpointing, backed by PostgreSQL with an in-memory fallback.

The checkpointer is what makes a follow-up question work. When a conversation id is
supplied, LangGraph loads that thread's prior state — including the message history the
tools produced — so "what about their profit margins" has "their" to resolve.

Postgres is used when reachable; otherwise the process falls back to an in-memory saver,
which keeps follow-ups working within a single process but loses them on restart. That
degradation is announced, never silent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog
from langgraph.checkpoint.memory import InMemorySaver

from services.config import get_settings
from services.database import CONNECT_TIMEOUT_SECONDS

logger = structlog.get_logger("checkpointer")


@dataclass
class CheckpointerHandle:
    """A checkpointer plus how it was obtained, so /health can report the truth."""

    saver: Any
    backend: str
    persistent: bool
    error: str | None = None

    async def close(self) -> None:
        closer = getattr(self, "_close", None)
        if closer is not None:
            await closer()


async def build_checkpointer(
    dsn: str | None = None, *, connect_timeout: float | None = None
) -> CheckpointerHandle:
    """Return a LangGraph checkpointer, preferring PostgreSQL.

    Never raises: an unreachable database downgrades persistence rather than preventing
    the agent from starting.
    """
    dsn = dsn or get_settings().database_url

    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from psycopg_pool import AsyncConnectionPool

        pool = AsyncConnectionPool(
            dsn,
            min_size=1,
            max_size=5,
            open=False,
            kwargs={"autocommit": True, "prepare_threshold": 0},
        )
        await pool.open(wait=True, timeout=connect_timeout or CONNECT_TIMEOUT_SECONDS)
        saver = AsyncPostgresSaver(pool)  # type: ignore[arg-type]
        await saver.setup()

        handle = CheckpointerHandle(saver=saver, backend="postgres", persistent=True)
        handle._close = pool.close  # type: ignore[attr-defined]
        logger.info("checkpointer_ready", backend="postgres")
        return handle
    except Exception as exc:  # noqa: BLE001 - degrade, do not fail
        error = f"{type(exc).__name__}: {str(exc)[:200]}"
        logger.warning("checkpointer_fallback", backend="memory", reason=error)
        return CheckpointerHandle(
            saver=InMemorySaver(), backend="memory", persistent=False, error=error
        )
