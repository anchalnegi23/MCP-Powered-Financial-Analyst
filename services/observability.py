"""Structured logging and request tracing.

Every log line is a JSON object (or a readable console line in development) carrying the
same context keys, so one request can be followed end to end:

    API -> agent -> MCP client -> MCP server -> tool

The mechanism is a `request_id` bound to a context variable at the API boundary. Because
structlog's contextvars are async-aware, every subsequent log line in that request — in
any module, including inside a tool call — carries it automatically, without threading a
parameter through twenty function signatures.

Secrets never reach the logs. A processor scrubs every event before it is rendered, which
is a belt-and-braces measure on top of not logging credentials in the first place: the
one place a key tends to appear is inside a third-party exception message nobody
anticipated.
"""

from __future__ import annotations

import logging
import sys
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import structlog

from services.config import get_settings
from services.redaction import scrub

# Keys whose value is dropped entirely rather than scrubbed, wherever they appear.
SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "anthropic_api_key",
    "openai_api_key",
    "newsapi_api_key",
    "fmp_api_key",
    "alpha_vantage_api_key",
    "password",
    "secret",
    "token",
    "authorization",
    "database_url",
    "redis_url",
    "dsn",
}

REDACTED = "[REDACTED]"


def redact_processor(_logger: Any, _name: str, event: dict[str, Any]) -> dict[str, Any]:
    """Drop sensitive keys and scrub credential-shaped text from every event.

    Applied to all log output, so a key that leaks into an exception message from a
    library we do not control still never reaches a log file.
    """
    for key in list(event):
        if key.lower() in SENSITIVE_KEYS:
            event[key] = REDACTED
        elif isinstance(event[key], str):
            event[key] = scrub(event[key])
        elif isinstance(event[key], dict):
            event[key] = {
                k: (REDACTED if k.lower() in SENSITIVE_KEYS else v) for k, v in event[key].items()
            }
    return event


def configure_logging(*, json_logs: bool | None = None, level: str | None = None) -> None:
    """Configure structlog once, at process start.

    JSON in production so logs are queryable; a coloured console renderer in development
    because a human is reading them.
    """
    settings = get_settings()
    resolved_level = (level or settings.log_level).upper()
    use_json = json_logs if json_logs is not None else not sys.stderr.isatty()

    logging.basicConfig(
        format="%(message)s", stream=sys.stdout, level=getattr(logging, resolved_level, 20)
    )

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,  # request_id and friends, automatically
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        redact_processor,  # last transform before rendering: nothing escapes it
    ]
    processors.append(
        structlog.processors.JSONRenderer()
        if use_json
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, resolved_level, 20)
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]


def bind_request_context(**values: Any) -> None:
    """Bind values onto every subsequent log line in this async context."""
    structlog.contextvars.bind_contextvars(**values)


def clear_request_context(*keys: str) -> None:
    if keys:
        structlog.contextvars.unbind_contextvars(*keys)
    else:
        structlog.contextvars.clear_contextvars()


@contextmanager
def traced(operation: str, logger: Any | None = None, **context: Any) -> Iterator[dict[str, Any]]:
    """Time an operation and log its start, outcome and duration.

    The yielded dict can be extended inside the block; whatever it holds at exit is
    included in the completion event.
    """
    log = logger or structlog.get_logger("trace")
    extra: dict[str, Any] = {}
    started = time.perf_counter()
    log.debug(f"{operation}_started", **context)
    try:
        yield extra
    except Exception as exc:
        log.warning(
            f"{operation}_failed",
            error_type=type(exc).__name__,
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
            **context,
            **extra,
        )
        raise
    log.info(
        f"{operation}_completed",
        duration_ms=round((time.perf_counter() - started) * 1000, 2),
        **context,
        **extra,
    )
