"""Caching for expensive, repeatable provider calls.

Financial data is a near-ideal caching target: a company profile changes a few times a
year, an annual income statement never changes once filed, and a price quote is stale in
seconds. So the TTL is per data kind, not global.

Two rules shape the design:

- **A cache failure is never an application failure.** Every operation degrades to a
  miss. Redis being down must slow the system, not break it.
- **Nothing user-specific or credential-bearing is cached.** Only provider responses
  keyed by their query parameters, so there is no cross-user leakage to design around.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import inspect
import json
import time
import typing
from typing import Any

import structlog

from services.config import get_settings

logger = structlog.get_logger("cache")

# Per-kind TTLs in seconds. A filed annual statement is immutable; a quote is not.
TTL_BY_KIND: dict[str, int] = {
    "quote": 60,
    "historical": 3600,
    "profile": 86_400,
    "statement": 21_600,
    "ratios": 21_600,
    "earnings": 21_600,
    "statistics": 900,
    "news": 900,
    "documents": 3600,
}
DEFAULT_TTL = 900

# Never cached: anything that could carry a credential or identify a person.
FORBIDDEN_KEY_PARTS = ("api_key", "apikey", "token", "secret", "password", "authorization")


class CacheBackend:
    """Interface implemented by both backends."""

    async def get(self, key: str) -> Any | None:  # pragma: no cover - interface
        raise NotImplementedError

    async def set(self, key: str, value: Any, ttl: int) -> None:  # pragma: no cover
        raise NotImplementedError

    async def delete(self, key: str) -> None:  # pragma: no cover
        raise NotImplementedError

    async def clear(self) -> None:  # pragma: no cover
        raise NotImplementedError


class MemoryCache(CacheBackend):
    """Process-local cache. The default when Redis is not configured or not reachable.

    Correct for a single process; it simply does not share between workers, which is why
    Redis is preferred in production.
    """

    def __init__(self, max_entries: int = 2000) -> None:
        self._store: dict[str, tuple[float, Any]] = {}
        self.max_entries = max_entries
        self.name = "memory"

    async def get(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if expires_at < time.time():
            self._store.pop(key, None)
            return None
        return value

    async def set(self, key: str, value: Any, ttl: int) -> None:
        if len(self._store) >= self.max_entries:
            # Evict the soonest-to-expire entry. Crude, but this is the fallback backend;
            # ponytail: swap for an LRU only if profiling says it matters.
            oldest = min(self._store, key=lambda k: self._store[k][0])
            self._store.pop(oldest, None)
        self._store[key] = (time.time() + ttl, value)

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)

    async def clear(self) -> None:
        self._store.clear()


class RedisCache(CacheBackend):
    """Redis-backed cache, shared across processes. Every operation degrades to a miss."""

    def __init__(self, client: Any) -> None:
        self.client = client
        self.name = "redis"

    async def get(self, key: str) -> Any | None:
        try:
            raw = await self.client.get(key)
        except Exception as exc:  # noqa: BLE001 - a cache failure is a miss, not an error
            logger.warning("cache_get_failed", error_type=type(exc).__name__)
            return None
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            logger.warning("cache_value_corrupt", key=key)
            return None

    async def set(self, key: str, value: Any, ttl: int) -> None:
        try:
            await self.client.set(key, json.dumps(value, default=str), ex=ttl)
        except Exception as exc:  # noqa: BLE001
            logger.warning("cache_set_failed", error_type=type(exc).__name__)

    async def delete(self, key: str) -> None:
        try:
            await self.client.delete(key)
        except Exception as exc:  # noqa: BLE001
            logger.warning("cache_delete_failed", error_type=type(exc).__name__)

    async def clear(self) -> None:
        try:
            await self.client.flushdb()
        except Exception as exc:  # noqa: BLE001
            logger.warning("cache_clear_failed", error_type=type(exc).__name__)


class Cache:
    """The application's cache handle: a backend, plus key construction and statistics."""

    def __init__(self, backend: CacheBackend | None = None) -> None:
        self.backend: CacheBackend = backend or MemoryCache()
        self.hits = 0
        self.misses = 0
        self.enabled = True

    @property
    def backend_name(self) -> str:
        return getattr(self.backend, "name", "unknown")

    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "backend": self.backend_name,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
            "enabled": self.enabled,
        }

    async def get(self, key: str) -> Any | None:
        if not self.enabled:
            return None
        value = await self.backend.get(key)
        if value is None:
            self.misses += 1
        else:
            self.hits += 1
        return value

    async def set(self, key: str, value: Any, *, kind: str = "default") -> None:
        if not self.enabled:
            return
        await self.backend.set(key, value, ttl_for(kind))

    async def clear(self) -> None:
        await self.backend.clear()
        self.hits = self.misses = 0


def ttl_for(kind: str) -> int:
    """TTL for a data kind. `CACHE_TTL_SECONDS=0` disables caching entirely."""
    configured = get_settings().cache_ttl_seconds
    if configured == 0:
        return 0
    return TTL_BY_KIND.get(kind, DEFAULT_TTL)


def make_key(namespace: str, **parameters: Any) -> str:
    """Build a cache key from a namespace and the call's parameters.

    Parameters are sorted and hashed, so `(ticker=NVDA, period=annual)` and
    `(period=annual, ticker=NVDA)` are the same key while NVDA and AMD never collide.
    Credential-shaped parameters are refused outright rather than silently dropped —
    a key containing an API key would end up in Redis in plaintext.
    """
    for name in parameters:
        if any(part in name.lower() for part in FORBIDDEN_KEY_PARTS):
            raise ValueError(
                f"Refusing to build a cache key containing '{name}': credentials must "
                "never be part of a cache key."
            )

    payload = json.dumps(
        {k: v for k, v in sorted(parameters.items()) if v is not None},
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha256(payload.encode()).hexdigest()[:24]
    return f"fa:{namespace}:{digest}"


cache = Cache()


async def build_cache(redis_url: str | None = None) -> Cache:
    """Return the application cache, preferring Redis, falling back to memory.

    Never raises: an unreachable Redis downgrades to a process-local cache.
    """
    url = redis_url or get_settings().redis_url
    try:
        import redis.asyncio as redis_asyncio

        client = redis_asyncio.from_url(url, decode_responses=True, socket_connect_timeout=3)
        await asyncio.wait_for(client.ping(), timeout=3)
        cache.backend = RedisCache(client)
        logger.info("cache_ready", backend="redis")
    except Exception as exc:  # noqa: BLE001 - degrade, do not fail
        cache.backend = MemoryCache()
        logger.warning(
            "cache_fallback", backend="memory", reason=f"{type(exc).__name__}: {str(exc)[:120]}"
        )
    return cache


def cached(namespace: str, kind: str = "default"):
    """Cache an async service function's result, keyed by its arguments.

    Applied at the service layer rather than inside the provider adapter, so what is
    cached is the *normalised domain object*: a provider swap does not invalidate it, and
    the cached value has a shape we control.
    """

    def decorator(fn):
        signature = inspect.signature(fn)
        # Resolved, not read from __annotations__: modules using
        # `from __future__ import annotations` store the return type as a *string*, and a
        # string has no model_validate, so cached values would come back as raw dicts.
        model_type = _resolved_return_type(fn)

        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any):
            if ttl_for(kind) == 0:  # caching disabled by configuration
                return await fn(*args, **kwargs)

            # Bind to the signature so f("NVDA") and f(ticker="NVDA") share a key.
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            key = make_key(namespace, **dict(bound.arguments))

            hit = await cache.get(key)
            if hit is not None:
                revived = _revive(hit, model_type, namespace)
                if revived is not None:
                    logger.info("cache_hit", namespace=namespace, kind=kind)
                    return revived

            result = await fn(*args, **kwargs)
            await cache.set(key, _serialise(result), kind=kind)
            return result

        wrapper.__wrapped__ = fn  # type: ignore[attr-defined]
        return wrapper

    return decorator


def _resolved_return_type(fn: Any) -> Any:
    """The function's actual return class, resolving PEP 563 string annotations."""
    try:
        return typing.get_type_hints(fn).get("return")
    except Exception:  # noqa: BLE001 - an unresolvable hint just means no revival
        return None


def _serialise(result: Any) -> Any:
    return result.model_dump(mode="json") if hasattr(result, "model_dump") else result


def _revive(payload: Any, model_type: Any, namespace: str) -> Any | None:
    """Rebuild a domain object from a cached payload.

    A payload whose shape no longer matches the model is treated as a miss rather than
    an error: after a deploy that changes a model, stale entries must not break requests.
    """
    if model_type is None or not hasattr(model_type, "model_validate"):
        return payload
    try:
        return model_type.model_validate(payload)
    except Exception:  # noqa: BLE001 - a stale shape is a miss
        logger.warning("cache_shape_stale", namespace=namespace)
        return None
