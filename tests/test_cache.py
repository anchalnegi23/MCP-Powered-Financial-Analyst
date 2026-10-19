"""Caching: correctness, isolation, TTLs, failure tolerance and measured speed-up."""

from __future__ import annotations

import asyncio
import time

import pytest

from models.market import StockQuote
from services import market_service
from services.cache import (
    Cache,
    MemoryCache,
    RedisCache,
    build_cache,
    cache,
    cached,
    make_key,
    ttl_for,
)
from services.config import get_settings
from services.errors import ProviderTimeoutError
from tests.conftest import NVDA_INFO, FakeTicker

pytestmark = pytest.mark.asyncio(loop_scope="module")


# ── Key construction ──────────────────────────────────────────────────────────


async def test_the_same_call_produces_the_same_key():
    assert make_key("market.quote", ticker="NVDA") == make_key("market.quote", ticker="NVDA")


async def test_argument_order_does_not_change_the_key():
    assert make_key("s", ticker="NVDA", period="annual") == make_key(
        "s", period="annual", ticker="NVDA"
    )


async def test_different_companies_never_share_a_key():
    """The collision that would matter most: serving AMD's data as NVIDIA's."""
    assert make_key("market.quote", ticker="NVDA") != make_key("market.quote", ticker="AMD")


async def test_different_parameters_never_share_a_key():
    assert make_key("f.statement", ticker="NVDA", period="annual") != make_key(
        "f.statement", ticker="NVDA", period="quarterly"
    )


async def test_different_namespaces_never_share_a_key():
    assert make_key("market.quote", ticker="NVDA") != make_key("market.profile", ticker="NVDA")


@pytest.mark.parametrize(
    "name", ["api_key", "apiKey", "access_token", "secret", "password", "authorization"]
)
async def test_credential_parameters_are_refused_in_a_key(name):
    """A key containing an API key would sit in Redis in plaintext."""
    with pytest.raises(ValueError, match="never be part of a cache key"):
        make_key("news", **{name: "sk-live-abc", "ticker": "NVDA"})


# ── TTL ───────────────────────────────────────────────────────────────────────


async def test_ttls_differ_by_data_kind():
    """A filed annual statement is stable; a quote is stale in a minute."""
    assert ttl_for("quote") < ttl_for("statement") < ttl_for("profile")


async def test_an_unknown_kind_gets_the_default_ttl():
    assert ttl_for("something_new") == 900


async def test_setting_the_ttl_to_zero_disables_caching(monkeypatch):
    monkeypatch.setenv("CACHE_TTL_SECONDS", "0")
    get_settings.cache_clear()
    try:
        assert ttl_for("quote") == 0

        calls = {"n": 0}

        @cached("test.disabled", "quote")
        async def fetch(ticker: str) -> dict:
            calls["n"] += 1
            return {"ticker": ticker}

        await fetch("NVDA")
        await fetch("NVDA")
        assert calls["n"] == 2  # every call went through
    finally:
        get_settings.cache_clear()


async def test_expired_entries_are_not_served():
    backend = MemoryCache()
    await backend.set("k", {"v": 1}, ttl=1)
    assert await backend.get("k") == {"v": 1}

    await backend.set("k", {"v": 1}, ttl=0)
    await asyncio.sleep(0.01)
    assert await backend.get("k") is None


# ── Caching behaviour ─────────────────────────────────────────────────────────


async def test_a_repeated_call_is_served_from_cache(patch_ticker):
    calls = {"n": 0}

    def factory(_symbol):
        calls["n"] += 1
        return FakeTicker(info=NVDA_INFO)

    patch_ticker(factory)

    first = await market_service.get_stock_price("NVDA")
    second = await market_service.get_stock_price("NVDA")

    assert calls["n"] == 1  # the provider was hit once
    assert first.price == second.price == 120.5
    assert cache.hits == 1


async def test_a_cached_value_is_revived_as_a_domain_model(patch_ticker):
    """A cache that returns raw dicts silently breaks every caller downstream."""
    patch_ticker(FakeTicker(info=NVDA_INFO))

    await market_service.get_stock_price("NVDA")
    second = await market_service.get_stock_price("NVDA")

    assert isinstance(second, StockQuote)
    assert second.ticker == "NVDA"


async def test_different_companies_do_not_serve_each_others_data(patch_ticker):
    responses = {
        "NVDA": {**NVDA_INFO, "currentPrice": 120.5, "longName": "NVIDIA Corporation"},
        "AMD": {**NVDA_INFO, "currentPrice": 160.0, "longName": "Advanced Micro Devices"},
    }
    patch_ticker(lambda symbol: FakeTicker(info=responses[symbol]))

    nvda = await market_service.get_stock_price("NVDA")
    amd = await market_service.get_stock_price("AMD")
    nvda_again = await market_service.get_stock_price("NVDA")

    assert nvda.price == 120.5
    assert amd.price == 160.0
    assert nvda_again.price == 120.5
    assert nvda_again.company_name == "NVIDIA Corporation"


async def test_a_stale_cached_shape_is_treated_as_a_miss():
    """After a deploy that changes a model, old entries must not break requests."""
    calls = {"n": 0}

    @cached("test.shape", "quote")
    async def fetch(ticker: str) -> StockQuote:
        calls["n"] += 1
        return StockQuote(
            ticker="NVDA", price=1.0, as_of="2025-01-01T00:00:00Z", source="test"
        )

    key = make_key("test.shape", ticker="NVDA")
    await cache.set(key, {"unexpected": "shape"}, kind="quote")

    result = await fetch("NVDA")
    assert isinstance(result, StockQuote)
    assert calls["n"] == 1


async def test_failures_are_not_cached(patch_ticker):
    """A cached error would turn a transient outage into a persistent one."""
    patch_ticker(FakeTicker(raises=TimeoutError("timed out")))
    with pytest.raises(ProviderTimeoutError):
        await market_service.get_stock_price("NVDA")

    patch_ticker(FakeTicker(info=NVDA_INFO))
    assert (await market_service.get_stock_price("NVDA")).price == 120.5


# ── Failure tolerance ─────────────────────────────────────────────────────────


class BrokenRedis:
    """A Redis client where every operation fails."""

    async def get(self, key):
        raise ConnectionError("redis down")

    async def set(self, key, value, ex=None):
        raise ConnectionError("redis down")

    async def delete(self, key):
        raise ConnectionError("redis down")

    async def flushdb(self):
        raise ConnectionError("redis down")


async def test_a_broken_cache_degrades_to_a_miss_rather_than_failing(patch_ticker):
    broken = Cache(RedisCache(BrokenRedis()))
    assert await broken.get("anything") is None
    await broken.set("k", {"v": 1}, kind="quote")  # must not raise
    await broken.clear()


async def test_the_application_still_serves_when_the_cache_is_broken(patch_ticker, monkeypatch):
    monkeypatch.setattr(cache, "backend", RedisCache(BrokenRedis()))
    patch_ticker(FakeTicker(info=NVDA_INFO))

    quote = await market_service.get_stock_price("NVDA")
    assert quote.price == 120.5  # answered from the provider


async def test_a_corrupt_cached_value_is_a_miss():
    class CorruptRedis(BrokenRedis):
        async def get(self, key):
            return "{not json"

    assert await RedisCache(CorruptRedis()).get("k") is None


async def test_unreachable_redis_falls_back_to_memory():
    built = await build_cache("redis://127.0.0.1:1/0")
    assert built.backend_name == "memory"


# ── Statistics and measured benefit ───────────────────────────────────────────


async def test_statistics_are_reported():
    cache.hits = cache.misses = 0
    await cache.get("missing-key")
    stats = cache.stats()

    assert stats["misses"] == 1
    assert stats["hit_rate"] == 0.0
    assert stats["backend"] == "memory"


async def test_caching_measurably_speeds_up_a_repeated_request(patch_ticker):
    """Demonstrated, not asserted: a slow provider is called once, not twice."""

    class SlowTicker(FakeTicker):
        @property
        def info(self):
            time.sleep(0.15)  # stands in for provider latency
            return NVDA_INFO

    patch_ticker(SlowTicker())

    started = time.perf_counter()
    await market_service.get_stock_price("NVDA")
    cold_ms = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    await market_service.get_stock_price("NVDA")
    warm_ms = (time.perf_counter() - started) * 1000

    assert cold_ms > 100, f"the slow provider was not actually exercised ({cold_ms:.1f}ms)"
    assert warm_ms < cold_ms / 5, f"cold {cold_ms:.1f}ms vs warm {warm_ms:.1f}ms"
