"""Financial News MCP: normalisation, empty results, date validation and the tool surface."""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta

import httpx
import pytest
import respx
from fastmcp import Client
from fastmcp.client.transports import StdioTransport
from fastmcp.exceptions import ToolError

from servers.news_mcp.server import mcp
from services import news_service
from services.config import get_settings
from services.errors import ProviderTimeoutError, RateLimitError
from tests.conftest import NVDA_NEWS_RAW, FakeTicker

EXPECTED_TOOLS = {"search_company_news", "get_latest_company_news", "search_news_by_date"}


@pytest.fixture(autouse=True)
def no_news_api_key(monkeypatch):
    """Default to the keyless fallback path; individual tests opt into NewsAPI."""
    monkeypatch.setenv("NEWSAPI_API_KEY", "")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
async def client():
    async with Client(mcp) as c:
        yield c


@pytest.fixture
def nvda_news(patch_ticker):
    patch_ticker(FakeTicker(news=NVDA_NEWS_RAW))


# ── Discovery ─────────────────────────────────────────────────────────────────


async def test_news_tools_are_discoverable(client):
    assert {t.name for t in await client.list_tools()} == EXPECTED_TOOLS


async def test_every_tool_documents_usage_returns_errors_and_limits(client):
    for tool in await client.list_tools():
        description = tool.description or ""
        for required in ("Use this", "Returns", "Errors", "Limitation"):
            assert required in description, f"{tool.name} lacks '{required}'"
        assert tool.output_schema


# ── Normalisation ─────────────────────────────────────────────────────────────


async def test_latest_news_returns_structured_articles(nvda_news):
    result = await news_service.get_latest_company_news("NVDA")

    assert result.article_count == 3  # the item with no URL is dropped
    first = result.articles[0]
    assert first.title == "Nvidia unveils next-generation data centre GPU"
    assert first.source == "Reuters"
    assert first.url == "https://example.com/nvidia-gpu-launch"
    assert first.published_at == datetime(2025, 3, 18, 14, 30, tzinfo=UTC)
    assert first.description
    assert first.ticker == "NVDA"


async def test_urls_are_preserved_verbatim(nvda_news):
    result = await news_service.get_latest_company_news("NVDA")
    urls = {a.url for a in result.articles}
    assert "https://example.com/nvidia-gpu-launch" in urls
    assert "https://example.com/nvidia-export-rules" in urls
    assert "https://example.com/nvidia-supplier" in urls


async def test_articles_without_a_url_are_dropped_not_published(nvda_news):
    """An article with no link cannot be cited, so it must not appear in a report."""
    result = await news_service.get_latest_company_news("NVDA")
    assert all(a.url for a in result.articles)
    assert "Headline with no link" not in {a.title for a in result.articles}


async def test_articles_are_sorted_newest_first_with_undated_last(nvda_news):
    result = await news_service.get_latest_company_news("NVDA")
    assert [a.source for a in result.articles] == ["Reuters", "Bloomberg", "MarketWatch"]
    assert result.articles[-1].published_at is None


async def test_both_provider_payload_shapes_are_handled(nvda_news):
    result = await news_service.get_latest_company_news("NVDA")
    assert {"Reuters", "MarketWatch"} <= {a.source for a in result.articles}


async def test_long_descriptions_are_truncated(patch_ticker):
    patch_ticker(
        FakeTicker(
            news=[
                {
                    "title": "Long piece",
                    "link": "https://example.com/long",
                    "publisher": "FT",
                    "content": {
                        "title": "Long piece",
                        "canonicalUrl": {"url": "https://example.com/long"},
                        "summary": "word " * 500,
                    },
                }
            ]
        )
    )
    result = await news_service.get_latest_company_news("NVDA")
    assert len(result.articles[0].description) <= news_service.MAX_DESCRIPTION_CHARS + 1


# ── Empty results ─────────────────────────────────────────────────────────────


async def test_empty_result_does_not_crash_and_explains_itself(patch_ticker):
    patch_ticker(FakeTicker(news=[]))
    result = await news_service.get_latest_company_news("NVDA")

    assert result.article_count == 0
    assert result.articles == []
    assert any("not that nothing happened" in note for note in result.notes)


async def test_keyword_search_with_no_matches_returns_a_note(nvda_news):
    result = await news_service.search_company_news("cryptocurrency mining", ticker="NVDA")
    assert result.article_count == 0
    assert any("No articles matched" in note for note in result.notes)


async def test_keyword_search_filters_the_fallback_feed(nvda_news):
    result = await news_service.search_company_news("export restrictions", ticker="NVDA")
    assert result.article_count == 1
    assert result.articles[0].source == "Bloomberg"


async def test_fallback_mode_is_disclosed(nvda_news):
    result = await news_service.search_company_news("export", ticker="NVDA")
    assert any("No news API key is configured" in note for note in result.notes)


# ── Validation ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("bad", ["", "   "])
async def test_empty_query_is_rejected(bad):
    with pytest.raises(ValueError, match="must not be empty"):
        await news_service.search_company_news(bad, ticker="NVDA")


async def test_keyword_only_search_without_a_key_explains_the_requirement():
    with pytest.raises(ValueError, match="NEWSAPI_API_KEY"):
        await news_service.search_company_news("AI chips")


@pytest.mark.parametrize(
    "bad", ["2025-13-01", "not-a-date", "01/02/2025", "2025-02-30", ""]
)
def test_invalid_dates_are_rejected(bad):
    with pytest.raises(ValueError):
        news_service.validate_date_range(bad or None, None) if bad else news_service.parse_date(bad)


def test_reversed_date_range_is_rejected():
    with pytest.raises(ValueError, match="after to_date"):
        news_service.validate_date_range("2025-06-01", "2025-01-01")


def test_future_start_date_is_rejected():
    future = (datetime.now(UTC).date() + timedelta(days=30)).isoformat()
    with pytest.raises(ValueError, match="in the future"):
        news_service.validate_date_range(future, None)


def test_valid_date_range_is_parsed():
    start, end = news_service.validate_date_range("2025-01-01", "2025-03-31")
    assert (start.isoformat(), end.isoformat()) == ("2025-01-01", "2025-03-31")


async def test_date_search_echoes_the_resolved_window(nvda_news):
    result = await news_service.search_news_by_date(
        "export", from_date="2025-03-01", to_date="2025-03-31", ticker="NVDA"
    )
    assert result.from_date.isoformat() == "2025-03-01"
    assert result.to_date.isoformat() == "2025-03-31"
    assert result.article_count == 1


async def test_old_window_without_a_key_is_explained(nvda_news):
    result = await news_service.search_news_by_date(
        "export", from_date="2020-01-01", to_date="2020-06-30", ticker="NVDA"
    )
    assert result.article_count == 0
    assert any("outside the fallback provider's retention" in n for n in result.notes)


# ── NewsAPI path ──────────────────────────────────────────────────────────────


@pytest.fixture
def newsapi(monkeypatch):
    """Enable the NewsAPI path with a fake key, and intercept the HTTP call with respx.

    respx intercepts at the transport layer rather than replacing httpx.AsyncClient, so
    the service's own client construction (timeouts, headers) is exercised for real.
    """
    monkeypatch.setenv("NEWSAPI_API_KEY", "test-key-SHOULD-NOT-LEAK")
    get_settings.cache_clear()

    with respx.mock(assert_all_called=False) as router:
        yield router.get(news_service.NEWSAPI_URL)


async def test_newsapi_articles_are_normalised(newsapi):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "status": "ok",
                "articles": [
                    {
                        "title": "Nvidia beats estimates",
                        "url": "https://news.example/nvda-beat",
                        "source": {"name": "Reuters"},
                        "publishedAt": "2025-02-26T21:05:00Z",
                        "description": "Data centre revenue rose sharply.",
                    },
                    {"title": "No URL here", "source": {"name": "X"}},
                ],
            },
        )

    newsapi.side_effect = handler
    result = await news_service.search_company_news("nvidia earnings", ticker="NVDA")

    assert result.source == "newsapi"
    assert result.article_count == 1  # the URL-less article is dropped
    assert result.articles[0].source == "Reuters"
    assert result.articles[0].url == "https://news.example/nvda-beat"


async def test_newsapi_key_travels_in_a_header_not_the_url(newsapi):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["header"] = request.headers.get("x-api-key")
        return httpx.Response(200, json={"articles": []})

    newsapi.side_effect = handler
    await news_service.search_company_news("nvidia", ticker="NVDA")

    assert "SHOULD-NOT-LEAK" not in seen["url"]
    assert seen["header"] == "test-key-SHOULD-NOT-LEAK"


async def test_newsapi_rate_limit_is_mapped(newsapi):
    newsapi.return_value = httpx.Response(429, json={"message": "rate limited"})
    with pytest.raises(RateLimitError):
        await news_service.search_company_news("nvidia", ticker="NVDA")


async def test_newsapi_timeout_is_mapped(newsapi):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    newsapi.side_effect = handler
    with pytest.raises(ProviderTimeoutError):
        await news_service.search_company_news("nvidia", ticker="NVDA")


async def test_newsapi_auth_failure_does_not_echo_the_key(newsapi):
    newsapi.return_value = httpx.Response(401, json={"message": "bad key"})
    with pytest.raises(Exception) as excinfo:
        await news_service.search_company_news("nvidia", ticker="NVDA")
    assert "SHOULD-NOT-LEAK" not in str(excinfo.value)


async def test_date_filters_are_forwarded_to_newsapi(newsapi):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["params"] = dict(request.url.params)
        return httpx.Response(200, json={"articles": []})

    newsapi.side_effect = handler
    await news_service.search_news_by_date(
        "nvidia", from_date="2025-01-01", to_date="2025-02-01", ticker="NVDA"
    )
    assert seen["params"]["from"] == "2025-01-01"
    assert seen["params"]["to"] == "2025-02-01"


# ── MCP tool surface ──────────────────────────────────────────────────────────


async def test_latest_news_tool_over_mcp(client, nvda_news):
    result = await client.call_tool("get_latest_company_news", {"ticker": "NVDA", "limit": 2})
    assert result.data.article_count == 2
    assert result.data.articles[0].url.startswith("https://")


async def test_search_tool_over_mcp(client, nvda_news):
    result = await client.call_tool(
        "search_company_news", {"query": "export restrictions", "ticker": "NVDA"}
    )
    assert result.data.article_count == 1


async def test_date_search_tool_over_mcp(client, nvda_news):
    result = await client.call_tool(
        "search_news_by_date",
        {"query": "export", "from_date": "2025-03-01", "to_date": "2025-03-31", "ticker": "NVDA"},
    )
    assert result.data.article_count == 1


@pytest.mark.parametrize(
    ("args", "match"),
    [
        ({"query": "", "ticker": "NVDA"}, "at least 1"),
        ({"query": "x", "ticker": "bad ticker!"}, "not a valid ticker"),
        ({"query": "x", "ticker": "NVDA", "limit": 0}, "greater than or equal to 1"),
        ({"query": "x", "ticker": "NVDA", "limit": 500}, "less than or equal to 50"),
    ],
)
async def test_invalid_parameters_produce_useful_errors(client, nvda_news, args, match):
    with pytest.raises(ToolError, match=match):
        await client.call_tool("search_company_news", args)


async def test_invalid_date_over_mcp_is_a_tool_error(client, nvda_news):
    with pytest.raises(ToolError, match="not a valid date"):
        await client.call_tool(
            "search_news_by_date", {"query": "x", "from_date": "18th March", "ticker": "NVDA"}
        )


async def test_provider_failure_over_mcp_is_a_tool_error(client, patch_ticker):
    patch_ticker(FakeTicker(raises=TimeoutError("timed out")))
    with pytest.raises(ToolError):
        await client.call_tool("get_latest_company_news", {"ticker": "NVDA"})


async def test_news_server_runs_as_an_independent_process():
    transport = StdioTransport(command=sys.executable, args=["-m", "servers.news_mcp.server"])
    async with Client(transport) as client:
        assert {t.name for t in await client.list_tools()} == EXPECTED_TOOLS


async def test_no_secrets_in_logs(client, nvda_news, monkeypatch, capsys):
    monkeypatch.setenv("NEWSAPI_API_KEY", "")
    with pytest.raises(ToolError):
        await client.call_tool("search_company_news", {"query": "x", "ticker": "bad ticker!"})
    captured = capsys.readouterr()
    assert "SHOULD-NOT-LEAK" not in captured.out + captured.err
