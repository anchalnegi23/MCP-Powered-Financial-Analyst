"""News service: provider articles in, normalised domain articles out.

Two providers are supported. NewsAPI is used when NEWSAPI_API_KEY is configured because
it supports real keyword and date-range search; otherwise it falls back to the ticker
news feed the market data provider already gives us for free. The choice is made here,
so the MCP tools never branch on which provider answered.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

import httpx
import structlog

from models.common import normalize_ticker, parse_date
from models.news import NewsArticle, NewsSearchResult
from services.cache import cached
from services.config import get_settings
from services.errors import (
    ProviderError,
    ProviderTimeoutError,
    RateLimitError,
)
from services.providers import yfinance_provider as provider
from services.redaction import scrub

logger = structlog.get_logger("service.news")

NEWSAPI_URL = "https://newsapi.org/v2/everything"
MAX_ARTICLES = 50
MAX_DESCRIPTION_CHARS = 600


def _truncate(text: Any) -> str | None:
    if not isinstance(text, str) or not text.strip():
        return None
    cleaned = text.strip()
    if len(cleaned) <= MAX_DESCRIPTION_CHARS:
        return cleaned
    return cleaned[:MAX_DESCRIPTION_CHARS] + "…"


def validate_date_range(
    from_date: str | None, to_date: str | None
) -> tuple[date | None, date | None]:
    """Parse and sanity-check a date window before it reaches any provider."""
    start = parse_date(from_date) if from_date else None
    end = parse_date(to_date) if to_date else None

    if start and end and start > end:
        raise ValueError(
            f"from_date ({start}) is after to_date ({end}). Swap them or widen the window."
        )
    today = datetime.now(UTC).date()
    if start and start > today:
        raise ValueError(f"from_date ({start}) is in the future; no news exists yet.")
    return start, end


def _nested_url(container: Any) -> str | None:
    """Pull a URL out of yfinance's {"url": ...} wrapper objects."""
    if isinstance(container, dict):
        value = container.get("url")
        return str(value) if value else None
    return str(container) if isinstance(container, str) and container else None


def _parse_timestamp(raw: Any) -> datetime | None:
    """Accept an epoch number or an ISO string; anything else is treated as unknown."""
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int | float):
        return datetime.fromtimestamp(raw, tz=UTC)
    if isinstance(raw, str):
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _from_yfinance(item: dict[str, Any], ticker: str | None) -> NewsArticle | None:
    """Normalise one yfinance news item.

    The provider has two payload shapes (a flat legacy one and a nested `content` one),
    and items missing a title or URL are dropped rather than published without a citation.
    """
    raw_content = item.get("content")
    content: dict[str, Any] = raw_content if isinstance(raw_content, dict) else item

    url = (
        _nested_url(content.get("canonicalUrl"))
        or _nested_url(content.get("clickThroughUrl"))
        or content.get("link")
        or item.get("link")
    )
    title = content.get("title") or item.get("title")
    if not url or not title:
        return None

    publisher = content.get("provider")
    source = (
        publisher.get("displayName")
        if isinstance(publisher, dict)
        else publisher or item.get("publisher")
    )

    return NewsArticle(
        title=str(title),
        source=str(source or "Unknown"),
        url=str(url),
        published_at=_parse_timestamp(
            content.get("pubDate") or item.get("providerPublishTime")
        ),
        description=_truncate(content.get("summary") or content.get("description")),
        ticker=ticker,
    )


def _from_newsapi(item: dict[str, Any], ticker: str | None) -> NewsArticle | None:
    url, title = item.get("url"), item.get("title")
    if not url or not title:
        return None
    return NewsArticle(
        title=str(title),
        source=str((item.get("source") or {}).get("name") or "Unknown"),
        url=str(url),
        published_at=_parse_timestamp(item.get("publishedAt")),
        description=_truncate(item.get("description")),
        ticker=ticker,
    )


async def _newsapi_search(
    query: str, *, start: date | None, end: date | None, limit: int, ticker: str | None
) -> list[NewsArticle]:
    """Query NewsAPI. Raises our own error types; never leaks the key into a message."""
    settings = get_settings()
    params: dict[str, Any] = {
        "q": query,
        "pageSize": min(limit, 100),
        "sortBy": "publishedAt",
        "language": "en",
    }
    if start:
        params["from"] = start.isoformat()
    if end:
        params["to"] = end.isoformat()

    try:
        async with httpx.AsyncClient(timeout=settings.http_timeout_seconds) as http:
            # The key travels in a header, never in the URL, so it cannot end up in a log
            # line or an exception's request repr.
            response = await http.get(
                NEWSAPI_URL, params=params, headers={"X-Api-Key": settings.newsapi_api_key or ""}
            )
    except httpx.TimeoutException as exc:
        raise ProviderTimeoutError(f"The news provider timed out searching for '{query}'.") from exc
    except httpx.HTTPError as exc:
        raise ProviderError(f"The news provider failed: {scrub(str(exc))[:200]}") from exc

    if response.status_code == 429:
        raise RateLimitError("The news provider rate-limited the request. Retry later.")
    if response.status_code == 401:
        raise ProviderError("The news provider rejected the configured credentials.")
    if response.status_code >= 400:
        raise ProviderError(f"The news provider returned HTTP {response.status_code}.")

    payload = response.json()
    articles = [_from_newsapi(item, ticker) for item in payload.get("articles", [])]
    return [a for a in articles if a is not None][:limit]


async def _yfinance_news(ticker: str, limit: int) -> list[NewsArticle]:
    items = await provider.fetch_news(
        ticker, count=limit, timeout=get_settings().http_timeout_seconds
    )
    articles = [_from_yfinance(item, ticker) for item in items]
    return [a for a in articles if a is not None]


def _use_newsapi() -> bool:
    return bool(get_settings().newsapi_api_key)


def _sort(articles: list[NewsArticle]) -> list[NewsArticle]:
    """Newest first; articles with no timestamp sort last rather than being dropped."""
    return sorted(
        articles,
        key=lambda a: a.published_at or datetime.min.replace(tzinfo=UTC),
        reverse=True,
    )


@cached("news.search", "news")
async def search_company_news(
    query: str, ticker: str | None = None, limit: int = 10
) -> NewsSearchResult:
    """Search news by keyword, optionally scoped to a ticker."""
    if not query or not query.strip():
        raise ValueError("Search query must not be empty.")
    limit = max(1, min(limit, MAX_ARTICLES))
    symbol = normalize_ticker(ticker) if ticker else None

    notes: list[str] = []
    if _use_newsapi():
        articles = await _newsapi_search(
            query, start=None, end=None, limit=limit, ticker=symbol
        )
        source = "newsapi"
    else:
        if not symbol:
            raise ValueError(
                "Keyword-only search requires NEWSAPI_API_KEY to be configured. Supply a "
                "`ticker` instead to use the fallback provider."
            )
        # The fallback feed has no keyword search, so filter its ticker feed locally.
        candidates = await _yfinance_news(symbol, MAX_ARTICLES)
        terms = query.lower().split()
        articles = [
            a
            for a in candidates
            if any(t in f"{a.title} {a.description or ''}".lower() for t in terms)
        ][:limit]
        source = provider.SOURCE
        notes.append(
            "No news API key is configured, so this searched the ticker's recent headlines "
            "locally rather than the full news archive. Coverage is limited and recent."
        )

    if not articles:
        notes.append(
            f"No articles matched '{query}'"
            + (f" for {symbol}" if symbol else "")
            + ". This means no coverage was found, not that nothing happened."
        )

    return NewsSearchResult(
        query=query,
        ticker=symbol,
        articles=_sort(articles),
        article_count=len(articles),
        source=source,
        notes=notes,
    )


@cached("news.latest", "news")
async def get_latest_company_news(ticker: str, limit: int = 10) -> NewsSearchResult:
    """Fetch the most recent headlines for one company."""
    symbol = normalize_ticker(ticker)
    limit = max(1, min(limit, MAX_ARTICLES))

    articles = await _yfinance_news(symbol, limit)
    notes: list[str] = []
    if not articles:
        notes.append(
            f"No recent articles were found for {symbol}. This means no coverage was "
            "found, not that nothing happened."
        )

    return NewsSearchResult(
        query=f"latest news for {symbol}",
        ticker=symbol,
        articles=_sort(articles)[:limit],
        article_count=min(len(articles), limit),
        source=provider.SOURCE,
        notes=notes,
    )


@cached("news.by_date", "news")
async def search_news_by_date(
    query: str,
    from_date: str,
    to_date: str | None = None,
    ticker: str | None = None,
    limit: int = 10,
) -> NewsSearchResult:
    """Search news within an explicit date window."""
    if not query or not query.strip():
        raise ValueError("Search query must not be empty.")
    start, end = validate_date_range(from_date, to_date)
    limit = max(1, min(limit, MAX_ARTICLES))
    symbol = normalize_ticker(ticker) if ticker else None

    notes: list[str] = []
    if _use_newsapi():
        articles = await _newsapi_search(query, start=start, end=end, limit=limit, ticker=symbol)
        source = "newsapi"
    else:
        if not symbol:
            raise ValueError(
                "Date-range search without a news API key requires a `ticker` to scope the "
                "fallback provider's feed."
            )
        candidates = await _yfinance_news(symbol, MAX_ARTICLES)
        terms = query.lower().split()
        articles = [
            a
            for a in candidates
            if a.published_at
            and (start is None or a.published_at.date() >= start)
            and (end is None or a.published_at.date() <= end)
            and any(t in f"{a.title} {a.description or ''}".lower() for t in terms)
        ][:limit]
        source = provider.SOURCE
        notes.append(
            "No news API key is configured. The fallback provider only carries recent "
            "headlines, so older windows will return nothing even when coverage exists."
        )
        if start and start < datetime.now(UTC).date() - timedelta(days=30):
            notes.append(
                f"The requested window starts {start}, which is outside the fallback "
                "provider's retention. Configure NEWSAPI_API_KEY for historical search."
            )

    if not articles:
        notes.append(f"No articles matched '{query}' between {start} and {end or 'today'}.")

    return NewsSearchResult(
        query=query,
        ticker=symbol,
        articles=_sort(articles),
        article_count=len(articles),
        from_date=start,
        to_date=end,
        source=source,
        notes=notes,
    )
