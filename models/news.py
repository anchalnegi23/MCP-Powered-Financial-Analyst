"""Domain models for financial news.

News is untrusted third-party text. These models keep the fields we need for citation
(title, source, URL, timestamp) and nothing else, so that whatever a publisher puts in a
headline arrives as data with a visible provenance rather than as free-floating prose.
"""

from datetime import date, datetime

from pydantic import BaseModel, Field


class NewsArticle(BaseModel):
    """One news article about a company."""

    title: str = Field(description="Headline as published.")
    source: str = Field(description="Publisher name, e.g. 'Reuters'.")
    url: str = Field(description="Canonical link to the article, preserved verbatim.")
    published_at: datetime | None = Field(
        default=None, description="Publication time in UTC. Null when the provider omits it."
    )
    description: str | None = Field(
        default=None, description="Publisher's own summary or excerpt, truncated."
    )
    ticker: str | None = Field(default=None, description="Ticker this article was found under.")
    company: str | None = Field(default=None, description="Company name, when known.")


class NewsSearchResult(BaseModel):
    """The outcome of a news query, including the case where nothing matched."""

    query: str = Field(description="What was searched for.")
    ticker: str | None = None
    articles: list[NewsArticle] = Field(description="Matching articles, newest first.")
    article_count: int
    from_date: date | None = Field(default=None, description="Start of the window, if filtered.")
    to_date: date | None = Field(default=None, description="End of the window, if filtered.")
    source: str = Field(description="News provider used.")
    notes: list[str] = Field(
        default_factory=list,
        description="Explanations for an empty or partial result, so the agent can report "
        "'no coverage found' rather than inferring silence means nothing happened.",
    )
