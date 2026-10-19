"""Domain models for market data.

These are *our* shapes, not the provider's. Every field is one we chose to support, so
swapping yfinance for another vendor changes only the service layer, and the agent never
sees a vendor-specific key like 'regularMarketPrice'.
"""

from datetime import date, datetime

from pydantic import BaseModel, Field

from models.common import Ticker


class StockQuote(BaseModel):
    """A point-in-time price snapshot for one symbol."""

    ticker: Ticker
    company_name: str | None = Field(default=None, description="Registered company name.")
    price: float = Field(description="Latest traded or last close price.")
    currency: str = Field(default="USD", description="ISO currency code of the price.")
    previous_close: float | None = None
    change: float | None = Field(default=None, description="Price change versus previous close.")
    change_percent: float | None = Field(default=None, description="Change as a percentage.")
    day_high: float | None = None
    day_low: float | None = None
    volume: int | None = None
    market_state: str | None = Field(
        default=None, description="Provider's market state, e.g. REGULAR, CLOSED, PRE."
    )
    as_of: datetime = Field(description="When this snapshot was retrieved (UTC).")
    source: str = Field(description="Data provider that supplied the figures.")


class PricePoint(BaseModel):
    """One daily OHLCV bar."""

    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int


class HistoricalPrices(BaseModel):
    """A time series of daily bars plus summary statistics computed over the window."""

    ticker: Ticker
    period: str = Field(description="Requested window, e.g. '1y'.")
    interval: str = Field(description="Bar size, e.g. '1d'.")
    currency: str = "USD"
    start_date: date
    end_date: date
    point_count: int = Field(description="Number of bars returned.")
    prices: list[PricePoint] = Field(description="Bars, oldest first.")
    period_high: float
    period_low: float
    period_return_percent: float = Field(
        description="Close-to-close return across the window, as a percentage."
    )
    truncated: bool = Field(
        default=False,
        description="True when the series was down-sampled or capped to limit response size.",
    )
    source: str


class CompanyProfile(BaseModel):
    """Descriptive, slow-changing facts about a listed company."""

    ticker: Ticker
    company_name: str
    sector: str | None = None
    industry: str | None = None
    country: str | None = None
    website: str | None = None
    employees: int | None = None
    exchange: str | None = None
    currency: str = "USD"
    description: str | None = Field(
        default=None, description="Business summary, truncated to keep responses small."
    )
    source: str


class MarketCap(BaseModel):
    """Market capitalisation and enterprise value."""

    ticker: Ticker
    company_name: str | None = None
    market_cap: float | None = Field(default=None, description="Shares outstanding x price.")
    enterprise_value: float | None = Field(
        default=None, description="Market cap plus net debt; None when the provider omits it."
    )
    shares_outstanding: float | None = None
    currency: str = "USD"
    as_of: datetime
    source: str


class StockStatistics(BaseModel):
    """Valuation and trading statistics used for screening and comparison."""

    ticker: Ticker
    company_name: str | None = None
    pe_ratio_trailing: float | None = None
    pe_ratio_forward: float | None = None
    price_to_book: float | None = None
    price_to_sales: float | None = None
    eps_trailing: float | None = None
    dividend_yield_percent: float | None = None
    beta: float | None = Field(default=None, description="Volatility versus the market.")
    fifty_two_week_high: float | None = None
    fifty_two_week_low: float | None = None
    fifty_day_average: float | None = None
    two_hundred_day_average: float | None = None
    average_volume: int | None = None
    currency: str = "USD"
    as_of: datetime
    missing_fields: list[str] = Field(
        default_factory=list,
        description="Statistics the provider did not supply. Explicitly listed so the "
        "agent can say 'not available' instead of inventing a value.",
    )
    source: str
