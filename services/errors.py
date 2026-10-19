"""Domain-level errors raised by the service layer.

The service layer's job is to make every provider failure look the same to the MCP
tools above it: a small, closed set of exceptions with readable messages. No provider
exception type, HTTP status code or vendor payload ever escapes past this boundary.
"""


class ProviderError(Exception):
    """Base class: something went wrong fetching data from an external provider."""


class TickerNotFoundError(ProviderError):
    """The symbol does not exist, is delisted, or the provider has no data for it."""


class ProviderTimeoutError(ProviderError):
    """The provider did not respond within the configured timeout."""


class RateLimitError(ProviderError):
    """The provider rejected the request because we exceeded its rate limit."""


class DataUnavailableError(ProviderError):
    """The symbol is valid but the specific data requested is missing or incomplete."""
