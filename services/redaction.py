"""Scrubbing for text that may carry credentials.

Provider exceptions and third-party log lines routinely echo back the URL or header that
failed, which can include an API key. Anything derived from an external message passes
through :func:`scrub` before it reaches a log or the agent.
"""

import re

REDACTED = "[REDACTED]"

# A labelled credential, optionally preceded by an auth scheme:
#   apikey=abc123   |   Authorization: Bearer eyJ...   |   "password": "hunter2"
# The scheme is matched explicitly so that "Authorization: Bearer <token>" does not
# consume only the word "Bearer" and leave the token itself behind.
_LABELLED = re.compile(
    r"(?i)\b(api[_-]?key|apikey|access[_-]?token|token|secret|password|authorization|bearer)\b"
    r"(\"?\s*[=:]\s*|\s+)"  # optional closing quote: {"password": "..."}
    r"((?:bearer|basic|token)\s+)?"
    r"(\"?)([^\s\"&,}]+)"
)

# Bare vendor-style keys (sk-..., pk_live_...) with no labelling word nearby.
_BARE_KEY = re.compile(r"\b(?:sk|pk|rk)[-_][A-Za-z0-9\-_]{8,}\b")

# Credentials embedded in a connection URL: postgresql://user:password@host/db
_URL_CREDENTIALS = re.compile(r"(?i)\b([a-z][a-z0-9+.\-]*://)([^:/\s]+):([^@/\s]+)@")


def _replace_labelled(match: re.Match[str]) -> str:
    label, separator, scheme, quote, _value = match.groups()
    return f"{label}{separator}{scheme or ''}{quote}{REDACTED}"


def scrub(text: str) -> str:
    """Replace anything that looks like a credential with a placeholder.

    Deliberately over-eager: a redacted diagnostic is recoverable, a leaked key is not.
    """
    if not text:
        return text
    result = _LABELLED.sub(_replace_labelled, text)
    result = _BARE_KEY.sub(REDACTED, result)
    return _URL_CREDENTIALS.sub(lambda m: f"{m.group(1)}{m.group(2)}:{REDACTED}@", result)
