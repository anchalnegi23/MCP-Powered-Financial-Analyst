"""Financial safety and reliability guardrails.

Three distinct jobs, deliberately kept separate:

1. **Input validation** — reject malformed tickers and dates before they reach a provider.
2. **Tool-result validation** — detect provider responses that are structurally wrong or
   financially implausible, so a corrupted number never becomes a confident claim.
3. **Output review** — check the generated report for advice-shaped language, false
   certainty, and figures that no tool produced.

What this is not: a content filter. It does not rewrite the model's analysis. It
annotates, flags and, where a statement would be actively unsafe, refuses to let it stand
unqualified.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

import structlog

from models.common import normalize_ticker, parse_date

logger = structlog.get_logger("guardrails")

# Phrases that turn analysis into personalised advice. Matched as whole phrases so that
# quoting a risk factor ("customers should expect delays") does not trip them.
ADVICE_PATTERNS = [
    re.compile(r"\byou should (?:buy|sell|short|invest|purchase|dump|hold)\b", re.I),
    re.compile(r"\b(?:i|we) recommend (?:buying|selling|shorting|investing)\b", re.I),
    re.compile(r"\bthis (?:stock|company) is a (?:buy|sell|strong buy|strong sell)\b", re.I),
    re.compile(r"\byou (?:must|need to) (?:buy|sell|invest)\b", re.I),
    re.compile(r"\b(?:definitely|certainly) (?:buy|sell|invest in)\b", re.I),
    re.compile(r"\bguaranteed (?:return|profit|gain)s?\b", re.I),
    re.compile(r"\bcan't lose\b|\bcannot lose\b|\brisk[- ]free (?:investment|return)\b", re.I),
]

# Claims of certainty about an inherently uncertain future.
CERTAINTY_PATTERNS = [
    re.compile(r"\bwill (?:certainly|definitely|undoubtedly)\b", re.I),
    re.compile(r"\bis guaranteed to\b", re.I),
    re.compile(r"\b(?:price|stock) will (?:reach|hit|double|triple)\b", re.I),
    re.compile(r"\bthere is no risk\b", re.I),
    re.compile(r"\bwithout a doubt\b", re.I),
]

# Instruction-shaped text appearing inside retrieved documents or articles.
INJECTION_PATTERNS = [
    re.compile(r"ignore (?:all )?(?:your )?(?:previous|prior|above) instructions", re.I),
    re.compile(r"disregard (?:all )?(?:your )?(?:previous|prior) (?:instructions|rules)", re.I),
    re.compile(r"you are now (?:a|an|in) \w+", re.I),
    re.compile(r"\bsystem prompt\b|\bnew instructions?:\b", re.I),
    re.compile(r"do not (?:mention|disclose|reveal) (?:this|that|the)", re.I),
    re.compile(r"(?:always|instead) (?:recommend|say|report) that", re.I),
]

# Plausibility bounds per metric. Values outside these are not rejected — real outliers
# exist — but they are flagged, because they are far more often a data error.
PLAUSIBLE_RANGES: dict[str, tuple[float, float]] = {
    "revenue_growth": (-100.0, 1000.0),
    "profit_margin": (-500.0, 100.0),
    "gross_margin": (-500.0, 100.0),
    "operating_margin": (-500.0, 100.0),
    "pe_ratio": (-1000.0, 2000.0),
    "debt_to_equity": (-100.0, 100.0),
    "roe": (-1000.0, 1000.0),
    "roa": (-1000.0, 1000.0),
    "cagr": (-100.0, 500.0),
}


@dataclass
class GuardrailReport:
    """The outcome of reviewing something. `blocking` means it must not stand as written."""

    passed: bool = True
    blocking: bool = False
    findings: list[str] = field(default_factory=list)

    def add(self, finding: str, *, blocking: bool = False) -> None:
        self.findings.append(finding)
        self.passed = False
        if blocking:
            self.blocking = True


# ── 1. Input validation ───────────────────────────────────────────────────────


def validate_ticker_input(value: Any) -> str:
    """Validate a ticker at the trust boundary.

    The LLM chooses these values, so they are untrusted input, not internal data.
    """
    if not isinstance(value, str):
        raise ValueError(f"Ticker must be a string, got {type(value).__name__}.")
    return normalize_ticker(value)


def validate_date_input(value: Any) -> str:
    """Validate an ISO date at the trust boundary, returning the normalised form."""
    return parse_date(value).isoformat()


# ── 2. Tool-result validation ─────────────────────────────────────────────────


def validate_tool_result(tool: str, result: Any) -> GuardrailReport:
    """Check a tool response for structural and numerical sanity.

    Providers do return corrupted data: nulls where numbers belong, strings that should be
    floats, NaN from a division upstream. Catching it here stops it becoming a confident
    claim in a report.
    """
    report = GuardrailReport()

    if result is None:
        report.add(f"Tool '{tool}' returned no data at all.", blocking=True)
        return report

    if not isinstance(result, dict):
        report.add(
            f"Tool '{tool}' returned {type(result).__name__} where a structured object was "
            "expected. Do not treat it as data.",
            blocking=True,
        )
        return report

    for key, value in result.items():
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            report.add(
                f"Tool '{tool}' returned a non-finite value for '{key}'. Treat it as "
                "unavailable, not as a number.",
                blocking=True,
            )

    metric = result.get("metric")
    value = result.get("value")
    if isinstance(metric, str) and isinstance(value, int | float):
        bounds = PLAUSIBLE_RANGES.get(metric)
        if bounds and not bounds[0] <= float(value) <= bounds[1]:
            report.add(
                f"'{metric}' came back as {value}, outside the plausible range "
                f"{bounds[0]} to {bounds[1]}. Verify the inputs before reporting it."
            )

    if "formula" in result and not result.get("inputs"):
        report.add(
            f"Tool '{tool}' returned a calculated value with no inputs recorded, so it "
            "cannot be audited."
        )

    if report.findings:
        logger.warning("tool_result_flagged", tool=tool, findings=report.findings)
    return report


def scan_retrieved_text(text: str, *, origin: str) -> GuardrailReport:
    """Detect instruction-shaped content inside retrieved third-party text.

    Retrieved documents and news articles are data. If one contains something shaped like
    an instruction, that is a fact to report, not a command to obey.
    """
    report = GuardrailReport()
    for pattern in INJECTION_PATTERNS:
        match = pattern.search(text or "")
        if match:
            report.add(
                f"Text retrieved from {origin} contains instruction-like content "
                f"({match.group(0)!r}). It is quoted content and must not be followed."
            )
    if report.findings:
        logger.warning("possible_prompt_injection", origin=origin, count=len(report.findings))
    return report


# ── 3. Output review ──────────────────────────────────────────────────────────

_NUMBER = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+\.\d+)(?![\w])")


def review_output(
    report_text: str,
    *,
    known_values: set[str] | None = None,
) -> GuardrailReport:
    """Review a generated report for advice, false certainty and unsourced figures.

    `known_values` is the set of figures that came from tool results, rendered as strings.
    Any number in the report that is not among them and is not obviously a year or a small
    ordinal gets flagged for review rather than silently trusted.
    """
    report = GuardrailReport()
    text = report_text or ""

    for pattern in ADVICE_PATTERNS:
        match = pattern.search(text)
        if match:
            report.add(
                f"The report contains personalised investment advice: {match.group(0)!r}. "
                "State what the data shows instead of what the reader should do.",
                blocking=True,
            )

    for pattern in CERTAINTY_PATTERNS:
        match = pattern.search(text)
        if match:
            report.add(
                f"The report claims certainty about the future: {match.group(0)!r}. "
                "Financial outcomes are uncertain and must be described as such."
            )

    if known_values is not None:
        unsourced = _unsourced_numbers(text, known_values)
        if unsourced:
            report.add(
                "These figures appear in the report but match no tool result: "
                + ", ".join(sorted(unsourced)[:10])
                + ". Verify them or remove them."
            )

    if not _distinguishes_fact_from_analysis(text):
        report.add(
            "The report does not distinguish retrieved facts from calculated metrics and "
            "interpretation. Label them."
        )

    return report


def _unsourced_numbers(text: str, known_values: set[str]) -> set[str]:
    """Numbers in the report that no tool produced.

    Years, small integers and percentages that round to a known value are excluded: the
    aim is to catch invented financials, not to flag "the last 3 years".
    """
    found = {match.group(1) for match in _NUMBER.finditer(text)}
    normalised_known = {v.lstrip("-").rstrip("0").rstrip(".") for v in known_values}

    unsourced: set[str] = set()
    for raw in found:
        plain = raw.replace(",", "")
        try:
            number = float(plain)
        except ValueError:
            continue
        if 1900 <= number <= 2100 and "." not in plain:
            continue  # a year
        if abs(number) < 10 and "." not in plain:
            continue  # a small ordinal or count
        if any(plain.rstrip("0").rstrip(".") == k for k in normalised_known):
            continue
        # A figure rounded for presentation (125.85 -> 125.9) is the same figure.
        if any(
            math.isclose(number, float(k), rel_tol=0.001, abs_tol=0.05)
            for k in known_values
            if _is_number(k)
        ):
            continue
        unsourced.add(raw)
    return unsourced


def _is_number(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _distinguishes_fact_from_analysis(text: str) -> bool:
    """A report must signal provenance somewhere, or a reader cannot audit it."""
    markers = (
        "(calculated)",
        "(retrieved)",
        "calculated",
        "retrieved",
        "according to",
        "reported",
        "not available",
    )
    lowered = text.lower()
    return any(marker in lowered for marker in markers)


DISCLAIMER = (
    "\n\n---\n\n*This is analysis of retrieved financial data, not investment advice. "
    "Figures marked as calculated were computed deterministically from the inputs shown; "
    "figures marked as retrieved come from third-party data providers and may be delayed, "
    "revised or incomplete.*"
)


def apply_output_guardrails(report_text: str, review: GuardrailReport) -> str:
    """Annotate a report with any guardrail findings, and always append the disclaimer.

    Findings are appended visibly rather than used to silently rewrite the text: a reader
    who can see that a figure was flagged is better served than one shown a quietly
    edited report.
    """
    text = report_text.rstrip()

    if review.findings:
        heading = (
            "## ⚠️ Guardrail Review"
            if review.blocking
            else "## Guardrail Notes"
        )
        bullets = "\n".join(f"- {finding}" for finding in review.findings)
        text = f"{text}\n\n{heading}\n\n{bullets}"

    if "not investment advice" not in text.lower():
        text += DISCLAIMER
    return text
