"""Domain models for deterministic financial calculations.

These are the *internal* representation returned by the Calculator MCP. They are
deliberately self-describing: every result carries the formula and the exact inputs
used, so an agent (or a human reading a report) can trace a number back to its source
instead of taking the LLM's word for it.
"""

from typing import Literal

from pydantic import BaseModel, Field

Unit = Literal["percent", "ratio", "times", "years"]


class CalculationResult(BaseModel):
    """A single deterministic metric computed in pure Python (no LLM involved)."""

    metric: str = Field(description="Machine-readable metric name, e.g. 'revenue_growth'.")
    label: str = Field(description="Human-readable metric name, e.g. 'Revenue Growth'.")
    value: float = Field(description="The computed value, rounded for presentation.")
    unit: Unit = Field(
        description="'percent' means the value is already x100 (20.0 == 20%); "
        "'ratio' and 'times' are plain multiples; 'years' is a duration."
    )
    formula: str = Field(description="The exact formula applied, for auditability.")
    inputs: dict[str, float] = Field(description="The exact inputs used, for auditability.")
    interpretation: str = Field(
        description="One-line plain-language reading of the number. Descriptive, not advice."
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Caveats that make the number unreliable or unusual "
        "(negative equity, negative earnings, tiny base effects). Empty when clean.",
    )


class CompanyMetric(BaseModel):
    """One company's value for a single metric, used as input to a comparison."""

    company: str = Field(description="Company name or ticker, e.g. 'NVDA'.")
    value: float = Field(description="The metric value for this company.")


class RankedCompany(BaseModel):
    """A company's placement in a metric comparison."""

    rank: int = Field(description="1 is best, given the metric's direction.")
    company: str
    value: float
    difference_from_best: float = Field(
        description="Best value minus this company's value, in the metric's own unit. "
        "0.0 for the top-ranked company."
    )


class ComparisonResult(BaseModel):
    """A deterministic side-by-side ranking of one metric across companies."""

    metric: str = Field(description="The metric being compared, e.g. 'operating_margin'.")
    unit: Unit
    higher_is_better: bool = Field(
        description="Direction used to rank. True for margins/growth, False for D/E or P/E."
    )
    ranked: list[RankedCompany] = Field(description="Companies sorted best-first.")
    best: str = Field(description="Company with the best value.")
    worst: str = Field(description="Company with the worst value.")
    spread: float = Field(description="Best value minus worst value, always >= 0.")
    average: float = Field(description="Arithmetic mean across all companies.")
    interpretation: str = Field(description="Plain-language summary. Descriptive, not advice.")
    warnings: list[str] = Field(default_factory=list)
