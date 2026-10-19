"""Pure-Python financial mathematics.

This module is the single source of truth for every number the system computes.
It has no MCP, no network and no LLM dependency, which is precisely the point:
language models are unreliable at arithmetic, so the agent is required to call
these functions instead of doing the maths in its head.

Every function either returns a fully-populated :class:`CalculationResult` or raises
:class:`CalculationError`. Nothing returns ``None`` and nothing silently returns 0.0
for an undefined quantity.
"""

import math

from models.calculations import (
    CalculationResult,
    CompanyMetric,
    ComparisonResult,
    RankedCompany,
    Unit,
)

# Presentation precision per unit. Rounding happens once, here, so the same inputs
# always produce byte-identical output across runs and processes.
_PRECISION: dict[Unit, int] = {"percent": 2, "times": 2, "ratio": 4, "years": 2}


class CalculationError(ValueError):
    """A calculation cannot be performed or would be financially meaningless.

    Raised for undefined maths (division by zero) and for inputs where the result
    would be numerically computable but misleading (e.g. growth from a negative base).
    """


def _check_finite(**values: float) -> None:
    """Reject NaN and infinity before they poison downstream arithmetic."""
    for name, value in values.items():
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise CalculationError(f"'{name}' must be a number, got {type(value).__name__}.")
        if math.isnan(value) or math.isinf(value):
            raise CalculationError(f"'{name}' must be a finite number, got {value}.")


def _require_nonzero(name: str, value: float, *, metric: str) -> None:
    """Guard a denominator. This is why division by zero never reaches the caller."""
    if value == 0:
        raise CalculationError(
            f"Cannot compute {metric}: '{name}' is zero, which makes the result undefined."
        )


def _build(
    *,
    metric: str,
    label: str,
    value: float,
    unit: Unit,
    formula: str,
    inputs: dict[str, float],
    interpretation: str,
    warnings: list[str] | None = None,
) -> CalculationResult:
    # Finite inputs can still overflow (1e308 / 1e-308). A result that is not a number
    # is undefined, not a very large number, so it is rejected rather than reported.
    if math.isnan(value) or math.isinf(value):
        raise CalculationError(
            f"Computing {label} from {inputs} overflowed to a non-finite value. "
            "Check the magnitude and units of the inputs."
        )

    return CalculationResult(
        metric=metric,
        label=label,
        value=round(value, _PRECISION[unit]),
        unit=unit,
        formula=formula,
        inputs=inputs,
        interpretation=interpretation,
        warnings=warnings or [],
    )


def _margin(
    *,
    metric: str,
    label: str,
    numerator_name: str,
    numerator: float,
    revenue: float,
) -> CalculationResult:
    """Shared implementation for the three margin metrics, which differ only in numerator."""
    _check_finite(**{numerator_name: numerator, "revenue": revenue})
    _require_nonzero("revenue", revenue, metric=label)
    if revenue < 0:
        raise CalculationError(
            f"Cannot compute {label}: revenue is negative ({revenue}), which is not a "
            "valid denominator. Check the units and sign convention of the source data."
        )

    value = numerator / revenue * 100
    warnings: list[str] = []
    if numerator < 0:
        warnings.append(
            f"{numerator_name.replace('_', ' ').title()} is negative, so the company is "
            "loss-making on this measure."
        )

    direction = "keeps" if value >= 0 else "loses"
    return _build(
        metric=metric,
        label=label,
        value=value,
        unit="percent",
        formula=f"({numerator_name} / revenue) * 100",
        inputs={numerator_name: numerator, "revenue": revenue},
        interpretation=(
            f"The company {direction} {abs(round(value, 2))} cents of "
            f"{numerator_name.replace('_', ' ')} per dollar of revenue."
        ),
        warnings=warnings,
    )


def calculate_revenue_growth(previous_revenue: float, current_revenue: float) -> CalculationResult:
    """Period-over-period revenue growth as a percentage.

    ``((current - previous) / previous) * 100``. Growth from a zero or negative base is
    undefined rather than infinite, so those inputs raise instead of returning a number.
    """
    _check_finite(previous_revenue=previous_revenue, current_revenue=current_revenue)
    _require_nonzero("previous_revenue", previous_revenue, metric="revenue growth")
    if previous_revenue < 0:
        raise CalculationError(
            "Cannot compute revenue growth from a negative base "
            f"({previous_revenue}); the percentage would have an inverted sign."
        )
    if current_revenue < 0:
        raise CalculationError(
            f"'current_revenue' is negative ({current_revenue}); revenue cannot be negative. "
            "Check whether the source data is a net figure or has a sign-convention issue."
        )

    value = (current_revenue - previous_revenue) / previous_revenue * 100
    warnings: list[str] = []
    if abs(previous_revenue) < abs(current_revenue) * 0.01:
        warnings.append(
            "The prior-period base is very small relative to the current period, so the "
            "growth percentage is exaggerated by a base effect."
        )

    direction = "grew" if value > 0 else "declined" if value < 0 else "was flat"
    return _build(
        metric="revenue_growth",
        label="Revenue Growth",
        value=value,
        unit="percent",
        formula="((current_revenue - previous_revenue) / previous_revenue) * 100",
        inputs={"previous_revenue": previous_revenue, "current_revenue": current_revenue},
        interpretation=f"Revenue {direction} {abs(round(value, 2))}% versus the prior period.",
        warnings=warnings,
    )


def calculate_profit_margin(net_income: float, revenue: float) -> CalculationResult:
    """Net profit margin: the share of revenue left after every expense, tax and interest."""
    return _margin(
        metric="profit_margin",
        label="Net Profit Margin",
        numerator_name="net_income",
        numerator=net_income,
        revenue=revenue,
    )


def calculate_gross_margin(gross_profit: float, revenue: float) -> CalculationResult:
    """Gross margin: the share of revenue left after the direct cost of goods sold."""
    return _margin(
        metric="gross_margin",
        label="Gross Margin",
        numerator_name="gross_profit",
        numerator=gross_profit,
        revenue=revenue,
    )


def calculate_operating_margin(operating_income: float, revenue: float) -> CalculationResult:
    """Operating margin: profitability of the core business, before interest and tax."""
    return _margin(
        metric="operating_margin",
        label="Operating Margin",
        numerator_name="operating_income",
        numerator=operating_income,
        revenue=revenue,
    )


def calculate_pe_ratio(price_per_share: float, earnings_per_share: float) -> CalculationResult:
    """Price-to-earnings ratio: how many dollars investors pay per dollar of annual earnings."""
    _check_finite(price_per_share=price_per_share, earnings_per_share=earnings_per_share)
    _require_nonzero("earnings_per_share", earnings_per_share, metric="P/E ratio")
    if price_per_share < 0:
        raise CalculationError(
            f"'price_per_share' is negative ({price_per_share}); a share price cannot be negative."
        )

    value = price_per_share / earnings_per_share
    warnings: list[str] = []
    if earnings_per_share < 0:
        warnings.append(
            "EPS is negative, so the P/E ratio is negative and not comparable to positive "
            "P/E ratios. Analysts normally treat P/E as not meaningful for loss-making companies."
        )

    return _build(
        metric="pe_ratio",
        label="P/E Ratio",
        value=value,
        unit="times",
        formula="price_per_share / earnings_per_share",
        inputs={"price_per_share": price_per_share, "earnings_per_share": earnings_per_share},
        interpretation=(
            f"Investors pay {round(value, 2)}x annual earnings per share."
            if earnings_per_share > 0
            else "P/E is not meaningful because earnings per share are negative."
        ),
        warnings=warnings,
    )


def calculate_debt_to_equity(total_debt: float, total_equity: float) -> CalculationResult:
    """Debt-to-equity: financial leverage, as debt carried per dollar of shareholder equity."""
    _check_finite(total_debt=total_debt, total_equity=total_equity)
    _require_nonzero("total_equity", total_equity, metric="debt-to-equity")
    if total_debt < 0:
        raise CalculationError(f"'total_debt' is negative ({total_debt}); debt cannot be negative.")

    value = total_debt / total_equity
    warnings: list[str] = []
    if total_equity < 0:
        warnings.append(
            "Shareholder equity is negative (often from large buybacks or accumulated losses), "
            "so debt-to-equity is negative and cannot be compared to normal positive ratios."
        )

    return _build(
        metric="debt_to_equity",
        label="Debt-to-Equity",
        value=value,
        unit="ratio",
        formula="total_debt / total_equity",
        inputs={"total_debt": total_debt, "total_equity": total_equity},
        interpretation=(
            f"The company carries ${round(value, 4)} of debt per dollar of equity."
            if total_equity > 0
            else "Debt-to-equity is not meaningful because shareholder equity is negative."
        ),
        warnings=warnings,
    )


def calculate_roe(net_income: float, shareholders_equity: float) -> CalculationResult:
    """Return on equity: profit generated per dollar of shareholder capital, as a percentage."""
    _check_finite(net_income=net_income, shareholders_equity=shareholders_equity)
    _require_nonzero("shareholders_equity", shareholders_equity, metric="return on equity")

    value = net_income / shareholders_equity * 100
    warnings: list[str] = []
    if shareholders_equity < 0:
        warnings.append(
            "Shareholder equity is negative, so ROE is not interpretable and should not be "
            "ranked against companies with positive equity."
        )
    if net_income < 0:
        warnings.append("Net income is negative, so the company destroyed equity value.")

    return _build(
        metric="roe",
        label="Return on Equity",
        value=value,
        unit="percent",
        formula="(net_income / shareholders_equity) * 100",
        inputs={"net_income": net_income, "shareholders_equity": shareholders_equity},
        interpretation=(
            f"The company earned {round(value, 2)}% on shareholder capital."
            if shareholders_equity > 0
            else "ROE is not meaningful because shareholder equity is negative."
        ),
        warnings=warnings,
    )


def calculate_roa(net_income: float, total_assets: float) -> CalculationResult:
    """Return on assets: how efficiently the asset base is converted into profit."""
    _check_finite(net_income=net_income, total_assets=total_assets)
    _require_nonzero("total_assets", total_assets, metric="return on assets")
    if total_assets < 0:
        raise CalculationError(
            f"'total_assets' is negative ({total_assets}); total assets cannot be negative."
        )

    value = net_income / total_assets * 100
    warnings = (
        ["Net income is negative, so return on assets is negative."] if net_income < 0 else []
    )

    return _build(
        metric="roa",
        label="Return on Assets",
        value=value,
        unit="percent",
        formula="(net_income / total_assets) * 100",
        inputs={"net_income": net_income, "total_assets": total_assets},
        interpretation=f"The company earned {round(value, 2)}% on its total asset base.",
        warnings=warnings,
    )


def calculate_cagr(beginning_value: float, ending_value: float, years: float) -> CalculationResult:
    """Compound annual growth rate: the constant yearly rate linking a start and end value."""
    _check_finite(beginning_value=beginning_value, ending_value=ending_value, years=years)
    if years <= 0:
        raise CalculationError(f"'years' must be greater than zero, got {years}.")
    if beginning_value <= 0:
        raise CalculationError(
            f"CAGR requires a positive beginning value, got {beginning_value}. "
            "Compound growth from zero or a negative base is undefined."
        )
    if ending_value < 0:
        raise CalculationError(f"CAGR requires a non-negative ending value, got {ending_value}.")

    value = ((ending_value / beginning_value) ** (1 / years) - 1) * 100
    warnings: list[str] = []
    if years < 1:
        warnings.append(
            f"The period is shorter than one year ({years}), so this annualises a partial "
            "period and will overstate volatility."
        )
    if ending_value == 0:
        warnings.append("The ending value is zero, giving a -100% CAGR.")

    return _build(
        metric="cagr",
        label="Compound Annual Growth Rate",
        value=value,
        unit="percent",
        formula="((ending_value / beginning_value) ** (1 / years) - 1) * 100",
        inputs={
            "beginning_value": beginning_value,
            "ending_value": ending_value,
            "years": years,
        },
        interpretation=(f"The value compounded at {round(value, 2)}% per year over {years} years."),
        warnings=warnings,
    )


def compare_financial_metrics(
    metric: str,
    companies: list[CompanyMetric],
    higher_is_better: bool = True,
    unit: Unit = "percent",
) -> ComparisonResult:
    """Rank the same metric across two or more companies, best-first.

    Ranking direction is explicit rather than inferred: margins and growth reward high
    values, while debt-to-equity and P/E reward low ones.
    """
    if len(companies) < 2:
        raise CalculationError(f"Comparison needs at least two companies, got {len(companies)}.")

    names = [c.company for c in companies]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise CalculationError(f"Duplicate companies in comparison: {sorted(duplicates)}.")

    _check_finite(**{f"{c.company}.value": c.value for c in companies})

    ordered = sorted(companies, key=lambda c: c.value, reverse=higher_is_better)
    best_value = ordered[0].value
    worst_value = ordered[-1].value
    values = [c.value for c in companies]

    ranked = [
        RankedCompany(
            rank=i + 1,
            company=c.company,
            value=c.value,
            difference_from_best=round(best_value - c.value, 4),
        )
        for i, c in enumerate(ordered)
    ]

    warnings: list[str] = []
    if len({c.value for c in companies}) < len(companies):
        warnings.append(
            "Two or more companies share the same value; ranking between them is arbitrary."
        )
    if any(c.value < 0 for c in companies) and any(c.value >= 0 for c in companies):
        warnings.append(
            "The comparison mixes positive and negative values, which often means the metric "
            "is not meaningful for at least one company."
        )

    direction = "highest" if higher_is_better else "lowest"
    return ComparisonResult(
        metric=metric,
        unit=unit,
        higher_is_better=higher_is_better,
        ranked=ranked,
        best=ordered[0].company,
        worst=ordered[-1].company,
        spread=round(abs(best_value - worst_value), 4),
        average=round(sum(values) / len(values), 4),
        interpretation=(
            f"{ordered[0].company} has the {direction} {metric} at {ordered[0].value}, "
            f"versus {ordered[-1].company} at {ordered[-1].value}."
        ),
        warnings=warnings,
    )
