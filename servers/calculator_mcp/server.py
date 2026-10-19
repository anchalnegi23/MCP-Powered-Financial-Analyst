"""Financial Calculator MCP server.

Exposes deterministic financial mathematics as MCP tools. This server is intentionally
*pure*: it makes no network calls, touches no database and needs no API key, so it can
be started and tested in complete isolation from the rest of the system.

Its reason to exist: large language models are unreliable at arithmetic. Any number that
appears in a report must come from one of these tools, not from the model's own
reasoning, so that the same inputs always produce the same figure.
"""

import os
import time
from typing import Annotated

import structlog
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import Field

from models.calculations import CalculationResult, CompanyMetric, ComparisonResult, Unit
from servers.calculator_mcp import calculations
from servers.calculator_mcp.calculations import CalculationError

logger = structlog.get_logger("calculator_mcp")

mcp = FastMCP(
    name="financial-calculator",
    instructions=(
        "Deterministic financial calculations. Always use these tools instead of computing "
        "ratios, margins or growth rates yourself: they are exact, reproducible and they "
        "return the formula and inputs used so the figure can be audited. Feed them values "
        "retrieved from the financial-statements or market-data servers."
    ),
)


def _run(tool: str, fn, /, **kwargs):
    """Invoke a pure calculation, converting domain errors into MCP tool errors.

    A bad input produces a ToolError that the agent can read and react to; it never
    propagates as an unhandled exception, so the server survives to answer the next call.
    """
    started = time.perf_counter()
    try:
        result = fn(**kwargs)
    except CalculationError as exc:
        logger.warning(
            "calculation_rejected",
            tool=tool,
            inputs=kwargs,
            reason=str(exc),
            duration_ms=round((time.perf_counter() - started) * 1000, 3),
        )
        raise ToolError(str(exc)) from exc
    logger.info(
        "calculation_completed",
        tool=tool,
        inputs=kwargs,
        duration_ms=round((time.perf_counter() - started) * 1000, 3),
    )
    return result


@mcp.tool
def calculate_revenue_growth(
    previous_revenue: Annotated[
        float, Field(description="Revenue in the earlier period. Must be greater than zero.")
    ],
    current_revenue: Annotated[
        float, Field(description="Revenue in the later period, in the same units and currency.")
    ],
) -> CalculationResult:
    """Compute period-over-period revenue growth as a percentage.

    Use this whenever a question involves revenue growth, sales growth, or "how fast is
    the company growing" between two specific periods (two years, two quarters). For a
    growth rate spanning more than two periods, use `calculate_cagr` instead.

    Formula: ((current_revenue - previous_revenue) / previous_revenue) * 100

    Returns a CalculationResult where `value` is the growth percentage already multiplied
    by 100 (so 20.0 means +20%), alongside the formula and inputs used.

    Errors: raises if `previous_revenue` is zero (growth from nothing is undefined) or
    negative (the sign of the percentage would invert), or if either input is not finite.
    Both revenues must be expressed in the same currency and scale.
    """
    return _run(
        "calculate_revenue_growth",
        calculations.calculate_revenue_growth,
        previous_revenue=previous_revenue,
        current_revenue=current_revenue,
    )


@mcp.tool
def calculate_profit_margin(
    net_income: Annotated[
        float, Field(description="Net income (bottom-line profit). May be negative for a loss.")
    ],
    revenue: Annotated[float, Field(description="Total revenue for the same period.")],
) -> CalculationResult:
    """Compute net profit margin: the percentage of revenue that survives as net income.

    Use this for questions about overall profitability, "how profitable is the company",
    or when comparing bottom-line efficiency between companies. This is the broadest
    margin measure: it is after cost of goods, operating expenses, interest and tax.
    For core-business profitability that excludes financing and tax effects, prefer
    `calculate_operating_margin`.

    Formula: (net_income / revenue) * 100

    Returns a CalculationResult with `value` as a percentage. A negative value means the
    company is loss-making, and a warning is attached saying so.

    Errors: raises if revenue is zero or negative. A negative `net_income` is accepted,
    since losses are a legitimate result.
    """
    return _run(
        "calculate_profit_margin",
        calculations.calculate_profit_margin,
        net_income=net_income,
        revenue=revenue,
    )


@mcp.tool
def calculate_gross_margin(
    gross_profit: Annotated[
        float, Field(description="Gross profit, i.e. revenue minus cost of goods sold.")
    ],
    revenue: Annotated[float, Field(description="Total revenue for the same period.")],
) -> CalculationResult:
    """Compute gross margin: the percentage of revenue left after direct production costs.

    Use this to assess pricing power and unit economics, and when comparing companies in
    the same industry. Gross margin is usually the most stable and most comparable margin
    across peers, because it excludes discretionary operating spend.

    Formula: (gross_profit / revenue) * 100

    Returns a CalculationResult with `value` as a percentage.

    Errors: raises if revenue is zero or negative. If you have cost of goods sold rather
    than gross profit, subtract it from revenue first and pass the difference.
    """
    return _run(
        "calculate_gross_margin",
        calculations.calculate_gross_margin,
        gross_profit=gross_profit,
        revenue=revenue,
    )


@mcp.tool
def calculate_operating_margin(
    operating_income: Annotated[
        float,
        Field(
            description="Operating income (EBIT): profit from core operations, before "
            "interest and tax. May be negative."
        ),
    ],
    revenue: Annotated[float, Field(description="Total revenue for the same period.")],
) -> CalculationResult:
    """Compute operating margin: profitability of the core business before interest and tax.

    Use this when the question is about operational efficiency or cost discipline, or when
    comparing companies with different capital structures and tax situations — operating
    margin strips out both, so it isolates how well the actual business runs.

    Formula: (operating_income / revenue) * 100

    Returns a CalculationResult with `value` as a percentage.

    Errors: raises if revenue is zero or negative. Negative operating income is accepted
    and flagged with a warning.
    """
    return _run(
        "calculate_operating_margin",
        calculations.calculate_operating_margin,
        operating_income=operating_income,
        revenue=revenue,
    )


@mcp.tool
def calculate_pe_ratio(
    price_per_share: Annotated[
        float, Field(description="Current market price of one share. Must not be negative.")
    ],
    earnings_per_share: Annotated[
        float,
        Field(
            description="Earnings per share over the trailing twelve months, or the "
            "forward estimate. Must not be zero. May be negative."
        ),
    ],
) -> CalculationResult:
    """Compute the price-to-earnings ratio: dollars paid per dollar of annual earnings.

    Use this for valuation questions — "is the stock expensive", "how does its valuation
    compare to a peer". Pair the market price from the market-data server with EPS from
    the financial-statements server, and be explicit about whether EPS is trailing or
    forward, because the two are not comparable.

    Formula: price_per_share / earnings_per_share

    Returns a CalculationResult with `unit` of "times" (e.g. 30.0 means 30x earnings).

    Errors: raises if `earnings_per_share` is zero (undefined) or `price_per_share` is
    negative. Negative EPS returns a negative P/E with a warning: analysts treat P/E as
    not meaningful for loss-making companies, and it must not be ranked against positive
    P/E ratios.

    Limitation: P/E is only comparable between companies in similar industries and growth
    stages. It says nothing about debt, which two companies may carry very differently.
    """
    return _run(
        "calculate_pe_ratio",
        calculations.calculate_pe_ratio,
        price_per_share=price_per_share,
        earnings_per_share=earnings_per_share,
    )


@mcp.tool
def calculate_debt_to_equity(
    total_debt: Annotated[
        float, Field(description="Total debt (short-term plus long-term). Must not be negative.")
    ],
    total_equity: Annotated[
        float,
        Field(
            description="Total shareholders' equity. Must not be zero. May be negative for "
            "companies with large buybacks or accumulated deficits."
        ),
    ],
) -> CalculationResult:
    """Compute the debt-to-equity ratio: financial leverage and balance-sheet risk.

    Use this for risk analysis, solvency questions, and any comparison of how aggressively
    two companies are financed. Higher means more leverage, which amplifies both returns
    and the risk of distress. Take both inputs from the balance sheet for the same date.

    Formula: total_debt / total_equity

    Returns a CalculationResult with `unit` of "ratio" (e.g. 0.5 means 50 cents of debt
    per dollar of equity).

    Errors: raises if `total_equity` is zero (undefined) or `total_debt` is negative.
    Negative equity yields a negative ratio with a warning; it is not comparable to normal
    positive ratios and should not be ranked against them.

    Limitation: "total debt" definitions vary by data provider — some include operating
    leases, some do not. Compare like with like.
    """
    return _run(
        "calculate_debt_to_equity",
        calculations.calculate_debt_to_equity,
        total_debt=total_debt,
        total_equity=total_equity,
    )


@mcp.tool
def calculate_roe(
    net_income: Annotated[float, Field(description="Net income for the period. May be negative.")],
    shareholders_equity: Annotated[
        float, Field(description="Total shareholders' equity. Must not be zero.")
    ],
) -> CalculationResult:
    """Compute return on equity: profit generated per dollar of shareholder capital.

    Use this to judge how effectively management converts owners' capital into profit, and
    when comparing capital efficiency between companies. High ROE can come from genuine
    profitability or simply from heavy leverage, so read it next to `calculate_debt_to_equity`.

    Formula: (net_income / shareholders_equity) * 100

    Returns a CalculationResult with `value` as a percentage.

    Errors: raises if `shareholders_equity` is zero. Negative equity or negative income
    returns a value carrying an explicit warning that the figure is not interpretable.
    """
    return _run(
        "calculate_roe",
        calculations.calculate_roe,
        net_income=net_income,
        shareholders_equity=shareholders_equity,
    )


@mcp.tool
def calculate_roa(
    net_income: Annotated[float, Field(description="Net income for the period. May be negative.")],
    total_assets: Annotated[
        float, Field(description="Total assets from the balance sheet. Must be greater than zero.")
    ],
) -> CalculationResult:
    """Compute return on assets: how efficiently the asset base is converted into profit.

    Use this to compare asset efficiency, especially between capital-intensive businesses.
    Unlike ROE, it is not inflated by leverage, so ROE far above ROA indicates the company
    is heavily financed by debt.

    Formula: (net_income / total_assets) * 100

    Returns a CalculationResult with `value` as a percentage.

    Errors: raises if `total_assets` is zero or negative. Negative net income is accepted
    and flagged.
    """
    return _run(
        "calculate_roa",
        calculations.calculate_roa,
        net_income=net_income,
        total_assets=total_assets,
    )


@mcp.tool
def calculate_cagr(
    beginning_value: Annotated[
        float, Field(description="Value at the start of the period. Must be greater than zero.")
    ],
    ending_value: Annotated[
        float, Field(description="Value at the end of the period. Must not be negative.")
    ],
    years: Annotated[
        float,
        Field(
            description="Number of years between the two values. Must be greater than zero; "
            "fractional values such as 2.5 are allowed."
        ),
    ],
) -> CalculationResult:
    """Compute the compound annual growth rate between a starting and ending value.

    Use this for multi-year growth questions — "how fast has revenue grown over five
    years", long-run share-price appreciation, or smoothing a volatile series into one
    annualised rate. For a single period-over-period change, use `calculate_revenue_growth`.

    Formula: ((ending_value / beginning_value) ** (1 / years) - 1) * 100

    Returns a CalculationResult with `value` as an annualised percentage.

    Errors: raises if `beginning_value` is zero or negative (compound growth from a
    non-positive base is undefined), if `ending_value` is negative, or if `years` is zero
    or negative. Periods under one year are computed but flagged, since annualising a
    partial period overstates the rate.

    Limitation: CAGR describes only the two endpoints. It hides everything in between,
    including a collapse followed by a recovery.
    """
    return _run(
        "calculate_cagr",
        calculations.calculate_cagr,
        beginning_value=beginning_value,
        ending_value=ending_value,
        years=years,
    )


@mcp.tool
def compare_financial_metrics(
    metric: Annotated[
        str,
        Field(
            description="Name of the metric being compared, e.g. 'operating_margin'. "
            "Used for labelling only."
        ),
    ],
    companies: Annotated[
        list[CompanyMetric],
        Field(
            description="At least two entries of {company, value}. Every value must "
            "already be in the same unit — do not mix percentages with ratios."
        ),
    ],
    higher_is_better: Annotated[
        bool,
        Field(
            description="True for metrics where more is better (margins, growth, ROE). "
            "False for metrics where less is better (debt-to-equity, P/E)."
        ),
    ] = True,
    unit: Annotated[
        Unit, Field(description="Unit of the supplied values: percent, ratio, times or years.")
    ] = "percent",
) -> ComparisonResult:
    """Rank two or more companies on a single metric, best-first, deterministically.

    Use this after computing the same metric for each company, to turn separate figures
    into an explicit ranking with the gap between them. It avoids the model having to eyeball
    which number is larger, and it makes the ranking direction explicit rather than assumed.

    Returns a ComparisonResult with the ranked list, best and worst company, spread between
    them, and the arithmetic average.

    Errors: raises if fewer than two companies are supplied, if the same company appears
    twice, or if any value is not finite. Ties and mixed positive/negative values are
    ranked but flagged with a warning, because a ranking across a sign change is usually
    a sign the metric is not meaningful for one of the companies.
    """
    return _run(
        "compare_financial_metrics",
        calculations.compare_financial_metrics,
        metric=metric,
        companies=companies,
        higher_is_better=higher_is_better,
        unit=unit,
    )


def main() -> None:
    """Console entrypoint. Defaults to stdio; set MCP_TRANSPORT=http for a networked server."""
    if os.getenv("MCP_TRANSPORT", "stdio") == "stdio":
        mcp.run()
    else:
        mcp.run(
            transport="http",
            host=os.getenv("MCP_HOST", "0.0.0.0"),
            port=int(os.getenv("MCP_PORT", "8001")),
        )


if __name__ == "__main__":
    main()
