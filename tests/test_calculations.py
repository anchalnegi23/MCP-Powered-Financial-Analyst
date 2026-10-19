"""Unit tests for the pure calculation layer: correctness and edge cases, no MCP involved."""

import math

import pytest

from models.calculations import CompanyMetric
from servers.calculator_mcp import calculations as calc
from servers.calculator_mcp.calculations import CalculationError

# ── Normal cases ──────────────────────────────────────────────────────────────


def test_revenue_growth_matches_worked_example():
    """The specification's worked example: 100 -> 120 must be exactly 20%."""
    result = calc.calculate_revenue_growth(previous_revenue=100, current_revenue=120)
    assert result.value == 20.0
    assert result.unit == "percent"
    assert result.metric == "revenue_growth"
    assert result.inputs == {"previous_revenue": 100, "current_revenue": 120}
    assert result.warnings == []


@pytest.mark.parametrize(
    ("previous", "current", "expected"),
    [
        (100, 120, 20.0),
        (100, 80, -20.0),
        (100, 100, 0.0),
        (26_974_000_000, 60_922_000_000, 125.85),  # NVDA FY2023 -> FY2024
        (0.5, 1.5, 200.0),
    ],
)
def test_revenue_growth_values(previous, current, expected):
    assert calc.calculate_revenue_growth(previous, current).value == expected


@pytest.mark.parametrize(
    ("fn", "kwargs", "expected"),
    [
        (calc.calculate_profit_margin, {"net_income": 25, "revenue": 100}, 25.0),
        (calc.calculate_gross_margin, {"gross_profit": 75, "revenue": 100}, 75.0),
        (calc.calculate_operating_margin, {"operating_income": 33, "revenue": 100}, 33.0),
        (calc.calculate_pe_ratio, {"price_per_share": 120, "earnings_per_share": 4}, 30.0),
        (calc.calculate_debt_to_equity, {"total_debt": 50, "total_equity": 200}, 0.25),
        (calc.calculate_roe, {"net_income": 20, "shareholders_equity": 100}, 20.0),
        (calc.calculate_roa, {"net_income": 15, "total_assets": 300}, 5.0),
        (calc.calculate_cagr, {"beginning_value": 100, "ending_value": 200, "years": 2}, 41.42),
    ],
)
def test_known_values(fn, kwargs, expected):
    assert fn(**kwargs).value == expected


def test_cagr_of_flat_series_is_zero():
    assert calc.calculate_cagr(beginning_value=500, ending_value=500, years=10).value == 0.0


def test_every_result_carries_its_own_audit_trail():
    """Traceability: the formula and the exact inputs travel with the number."""
    result = calc.calculate_roe(net_income=20, shareholders_equity=100)
    assert result.formula == "(net_income / shareholders_equity) * 100"
    assert result.inputs == {"net_income": 20, "shareholders_equity": 100}
    assert result.interpretation


def test_results_are_deterministic():
    a = calc.calculate_cagr(beginning_value=137, ending_value=911, years=3.5)
    b = calc.calculate_cagr(beginning_value=137, ending_value=911, years=3.5)
    assert a.model_dump() == b.model_dump()


# ── Division by zero: rejected, never crashes ─────────────────────────────────


@pytest.mark.parametrize(
    ("fn", "kwargs"),
    [
        (calc.calculate_revenue_growth, {"previous_revenue": 0, "current_revenue": 120}),
        (calc.calculate_profit_margin, {"net_income": 25, "revenue": 0}),
        (calc.calculate_gross_margin, {"gross_profit": 75, "revenue": 0}),
        (calc.calculate_operating_margin, {"operating_income": 33, "revenue": 0}),
        (calc.calculate_pe_ratio, {"price_per_share": 120, "earnings_per_share": 0}),
        (calc.calculate_debt_to_equity, {"total_debt": 50, "total_equity": 0}),
        (calc.calculate_roe, {"net_income": 20, "shareholders_equity": 0}),
        (calc.calculate_roa, {"net_income": 15, "total_assets": 0}),
        (calc.calculate_cagr, {"beginning_value": 0, "ending_value": 200, "years": 2}),
    ],
)
def test_zero_denominator_raises_instead_of_dividing(fn, kwargs):
    with pytest.raises(CalculationError):
        fn(**kwargs)


# ── Negative and nonsensical inputs ───────────────────────────────────────────


@pytest.mark.parametrize(
    ("fn", "kwargs"),
    [
        (calc.calculate_revenue_growth, {"previous_revenue": -100, "current_revenue": 120}),
        (calc.calculate_revenue_growth, {"previous_revenue": 100, "current_revenue": -120}),
        (calc.calculate_profit_margin, {"net_income": 25, "revenue": -100}),
        (calc.calculate_pe_ratio, {"price_per_share": -120, "earnings_per_share": 4}),
        (calc.calculate_debt_to_equity, {"total_debt": -50, "total_equity": 200}),
        (calc.calculate_roa, {"net_income": 15, "total_assets": -300}),
        (calc.calculate_cagr, {"beginning_value": -100, "ending_value": 200, "years": 2}),
        (calc.calculate_cagr, {"beginning_value": 100, "ending_value": -200, "years": 2}),
        (calc.calculate_cagr, {"beginning_value": 100, "ending_value": 200, "years": 0}),
        (calc.calculate_cagr, {"beginning_value": 100, "ending_value": 200, "years": -2}),
    ],
)
def test_invalid_inputs_are_rejected(fn, kwargs):
    with pytest.raises(CalculationError):
        fn(**kwargs)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_inputs_are_rejected(bad):
    with pytest.raises(CalculationError, match="finite"):
        calc.calculate_profit_margin(net_income=bad, revenue=100)


def test_booleans_are_not_accepted_as_numbers():
    with pytest.raises(CalculationError, match="must be a number"):
        calc.calculate_profit_margin(net_income=True, revenue=100)


# ── Legitimate-but-unusual inputs: computed, and flagged ──────────────────────


def test_loss_making_company_gets_negative_margin_and_a_warning():
    result = calc.calculate_profit_margin(net_income=-30, revenue=100)
    assert result.value == -30.0
    assert result.warnings and "loss-making" in result.warnings[0]


def test_negative_eps_yields_negative_pe_flagged_as_not_meaningful():
    result = calc.calculate_pe_ratio(price_per_share=50, earnings_per_share=-2)
    assert result.value == -25.0
    assert any("not meaningful" in w for w in result.warnings)
    assert "not meaningful" in result.interpretation


def test_negative_equity_is_computed_but_flagged():
    """Real companies (large buyback programmes) do run negative equity."""
    result = calc.calculate_debt_to_equity(total_debt=100, total_equity=-50)
    assert result.value == -2.0
    assert any("negative" in w for w in result.warnings)


def test_negative_equity_roe_is_flagged():
    result = calc.calculate_roe(net_income=10, shareholders_equity=-50)
    assert any("not interpretable" in w for w in result.warnings)


def test_tiny_base_growth_is_flagged_as_a_base_effect():
    result = calc.calculate_revenue_growth(previous_revenue=1, current_revenue=1000)
    assert result.value == 99900.0
    assert any("base effect" in w for w in result.warnings)


def test_sub_year_cagr_is_flagged():
    result = calc.calculate_cagr(beginning_value=100, ending_value=110, years=0.5)
    assert any("shorter than one year" in w for w in result.warnings)


def test_cagr_to_zero_is_total_loss():
    result = calc.calculate_cagr(beginning_value=100, ending_value=0, years=5)
    assert result.value == -100.0
    assert any("ending value is zero" in w for w in result.warnings)


def test_extreme_but_valid_magnitudes_stay_finite():
    result = calc.calculate_profit_margin(net_income=1e18, revenue=1e-6)
    assert math.isfinite(result.value)


# ── Comparison ────────────────────────────────────────────────────────────────


def test_comparison_ranks_higher_is_better():
    result = calc.compare_financial_metrics(
        metric="operating_margin",
        companies=[
            CompanyMetric(company="AMD", value=7.4),
            CompanyMetric(company="NVDA", value=54.1),
            CompanyMetric(company="INTC", value=-2.0),
        ],
    )
    assert [r.company for r in result.ranked] == ["NVDA", "AMD", "INTC"]
    assert result.best == "NVDA"
    assert result.worst == "INTC"
    assert result.spread == 56.1
    assert result.ranked[0].difference_from_best == 0.0
    assert result.ranked[1].difference_from_best == 46.7


def test_comparison_ranks_lower_is_better():
    result = calc.compare_financial_metrics(
        metric="debt_to_equity",
        companies=[CompanyMetric(company="A", value=1.8), CompanyMetric(company="B", value=0.3)],
        higher_is_better=False,
        unit="ratio",
    )
    assert result.best == "B"
    assert [r.rank for r in result.ranked] == [1, 2]


def test_comparison_average_is_exact():
    result = calc.compare_financial_metrics(
        metric="roe",
        companies=[CompanyMetric(company="A", value=10), CompanyMetric(company="B", value=20)],
    )
    assert result.average == 15.0


def test_comparison_requires_two_companies():
    with pytest.raises(CalculationError, match="at least two"):
        calc.compare_financial_metrics(
            metric="roe", companies=[CompanyMetric(company="A", value=10)]
        )


def test_comparison_rejects_duplicate_companies():
    with pytest.raises(CalculationError, match="Duplicate"):
        calc.compare_financial_metrics(
            metric="roe",
            companies=[CompanyMetric(company="A", value=10), CompanyMetric(company="A", value=20)],
        )


def test_comparison_flags_ties_and_sign_changes():
    tied = calc.compare_financial_metrics(
        metric="roe",
        companies=[CompanyMetric(company="A", value=10), CompanyMetric(company="B", value=10)],
    )
    assert any("same value" in w for w in tied.warnings)

    mixed = calc.compare_financial_metrics(
        metric="roe",
        companies=[CompanyMetric(company="A", value=10), CompanyMetric(company="B", value=-4)],
    )
    assert any("positive and negative" in w for w in mixed.warnings)


def test_comparison_rejects_non_finite_values():
    with pytest.raises(CalculationError, match="finite"):
        calc.compare_financial_metrics(
            metric="roe",
            companies=[
                CompanyMetric(company="A", value=10),
                CompanyMetric(company="B", value=float("inf")),
            ],
        )


def test_negative_net_income_roe_is_flagged():
    result = calc.calculate_roe(net_income=-40, shareholders_equity=200)
    assert result.value == -20.0
    assert any("destroyed equity value" in w for w in result.warnings)


def test_overflow_to_infinity_is_rejected_not_reported():
    """Finite inputs can still overflow. An infinite result is undefined, not very large."""
    with pytest.raises(CalculationError, match="overflowed"):
        calc.calculate_profit_margin(net_income=1e308, revenue=1e-308)
