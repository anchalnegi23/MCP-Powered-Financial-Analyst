"""Design rules for the agent, enforced as tests. Synchronous: no graph is run."""

from pathlib import Path

from agent.graph import _ensure_sections
from agent.state import REPORT_SECTIONS
from tests.fakes import REPORT_WITH_ALL_SECTIONS


def test_no_keyword_routing_exists_in_the_agent():
    """The rule of Phase 7, enforced as a test rather than a promise.

    Tool choice must be the model's. If someone later adds `if "stock" in query`, this
    fails instead of quietly turning the agent into a router.
    """
    source = Path("agent/graph.py").read_text()
    lowered = source.lower()

    for banned in ("in query", 'in state["query"]', "query.lower()", "in user_query"):
        assert banned not in lowered, f"keyword routing found: {banned}"

    # No tool is named anywhere in the graph: the model picks from the bound list.
    for tool_name in ("get_stock_price", "get_income_statement", "calculate_revenue_growth"):
        assert tool_name not in source, f"graph hardcodes tool '{tool_name}'"


def test_a_missing_section_is_added_and_marked_unavailable():
    """A silently absent section reads as 'nothing to report'. It usually is not."""
    partial = "## Executive Summary\nGood.\n\n## Conclusion\nDone."
    completed = _ensure_sections(partial, ["market tool failed"])

    for section in REPORT_SECTIONS:
        assert f"## {section}" in completed
    assert "Not available" in completed
    assert "market tool failed" in completed


def test_a_complete_report_is_left_untouched():
    assert _ensure_sections(REPORT_WITH_ALL_SECTIONS, []) == REPORT_WITH_ALL_SECTIONS
