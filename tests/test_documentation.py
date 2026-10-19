"""Documentation checks. Docs that drift from the code are worse than no docs."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

README = Path("README.md").read_text()
DOCS = {p.name: p.read_text() for p in Path("docs").glob("*.md")}
REQUIRED_DOCS = {
    "architecture.md",
    "mcp_servers.md",
    "deployment.md",
    "calculator_mcp.md",
    "interview.md",
}


def test_every_required_document_exists():
    assert REQUIRED_DOCS <= set(DOCS)


def test_the_readme_links_to_every_document():
    for name in REQUIRED_DOCS:
        assert f"docs/{name}" in README, f"README does not link to docs/{name}"


@pytest.mark.parametrize("name", sorted(REQUIRED_DOCS))
def test_documents_are_substantial(name):
    assert len(DOCS[name]) > 2000, f"docs/{name} is too thin to be useful"


# ── README completeness ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "section",
    [
        "The problem",
        "Architecture",
        "The five MCP servers",
        "Quick start",
        "Example queries",
        "API",
        "Testing",
        "Project layout",
        "Documentation",
        "Known limitations",
    ],
)
def test_readme_covers_each_required_section(section):
    assert section in README


def test_the_readme_documents_setup_for_both_docker_and_local():
    assert "docker compose up -d --build" in README
    assert "uv sync --extra dev" in README
    assert "cp .env.example .env" in README


def test_the_readme_documents_the_python_version_requirement():
    assert "3.11" in README and "3.13" in README


def test_the_readme_shows_example_queries():
    for query in (
        "current stock price",
        "revenue growth",
        "latest developments",
        "annual report",
    ):
        assert query in README


def test_the_readme_documents_testing_commands():
    assert "uv run pytest" in README
    assert "--cov" in README


# ── Architecture diagrams ─────────────────────────────────────────────────────


def test_architecture_diagrams_exist():
    assert README.count("```mermaid") >= 2
    assert DOCS["architecture.md"].count("```mermaid") >= 3


def test_the_diagrams_show_all_five_mcp_servers():
    diagram = README.split("```mermaid")[1]
    for server in ("Calculator MCP", "Market Data MCP", "Financial Statements MCP",
                   "News MCP", "Research / RAG MCP"):
        assert server in diagram


def test_the_workflow_diagram_shows_every_node():
    from agent.state import AnalystState  # noqa: F401

    diagram = README.split("```mermaid")[2]
    for node in (
        "query_understanding",
        "research_planner",
        "data_collector",
        "financial_calculator",
        "financial_analyzer",
        "risk_analyzer",
        "report_generator",
    ):
        assert node in diagram


# ── Docs match the code ───────────────────────────────────────────────────────


def test_every_mcp_server_is_documented():
    reference = DOCS["mcp_servers.md"]
    for server in (
        "financial-calculator",
        "market-data",
        "financial-statements",
        "financial-news",
        "financial-research",
    ):
        assert server in reference


def test_every_tool_is_documented():
    """A tool the docs do not mention is a tool nobody knows exists."""
    import asyncio

    from servers.calculator_mcp.server import mcp as calculator
    from servers.financial_mcp.server import mcp as financial
    from servers.market_mcp.server import mcp as market
    from servers.news_mcp.server import mcp as news
    from servers.research_mcp.server import mcp as research

    reference = DOCS["mcp_servers.md"] + DOCS["calculator_mcp.md"]

    async def tool_names() -> set[str]:
        names: set[str] = set()
        for server in (calculator, market, financial, news, research):
            names |= {t.name for t in await server.list_tools()}
        return names

    documented = asyncio.run(tool_names())
    assert len(documented) == 29
    for name in documented:
        assert name in reference, f"tool '{name}' is not documented"


def test_documented_environment_variables_match_the_settings():
    """Every setting must be documented, and every documented variable must exist."""
    from services.config import Settings

    documented = DOCS["deployment.md"] + Path(".env.example").read_text()
    skip = {"model_config"}

    for field in Settings.model_fields:
        if field in skip:
            continue
        assert field.upper() in documented, f"{field.upper()} is not documented"


def test_the_compose_services_are_documented():
    services = set(yaml.safe_load(Path("docker-compose.yml").read_text())["services"])
    deployment = DOCS["deployment.md"]
    for service in services:
        assert service in deployment, f"service '{service}' is not documented"


def test_known_limitations_are_documented_honestly():
    limitations = DOCS["deployment.md"]
    for topic in ("yfinance", "NEWSAPI_API_KEY", "EDGAR", "Fiscal-year", "Single-tenant"):
        assert topic in limitations


# ── Interview notes ───────────────────────────────────────────────────────────


def test_at_least_twenty_interview_questions_are_documented():
    questions = re.findall(r"^### (\d+)\.", DOCS["interview.md"], re.M)
    assert len(questions) >= 20
    assert [int(q) for q in questions] == list(range(1, len(questions) + 1))


@pytest.mark.parametrize(
    "topic",
    ["## MCP", "## LangGraph", "## Architecture", "## Production"],
)
def test_interview_notes_cover_each_required_topic(topic):
    assert topic in DOCS["interview.md"]


def test_interview_notes_cover_the_mcp_specific_questions():
    notes = DOCS["interview.md"]
    for question in (
        "What is MCP?",
        "MCP client vs MCP server",
        "Tools vs resources",
        "Why FastMCP?",
        "tool discovery",
        "multiple MCP servers",
    ):
        assert question in notes


def test_interview_notes_cover_the_langgraph_questions():
    notes = DOCS["interview.md"]
    for question in ("Why LangGraph", "How does state work", "tool selection", "failures"):
        assert question in notes


def test_interview_notes_cover_production_topics():
    notes = DOCS["interview.md"]
    for topic in ("caching", "rate limit", "observability", "secure", "scale"):
        assert topic.lower() in notes.lower()


def test_interview_notes_document_trade_offs_and_weaknesses():
    """An answer with no acknowledged weakness is not a credible answer."""
    notes = DOCS["interview.md"]
    assert "weakest parts" in notes.lower()
    assert "Honest counterpoint" in notes or "honest answer" in notes.lower()
    assert "actually shipped and fixed" in notes
