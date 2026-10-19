"""State and output models for the financial analyst graph.

The state is deliberately explicit about *provenance*. Retrieved facts, calculated
metrics and model analysis are separate fields, never one blob of prose, so the report
generator can label each and the API can return them separately.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

# The report sections required of every analysis, in order.
REPORT_SECTIONS: tuple[str, ...] = (
    "Executive Summary",
    "Financial Performance",
    "Revenue Growth",
    "Profitability",
    "Valuation",
    "Recent News",
    "Risks",
    "Bull Case",
    "Bear Case",
    "Conclusion",
)


class CompanyRef(BaseModel):
    """A company the query is about."""

    name: str = Field(description="Company name as the user referred to it.")
    ticker: str | None = Field(default=None, description="Resolved ticker, when known.")


class QueryUnderstanding(BaseModel):
    """What the model decided the user is actually asking for."""

    companies: list[CompanyRef] = Field(default_factory=list)
    intents: list[str] = Field(
        default_factory=list,
        description="Analytical goals, e.g. 'revenue growth', 'valuation', 'recent news'.",
    )
    comparison: bool = Field(
        default=False, description="True when two or more companies must be compared."
    )
    needs_documents: bool = Field(
        default=False,
        description="True when the question refers to filings, annual reports or transcripts.",
    )
    restated_question: str = Field(
        default="", description="The question restated unambiguously, with pronouns resolved."
    )


class ResearchPlan(BaseModel):
    """The model's plan for what to gather before analysing."""

    steps: list[str] = Field(default_factory=list, description="Ordered research steps.")
    data_needed: list[str] = Field(
        default_factory=list, description="Specific figures required, e.g. 'NVDA FY2024 revenue'."
    )
    calculations_needed: list[str] = Field(
        default_factory=list,
        description="Metrics to compute with the calculator server, e.g. 'revenue growth NVDA'.",
    )


class ToolRecord(BaseModel):
    """One tool call, flattened for the API response and the UI."""

    server: str
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    ok: bool
    duration_ms: float
    error: str | None = None


class MetricRecord(BaseModel):
    """A number produced by the deterministic calculator, kept traceable to its inputs."""

    metric: str
    label: str
    value: float
    unit: str
    formula: str
    inputs: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    company: str | None = None


class AnalysisResponse(BaseModel):
    """The structured result of one analysis. This is what the API returns."""

    answer: str = Field(description="The final report, in markdown.")
    companies: list[CompanyRef] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    mcp_servers_used: list[str] = Field(default_factory=list)
    metrics: list[MetricRecord] = Field(
        default_factory=list, description="Deterministic calculations, with their inputs."
    )
    tool_calls: list[ToolRecord] = Field(
        default_factory=list, description="Every tool invocation, successful or not."
    )
    sources: list[str] = Field(
        default_factory=list, description="Article URLs and document references cited."
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Failed tools, missing data and other caveats the reader must see.",
    )
    conversation_id: str | None = None


class AnalystState(TypedDict, total=False):
    """State threaded through the graph.

    `messages` carries the conversation for follow-up context; everything else is
    per-analysis working state.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    query: str
    understanding: QueryUnderstanding
    plan: ResearchPlan
    evidence: list[str]
    metrics: list[MetricRecord]
    analysis: str
    risks: str
    report: str
    warnings: list[str]
    stage: Literal[
        "query_understanding",
        "research_planner",
        "data_collector",
        "financial_calculator",
        "financial_analyzer",
        "risk_analyzer",
        "report_generator",
        "done",
    ]
