"""Domain models for financial-document retrieval.

Every retrieval result carries its metadata — company, document type, year, quarter,
source — because an unattributed quotation from a filing is worthless in an analyst
report. The point of RAG here is not just to find text but to be able to cite it.
"""

from typing import Literal

from pydantic import BaseModel, Field

DocumentType = Literal[
    "10-K",
    "10-Q",
    "annual_report",
    "earnings_report",
    "investor_presentation",
    "earnings_transcript",
]

SUPPORTED_DOCUMENT_TYPES: tuple[str, ...] = (
    "10-K",
    "10-Q",
    "annual_report",
    "earnings_report",
    "investor_presentation",
    "earnings_transcript",
)


class DocumentMetadata(BaseModel):
    """What we know about a document, used both for filtering and for citation."""

    document_id: str = Field(description="Stable identifier for the document.")
    company: str = Field(description="Company name, e.g. 'NVIDIA Corporation'.")
    ticker: str = Field(description="Ticker symbol, e.g. 'NVDA'.")
    document_type: DocumentType
    year: int = Field(description="Fiscal year the document covers.", ge=1900, le=2200)
    quarter: int | None = Field(
        default=None, description="Fiscal quarter 1-4, for quarterly documents.", ge=1, le=4
    )
    title: str = Field(description="Human-readable document title.")
    source: str = Field(description="Where the document came from: URL, filing id or path.")


class DocumentChunk(BaseModel):
    """One retrieved passage, with its position and its provenance."""

    chunk_id: str
    text: str = Field(description="The passage itself, as indexed.")
    chunk_index: int = Field(description="Position of this chunk within its document.")
    metadata: DocumentMetadata
    relevance_score: float = Field(
        description="0-1, higher is more relevant. Derived from vector distance; it is a "
        "similarity heuristic, not a probability of correctness."
    )


class SearchResult(BaseModel):
    """The outcome of a semantic search over the document corpus."""

    query: str
    chunks: list[DocumentChunk] = Field(description="Matches, most relevant first.")
    chunk_count: int
    filters_applied: dict[str, str | int] = Field(
        default_factory=dict, description="The metadata filters actually used."
    )
    notes: list[str] = Field(
        default_factory=list,
        description="Why a result is empty or partial, so 'no match' is never mistaken "
        "for 'the document says nothing'.",
    )


class Document(BaseModel):
    """A whole indexed document, reassembled from its chunks."""

    metadata: DocumentMetadata
    text: str
    chunk_count: int
    truncated: bool = Field(
        default=False, description="True when the returned text was capped for size."
    )


class DocumentSummary(BaseModel):
    """A document listing entry: metadata without the body."""

    metadata: DocumentMetadata
    chunk_count: int


class DocumentListing(BaseModel):
    """The result of listing documents by company or year."""

    documents: list[DocumentSummary]
    document_count: int
    filters_applied: dict[str, str | int] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class IndexResult(BaseModel):
    """The outcome of indexing one document."""

    document_id: str
    chunks_indexed: int
    metadata: DocumentMetadata
