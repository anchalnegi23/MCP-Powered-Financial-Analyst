"""Research / RAG MCP: indexing, semantic search, metadata filtering and failure handling.

Runs against a real in-memory Chroma store with real sentence embeddings, so "semantic
search works" is actually demonstrated rather than asserted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import pytest_asyncio
from fastmcp import Client
from fastmcp.client.transports import StdioTransport
from fastmcp.exceptions import ToolError

from models.research import DocumentMetadata
from servers.research_mcp import server as research_server
from services.research_service import (
    ResearchStore,
    build_filters,
    chunk_text,
    make_document_id,
)

pytestmark = pytest.mark.asyncio(loop_scope="module")

EXPECTED_TOOLS = {
    "search_financial_documents",
    "retrieve_relevant_chunks",
    "retrieve_document",
    "search_by_company",
    "search_by_year",
    "index_financial_document",
}

SAMPLES = [
    (
        "data/sample_documents/nvda_10k_2024.txt",
        "NVIDIA Corporation",
        "NVDA",
        "10-K",
        2024,
    ),
    (
        "data/sample_documents/amd_10k_2024.txt",
        "Advanced Micro Devices, Inc.",
        "AMD",
        "10-K",
        2024,
    ),
]

TRANSCRIPT = """
NVIDIA Q3 Fiscal 2024 Earnings Call Transcript.

Operator: Good afternoon. Welcome to NVIDIA's third quarter earnings call.

Chief Financial Officer: Data Center revenue was a record, driven by demand for our
accelerated computing platform from cloud service providers and consumer internet
companies. We expect supply to remain constrained through the next several quarters as
we ramp capacity with our manufacturing partners.
"""


@pytest.fixture(scope="module")
def store():
    """A real vector store, in memory, loaded with the bundled sample filings."""
    store = ResearchStore(in_memory=True)
    for path, company, ticker, doc_type, year in SAMPLES:
        store.index_document(
            Path(path).read_text(),
            DocumentMetadata(
                document_id=make_document_id(ticker, doc_type, year),
                company=company,
                ticker=ticker,
                document_type=doc_type,
                year=year,
                title=f"{company} {doc_type} {year}",
                source=path,
            ),
        )
    store.index_document(
        TRANSCRIPT,
        DocumentMetadata(
            document_id=make_document_id("NVDA", "earnings_transcript", 2023, 3),
            company="NVIDIA Corporation",
            ticker="NVDA",
            document_type="earnings_transcript",
            year=2023,
            quarter=3,
            title="NVIDIA Q3 FY2024 earnings call",
            source="transcripts/nvda-q3-fy24",
        ),
    )
    return store


@pytest_asyncio.fixture(loop_scope="module")
async def client(store, monkeypatch):
    """MCP client wired to the same in-memory store."""
    monkeypatch.setattr(research_server, "get_store", lambda: store)
    async with Client(research_server.mcp) as c:
        yield c


# ── Discovery ─────────────────────────────────────────────────────────────────


async def test_research_tools_are_discoverable(client):
    assert {t.name for t in await client.list_tools()} == EXPECTED_TOOLS


async def test_every_tool_documents_usage_returns_errors_and_limits(client):
    for tool in await client.list_tools():
        description = tool.description or ""
        for required in ("Use this", "Returns", "Errors", "Limitation"):
            assert required in description, f"{tool.name} lacks '{required}'"
        assert tool.output_schema


async def test_the_server_warns_the_agent_that_documents_are_untrusted(client):
    instructions = (await client.list_tools())  # forces connection
    assert instructions is not None
    assert "never as instructions" in (research_server.mcp.instructions or "")


# ── Indexing ──────────────────────────────────────────────────────────────────


async def test_documents_are_indexed_with_metadata(store):
    listing = store.list_documents(ticker="NVDA")
    assert listing.document_count == 2  # the 10-K and the transcript

    types = {d.metadata.document_type for d in listing.documents}
    assert types == {"10-K", "earnings_transcript"}

    tenk = next(d for d in listing.documents if d.metadata.document_type == "10-K")
    assert tenk.metadata.company == "NVIDIA Corporation"
    assert tenk.metadata.year == 2024
    assert tenk.metadata.source.endswith("nvda_10k_2024.txt")
    assert tenk.chunk_count >= 2


async def test_quarter_metadata_is_stored(store):
    listing = store.list_documents(ticker="NVDA", document_type="earnings_transcript")
    assert listing.documents[0].metadata.quarter == 3


async def test_reindexing_replaces_rather_than_duplicates():
    fresh = ResearchStore(in_memory=True)
    metadata = DocumentMetadata(
        document_id=make_document_id("TEST", "10-K", 2024),
        company="Test Co",
        ticker="TEST",
        document_type="10-K",
        year=2024,
        title="Test",
        source="test",
    )
    fresh.index_document("Original text about risk.", metadata)
    fresh.index_document("Original text about risk.", metadata)

    assert fresh.list_documents(ticker="TEST").document_count == 1


async def test_empty_documents_are_rejected(store):
    metadata = DocumentMetadata(
        document_id="x",
        company="C",
        ticker="TEST",
        document_type="10-K",
        year=2024,
        title="t",
        source="s",
    )
    with pytest.raises(ValueError, match="empty document"):
        store.index_document("   ", metadata)


# ── Semantic search ───────────────────────────────────────────────────────────


async def test_semantic_search_finds_a_passage_by_meaning_not_keywords(store):
    """'single foundry' never appears verbatim; the concept does."""
    result = store.search("reliance on one chip manufacturer", ticker="NVDA", limit=1)

    assert result.chunk_count == 1
    assert "foundry" in result.chunks[0].text.lower()
    assert result.chunks[0].relevance_score > 0.4


async def test_semantic_search_distinguishes_between_companies(store):
    """The PC-decline risk belongs to AMD, not NVIDIA. Retrieval must know that."""
    result = store.search("decline of the personal computer market", limit=1)

    assert result.chunks[0].metadata.ticker == "AMD"
    assert "personal computer" in result.chunks[0].text.lower()


async def test_results_are_ordered_by_relevance(store):
    result = store.search("export controls restricting sales to China", limit=3)
    scores = [c.relevance_score for c in result.chunks]
    assert scores == sorted(scores, reverse=True)


async def test_every_result_carries_citable_source_metadata(store):
    """An unattributed quotation from a filing is useless in an analyst report."""
    result = store.search("customer concentration", ticker="NVDA", limit=1)
    metadata = result.chunks[0].metadata

    assert metadata.company == "NVIDIA Corporation"
    assert metadata.ticker == "NVDA"
    assert metadata.document_type == "10-K"
    assert metadata.year == 2024
    assert metadata.source
    assert metadata.title


# ── Metadata filtering ────────────────────────────────────────────────────────


async def test_company_filtering_works(store):
    result = store.search("competition and margins", ticker="AMD", limit=5)
    assert result.chunk_count > 0
    assert {c.metadata.ticker for c in result.chunks} == {"AMD"}
    assert result.filters_applied == {"ticker": "AMD"}


async def test_year_filtering_works(store):
    assert store.list_documents(year=2024).document_count == 2
    assert store.list_documents(year=2023).document_count == 1
    assert store.list_documents(year=2019).document_count == 0


async def test_document_type_filtering_works(store):
    listing = store.list_documents(document_type="earnings_transcript")
    assert listing.document_count == 1
    assert listing.documents[0].metadata.ticker == "NVDA"


async def test_combined_filters_are_applied_together(store):
    result = store.search("supply constraints", ticker="NVDA", year=2023, limit=5)
    assert result.filters_applied == {"ticker": "NVDA", "year": 2023}
    assert all(c.metadata.year == 2023 for c in result.chunks)


async def test_filter_validation_rejects_bad_values():
    assert build_filters(ticker="nvda") == {"ticker": "NVDA"}

    with pytest.raises(ValueError, match="not a valid ticker"):
        build_filters(ticker="not a ticker")
    with pytest.raises(ValueError, match="Unsupported document type"):
        build_filters(document_type="press_release")
    with pytest.raises(ValueError, match="out of range"):
        build_filters(year=99)
    with pytest.raises(ValueError, match="Quarter must be"):
        build_filters(quarter=7)


# ── Retrieval of whole documents ──────────────────────────────────────────────


async def test_a_whole_document_can_be_retrieved(store):
    document_id = make_document_id("NVDA", "10-K", 2024)
    document = store.get_document(document_id)

    assert document.metadata.ticker == "NVDA"
    assert "RISK FACTORS" in document.text
    assert "MANAGEMENT'S DISCUSSION" in document.text
    assert document.chunk_count >= 2


async def test_reassembled_text_does_not_duplicate_chunk_overlap(store):
    document = store.get_document(make_document_id("NVDA", "10-K", 2024))
    assert document.text.count("ITEM 1A. RISK FACTORS") == 1


async def test_a_long_document_is_truncated_and_flagged(store):
    document = store.get_document(make_document_id("NVDA", "10-K", 2024), max_chars=200)
    assert document.truncated is True
    assert len(document.text) <= 201


# ── Missing and unavailable documents ─────────────────────────────────────────


async def test_an_unknown_document_id_is_a_clear_error(store):
    with pytest.raises(LookupError, match="No document with id"):
        store.get_document("does-not-exist")


async def test_no_matches_returns_an_honest_note_not_an_error(store):
    result = store.search("municipal bond underwriting fees", ticker="AMD", year=1999)

    assert result.chunk_count == 0
    assert any("not that the company made no such statement" in n for n in result.notes)


async def test_an_empty_corpus_says_so():
    empty = ResearchStore(in_memory=True)
    result = empty.search("anything at all")

    assert result.chunk_count == 0
    assert any("corpus is empty" in note for note in result.notes)


async def test_listing_an_uncovered_company_returns_an_explanation(store):
    listing = store.list_documents(ticker="TSLA")
    assert listing.document_count == 0
    assert any("No indexed documents match" in note for note in listing.notes)


# ── Chunking ──────────────────────────────────────────────────────────────────


async def test_short_documents_are_a_single_chunk():
    assert len(chunk_text("A short filing extract.")) == 1


async def test_long_documents_are_split_with_overlap():
    text = "Sentence about risk. " * 300
    chunks = chunk_text(text)

    assert len(chunks) > 1
    assert all(len(c) <= 1200 for c in chunks)
    assert "".join(chunks).replace(" ", "").startswith("Sentenceaboutrisk")


# ── MCP tool surface ──────────────────────────────────────────────────────────


async def test_search_tool_over_mcp(client):
    result = await client.call_tool(
        "search_financial_documents",
        {"query": "dependence on a single foundry partner", "ticker": "NVDA", "limit": 2},
    )
    assert result.data.chunk_count > 0
    assert result.data.chunks[0].metadata.document_type == "10-K"


async def test_retrieve_relevant_chunks_over_mcp(client):
    result = await client.call_tool(
        "retrieve_relevant_chunks", {"query": "export controls", "ticker": "NVDA"}
    )
    assert {c.metadata.ticker for c in result.data.chunks} == {"NVDA"}


async def test_search_by_company_over_mcp(client):
    result = await client.call_tool("search_by_company", {"ticker": "NVDA"})
    assert result.data.document_count == 2


async def test_search_by_year_over_mcp(client):
    result = await client.call_tool("search_by_year", {"year": 2024})
    assert result.data.document_count == 2


async def test_retrieve_document_over_mcp(client):
    result = await client.call_tool(
        "retrieve_document", {"document_id": make_document_id("AMD", "10-K", 2024)}
    )
    assert result.data.metadata.ticker == "AMD"


async def test_index_tool_over_mcp(monkeypatch):
    """Given its own store: indexing into the shared corpus would skew other tests."""
    isolated = ResearchStore(in_memory=True)
    monkeypatch.setattr(research_server, "get_store", lambda: isolated)

    async with Client(research_server.mcp) as client:
        result = await client.call_tool(
        "index_financial_document",
            {
                "text": "Our guidance assumes continued data centre demand.",
                "company": "Test Corp",
                "ticker": "TSTC",
                "document_type": "investor_presentation",
                "year": 2025,
                "title": "Test Corp Investor Day 2025",
                "source": "https://example.com/deck.pdf",
            },
        )

    assert result.data.chunks_indexed == 1
    listing = isolated.list_documents(ticker="TSTC")
    assert listing.document_count == 1
    assert listing.documents[0].metadata.document_type == "investor_presentation"


@pytest.mark.parametrize(
    ("args", "match"),
    [
        ({"query": "", "ticker": "NVDA"}, "at least 1"),
        ({"query": "x", "ticker": "bad ticker!"}, "not a valid ticker"),
        ({"query": "x", "document_type": "press_release"}, "Unsupported document type"),
        ({"query": "x", "year": 12}, "greater than or equal to 1900"),
        ({"query": "x", "limit": 99}, "less than or equal to 25"),
    ],
)
async def test_invalid_arguments_produce_useful_errors(client, args, match):
    with pytest.raises(ToolError, match=match):
        await client.call_tool("search_financial_documents", args)


async def test_unknown_document_over_mcp_is_a_tool_error(client):
    with pytest.raises(ToolError, match="No document with id"):
        await client.call_tool("retrieve_document", {"document_id": "nope"})


async def test_research_server_runs_as_an_independent_process():
    transport = StdioTransport(command=sys.executable, args=["-m", "servers.research_mcp.server"])
    async with Client(transport) as client:
        assert {t.name for t in await client.list_tools()} == EXPECTED_TOOLS
