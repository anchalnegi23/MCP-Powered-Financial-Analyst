"""Financial document retrieval, backed by a Chroma vector store.

Design decisions worth stating:

- **Chunk-level indexing with document-level metadata.** A 10-K is far too large to
  retrieve whole, and an answer needs to cite a specific passage. Chunks carry their
  parent's metadata so any retrieved passage can be attributed.
- **Metadata filtering happens in the vector store, not afterwards.** Filtering after
  retrieval would silently return fewer results than asked for, or none at all, while
  looking like a relevance failure.
- **Retrieved text is untrusted.** Filings and transcripts are third-party documents;
  the MCP layer labels them as such so the agent treats them as quotations rather than
  instructions.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from collections.abc import Mapping
from typing import Any

import structlog

from models.common import normalize_ticker
from models.research import (
    SUPPORTED_DOCUMENT_TYPES,
    Document,
    DocumentChunk,
    DocumentListing,
    DocumentMetadata,
    DocumentSummary,
    IndexResult,
    SearchResult,
)
from services.config import get_settings

logger = structlog.get_logger("service.research")

COLLECTION_NAME = "financial_documents"

# ~1200 characters is roughly a long paragraph: big enough to carry an argument, small
# enough that a retrieved chunk is quotable rather than a wall of text.
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
MAX_DOCUMENT_CHARS = 20_000
MAX_RESULTS = 25


class ResearchStore:
    """A vector store of financial documents, with metadata filtering.

    `embedding_function` is injectable: production uses Chroma's default sentence
    embeddings, and tests can substitute a fast deterministic one.
    """

    def __init__(
        self,
        persist_directory: str | None = None,
        *,
        in_memory: bool = False,
        embedding_function: Any | None = None,
        collection_name: str | None = None,
    ) -> None:
        import chromadb

        self.persist_directory = persist_directory or get_settings().chroma_persist_dir
        if in_memory:
            # chromadb's EphemeralClient is process-wide, so two in-memory stores would
            # otherwise share data. A unique collection name keeps them isolated.
            self.client = chromadb.EphemeralClient()
            collection_name = collection_name or f"ephemeral_{uuid.uuid4().hex[:12]}"
        else:
            self.client = chromadb.PersistentClient(path=self.persist_directory)

        self.collection_name = collection_name or COLLECTION_NAME
        kwargs: dict[str, Any] = {
            "name": self.collection_name,
            "metadata": {"hnsw:space": "cosine"},
        }
        if embedding_function is not None:
            kwargs["embedding_function"] = embedding_function
        self.collection = self.client.get_or_create_collection(**kwargs)
        logger.info("research_store_ready", in_memory=in_memory, count=self.collection.count())

    # ── indexing ─────────────────────────────────────────────────────────────

    def index_document(self, text: str, metadata: DocumentMetadata) -> IndexResult:
        """Split a document into overlapping chunks and store them with their metadata."""
        if not text or not text.strip():
            raise ValueError("Cannot index an empty document.")

        chunks = chunk_text(text)
        flat = _flatten_metadata(metadata)

        self.collection.upsert(
            ids=[f"{metadata.document_id}::{i}" for i in range(len(chunks))],
            documents=chunks,
            metadatas=[{**flat, "chunk_index": i} for i in range(len(chunks))],
        )
        logger.info(
            "document_indexed",
            document_id=metadata.document_id,
            ticker=metadata.ticker,
            chunks=len(chunks),
        )
        return IndexResult(
            document_id=metadata.document_id, chunks_indexed=len(chunks), metadata=metadata
        )

    def clear(self) -> None:
        """Drop every indexed document. Used by tests and by a full re-index."""
        existing = self.collection.get(include=[])
        if existing["ids"]:
            self.collection.delete(ids=existing["ids"])

    # ── retrieval ────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        *,
        ticker: str | None = None,
        document_type: str | None = None,
        year: int | None = None,
        quarter: int | None = None,
        limit: int = 5,
    ) -> SearchResult:
        """Semantic search with optional metadata filters, applied in the vector store."""
        if not query or not query.strip():
            raise ValueError("Search query must not be empty.")
        limit = max(1, min(limit, MAX_RESULTS))

        filters = build_filters(
            ticker=ticker, document_type=document_type, year=year, quarter=quarter
        )
        where = _to_chroma_where(filters)

        if self.collection.count() == 0:
            return SearchResult(
                query=query,
                chunks=[],
                chunk_count=0,
                filters_applied=filters,
                notes=[
                    "The document corpus is empty. Index filings before searching, or "
                    "answer from the financial-statement and news servers instead."
                ],
            )

        response = self.collection.query(
            query_texts=[query], n_results=limit, where=where or None
        )
        chunks = _to_chunks(response)

        notes: list[str] = []
        if not chunks:
            described = ", ".join(f"{k}={v}" for k, v in filters.items()) or "no filters"
            notes.append(
                f"No indexed passages matched '{query}' ({described}). This means the "
                "corpus contains no such document, not that the company made no such "
                "statement."
            )
        return SearchResult(
            query=query,
            chunks=chunks,
            chunk_count=len(chunks),
            filters_applied=filters,
            notes=notes,
        )

    def get_document(self, document_id: str, *, max_chars: int = MAX_DOCUMENT_CHARS) -> Document:
        """Reassemble a whole document from its chunks, in order."""
        response = self.collection.get(
            where={"document_id": document_id}, include=["documents", "metadatas"]
        )
        if not response["ids"]:
            raise LookupError(
                f"No document with id '{document_id}' is indexed. Use search_by_company "
                "to list what is available."
            )

        pairs: list[tuple[str, Mapping[str, Any]]] = list(
            zip(response["documents"] or [], response["metadatas"] or [], strict=True)
        )
        ordered = sorted(pairs, key=lambda pair: int(pair[1].get("chunk_index", 0) or 0))
        # Overlapping chunks would duplicate text if naively concatenated, so the overlap
        # is trimmed back off on the way out.
        parts = [ordered[0][0]] + [text[CHUNK_OVERLAP:] for text, _ in ordered[1:]]
        text = "".join(parts)
        truncated = len(text) > max_chars

        return Document(
            metadata=_from_flat(ordered[0][1]),
            text=text[:max_chars] + ("…" if truncated else ""),
            chunk_count=len(ordered),
            truncated=truncated,
        )

    def list_documents(
        self,
        *,
        ticker: str | None = None,
        year: int | None = None,
        document_type: str | None = None,
    ) -> DocumentListing:
        """List indexed documents matching the given metadata filters."""
        filters = build_filters(ticker=ticker, document_type=document_type, year=year)
        where = _to_chroma_where(filters)

        response = self.collection.get(where=where or None, include=["metadatas"])
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for meta in response["metadatas"] or []:
            grouped.setdefault(str(meta["document_id"]), []).append(meta)

        documents = [
            DocumentSummary(metadata=_from_flat(metas[0]), chunk_count=len(metas))
            for metas in grouped.values()
        ]
        documents.sort(key=lambda d: (d.metadata.year, d.metadata.document_id), reverse=True)

        notes: list[str] = []
        if not documents:
            described = ", ".join(f"{k}={v}" for k, v in filters.items()) or "no filters"
            notes.append(f"No indexed documents match {described}.")

        return DocumentListing(
            documents=documents,
            document_count=len(documents),
            filters_applied=filters,
            notes=notes,
        )


# ── helpers ───────────────────────────────────────────────────────────────────


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks, preferring to break at a paragraph or sentence.

    The overlap exists so a statement that straddles a boundary is still retrievable in
    full from at least one chunk.
    """
    cleaned = re.sub(r"\n{3,}", "\n\n", text.strip())
    if len(cleaned) <= size:
        return [cleaned]

    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + size, len(cleaned))
        if end < len(cleaned):
            window = cleaned[start:end]
            for separator in ("\n\n", ". ", "\n", " "):
                cut = window.rfind(separator)
                if cut > size // 2:
                    end = start + cut + len(separator)
                    break
        chunks.append(cleaned[start:end].strip())
        if end >= len(cleaned):
            break
        start = max(end - overlap, start + 1)

    return [c for c in chunks if c]


def validate_document_type(document_type: str) -> str:
    """Reject unsupported document types with a message listing what is supported."""
    if document_type not in SUPPORTED_DOCUMENT_TYPES:
        raise ValueError(
            f"Unsupported document type '{document_type}'. Supported types: "
            f"{', '.join(SUPPORTED_DOCUMENT_TYPES)}."
        )
    return document_type


def validate_year(year: int) -> int:
    if not isinstance(year, int) or isinstance(year, bool):
        raise ValueError(f"Year must be an integer, got {type(year).__name__}.")
    if not 1900 <= year <= 2200:
        raise ValueError(f"Year {year} is out of range. Use a four-digit fiscal year.")
    return year


def build_filters(
    *,
    ticker: str | None = None,
    document_type: str | None = None,
    year: int | None = None,
    quarter: int | None = None,
) -> dict[str, str | int]:
    """Validate and normalise metadata filters before they reach the vector store."""
    filters: dict[str, str | int] = {}
    if ticker:
        filters["ticker"] = normalize_ticker(ticker)
    if document_type:
        filters["document_type"] = validate_document_type(document_type)
    if year is not None:
        filters["year"] = validate_year(year)
    if quarter is not None:
        if quarter not in (1, 2, 3, 4):
            raise ValueError(f"Quarter must be 1, 2, 3 or 4, got {quarter}.")
        filters["quarter"] = quarter
    return filters


def _to_chroma_where(filters: dict[str, str | int]) -> dict[str, Any]:
    """Chroma needs an explicit $and when more than one condition is present."""
    if not filters:
        return {}
    if len(filters) == 1:
        return dict(filters)
    return {"$and": [{key: value} for key, value in filters.items()]}


def _flatten_metadata(metadata: DocumentMetadata) -> dict[str, Any]:
    """Chroma metadata values must be scalars, so None is dropped rather than stored."""
    flat = metadata.model_dump()
    return {k: v for k, v in flat.items() if v is not None}


def _from_flat(meta: Mapping[str, Any]) -> DocumentMetadata:
    """Rebuild domain metadata from the flat scalars Chroma stores."""
    return DocumentMetadata(
        document_id=meta["document_id"],
        company=meta["company"],
        ticker=meta["ticker"],
        document_type=meta["document_type"],
        year=int(meta["year"]),
        quarter=int(meta["quarter"]) if meta.get("quarter") is not None else None,
        title=meta["title"],
        source=meta["source"],
    )


def _to_chunks(response: Any) -> list[DocumentChunk]:
    """Convert a Chroma query response into domain chunks, cosine distance -> 0-1 score."""
    ids = (response.get("ids") or [[]])[0]
    documents = (response.get("documents") or [[]])[0]
    metadatas = (response.get("metadatas") or [[]])[0]
    distances = (response.get("distances") or [[]])[0]

    chunks: list[DocumentChunk] = []
    for chunk_id, text, meta, distance in zip(
        ids, documents, metadatas, distances, strict=False
    ):
        chunks.append(
            DocumentChunk(
                chunk_id=chunk_id,
                text=text,
                chunk_index=int(meta.get("chunk_index", 0)),
                metadata=_from_flat(meta),
                relevance_score=round(max(0.0, min(1.0, 1.0 - float(distance) / 2.0)), 4),
            )
        )
    return chunks


def make_document_id(ticker: str, document_type: str, year: int, quarter: int | None = None) -> str:
    """A stable id, so re-indexing the same document replaces it rather than duplicating."""
    parts = f"{ticker.upper()}|{document_type}|{year}|{quarter or ''}"
    return hashlib.sha1(parts.encode()).hexdigest()[:16]
