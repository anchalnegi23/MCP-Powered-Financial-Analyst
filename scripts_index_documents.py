"""Index the bundled sample filings into the vector store.

Run once before asking document questions:

    uv run python scripts_index_documents.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from models.research import DocumentMetadata
from services.research_service import ResearchStore, make_document_id

SAMPLES = [
    {
        "path": "data/sample_documents/nvda_10k_2024.txt",
        "company": "NVIDIA Corporation",
        "ticker": "NVDA",
        "document_type": "10-K",
        "year": 2024,
        "title": "NVIDIA Corporation Form 10-K, Fiscal Year 2024 (extract)",
        "source": "data/sample_documents/nvda_10k_2024.txt",
    },
    {
        "path": "data/sample_documents/amd_10k_2024.txt",
        "company": "Advanced Micro Devices, Inc.",
        "ticker": "AMD",
        "document_type": "10-K",
        "year": 2024,
        "title": "AMD Form 10-K, Fiscal Year 2024 (extract)",
        "source": "data/sample_documents/amd_10k_2024.txt",
    },
]


def main() -> int:
    store = ResearchStore()
    for sample in SAMPLES:
        path = Path(sample["path"])
        if not path.exists():
            print(f"skipped (missing): {path}")
            continue

        metadata = DocumentMetadata(
            document_id=make_document_id(
                sample["ticker"], sample["document_type"], sample["year"]
            ),
            company=sample["company"],
            ticker=sample["ticker"],
            document_type=sample["document_type"],
            year=sample["year"],
            title=sample["title"],
            source=sample["source"],
        )
        result = store.index_document(path.read_text(), metadata)
        print(f"indexed {metadata.ticker} {metadata.document_type} {metadata.year}: "
              f"{result.chunks_indexed} chunks (id {result.document_id})")

    print(f"corpus now holds {store.collection.count()} chunks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
