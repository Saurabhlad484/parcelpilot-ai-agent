"""Create page-traceable, semantic chunks from processed ParcelPilot PDFs.

This module is intentionally storage-agnostic. It converts the page-level
objects returned by ``document_processor`` into deterministic chunk records for
a later vector-store layer; it does not create embeddings or persist anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import re

from src.document_processor import ProcessedDocument, process_all_documents


# The supplied policy PDFs are short. Keeping an entire semantic section within
# 1,000 characters retains related rules while remaining a useful retrieval unit.
MAX_CHUNK_CHARS = 1_000
MIN_CHUNK_CHARS = 180
_SECTION_START = re.compile(r"(?=(?:\d+\.\s+[A-Z]|KI-\d+\s+-))")


@dataclass(frozen=True)
class DocumentChunk:
    """One deterministic, page-traceable section of a processed document."""

    chunk_id: str
    document_id: str
    source_file: str
    document_type: str
    status: str
    applies_to: str
    account_id: str | None
    page_number: int
    page_range: str
    chunk_index: int
    chunk_text: str


def normalize_text(text: str) -> str:
    """Normalise PDF extraction whitespace without changing the source meaning."""
    return " ".join(text.split())


def split_page_into_sections(text: str) -> list[str]:
    """Split a page at numbered and known-issue headings when they are present."""
    normalised = normalize_text(text)
    if not normalised:
        return []

    starts = [match.start() for match in _SECTION_START.finditer(normalised)]
    if not starts:
        return [normalised]

    sections: list[str] = []
    if starts[0] > 0:
        sections.append(normalised[: starts[0]].strip())
    sections.extend(
        normalised[start:end].strip()
        for start, end in zip(starts, starts[1:] + [len(normalised)])
    )
    return [section for section in sections if section]


def _split_oversized_section(section: str) -> list[str]:
    """Split only an oversized semantic section, preferring sentence boundaries."""
    if len(section) <= MAX_CHUNK_CHARS:
        return [section]

    sentences = re.split(r"(?<=[.!?])\s+", section)
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > MAX_CHUNK_CHARS:
            pieces.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def _pack_sections(sections: list[str]) -> list[str]:
    """Keep whole sections together and merge only small adjacent context blocks."""
    chunks: list[str] = []
    current = ""

    for section in sections:
        for piece in _split_oversized_section(section):
            candidate = f"{current} {piece}".strip()
            if current and len(candidate) > MAX_CHUNK_CHARS:
                chunks.append(current)
                current = piece
            else:
                current = candidate

            # A short preamble is combined with the immediately following
            # section, rather than becoming a standalone, low-value chunk.
            if len(current) >= MIN_CHUNK_CHARS:
                chunks.append(current)
                current = ""

    if current:
        if chunks and len(current) < MIN_CHUNK_CHARS and len(f"{chunks[-1]} {current}") <= MAX_CHUNK_CHARS:
            chunks[-1] = f"{chunks[-1]} {current}"
        else:
            chunks.append(current)
    return chunks


def chunk_document(document: ProcessedDocument) -> list[DocumentChunk]:
    """Create page-preserving, deterministic chunks for one processed PDF."""
    chunks: list[DocumentChunk] = []
    chunk_index = 0

    for page in document.pages:
        sections = split_page_into_sections(page.text)
        for text in _pack_sections(sections):
            # The title gives each extracted section enough document context for
            # a later retrieval consumer without mixing in another document.
            chunk_text = f"{document.metadata.document_title}\n{text}"
            content_hash = sha256(chunk_text.encode("utf-8")).hexdigest()[:12]
            chunk_id = (
                f"{document.document_id}-p{page.page_number}-"
                f"c{chunk_index}-{content_hash}"
            )
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=document.document_id,
                    source_file=document.metadata.source_file,
                    document_type=document.metadata.document_type,
                    status=document.metadata.status,
                    applies_to=document.metadata.applicability,
                    account_id=document.metadata.account_id,
                    page_number=page.page_number,
                    page_range=str(page.page_number),
                    chunk_index=chunk_index,
                    chunk_text=chunk_text,
                )
            )
            chunk_index += 1
    return chunks


def chunk_all_documents(documents: list[ProcessedDocument]) -> list[DocumentChunk]:
    """Chunk each supplied document independently, preserving its boundary."""
    return [chunk for document in documents for chunk in chunk_document(document)]


def validate_chunks(
    chunks: list[DocumentChunk], documents: list[ProcessedDocument]
) -> None:
    """Validate metadata, page traceability, and required account isolation facts."""
    required_fields = (
        "chunk_id",
        "document_id",
        "source_file",
        "document_type",
        "status",
        "applies_to",
        "page_number",
        "page_range",
        "chunk_index",
        "chunk_text",
    )
    document_by_id = {document.document_id: document for document in documents}

    if not chunks:
        raise ValueError("Chunking produced no chunks.")
    if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
        raise ValueError("Chunk IDs must be unique.")

    for chunk in chunks:
        for field_name in required_fields:
            value = getattr(chunk, field_name)
            if value is None or (isinstance(value, str) and not value.strip()):
                raise ValueError(f"Chunk {chunk.chunk_id} is missing {field_name}.")
        document = document_by_id.get(chunk.document_id)
        if document is None:
            raise ValueError(f"Chunk {chunk.chunk_id} references an unknown document.")
        if chunk.page_number not in {page.page_number for page in document.pages}:
            raise ValueError(f"Chunk {chunk.chunk_id} has invalid page metadata.")
        if chunk.account_id != document.metadata.account_id:
            raise ValueError(f"Chunk {chunk.chunk_id} has incorrect account applicability.")

    status_by_document = {
        chunk.document_id: chunk.status
        for chunk in chunks
    }
    if status_by_document.get("support-policy-v2") != "deprecated":
        raise ValueError("Deprecated policy chunks must retain status='deprecated'.")

    account_ids_by_document = {
        chunk.document_id: chunk.account_id
        for chunk in chunks
    }
    if account_ids_by_document.get("northstar-logistics-enterprise-agreement") != "ACCT-001":
        raise ValueError("Northstar agreement chunks must retain ACCT-001.")
    if account_ids_by_document.get("lumenworks-service-agreement") != "ACCT-002":
        raise ValueError("LumenWorks agreement chunks must retain ACCT-002.")


def main() -> None:
    """Run executable validation for the standalone document chunking layer."""
    documents = process_all_documents()
    chunks = chunk_all_documents(documents)
    validate_chunks(chunks, documents)

    print(f"Documents processed: {len(documents)}")
    print(f"Chunks created: {len(chunks)}")
    for document in documents:
        document_chunks = [chunk for chunk in chunks if chunk.document_id == document.document_id]
        print(f"{document.metadata.source_file}: {len(document_chunks)} chunk(s)")

    print("Representative chunks:")
    for document in documents:
        chunk = next(item for item in chunks if item.document_id == document.document_id)
        preview = chunk.chunk_text.replace("\n", " ")[:220]
        print(
            f"- {chunk.chunk_id} | page={chunk.page_range} | "
            f"status={chunk.status} | account_id={chunk.account_id} | {preview}"
        )
    print("Validation: PASS")


if __name__ == "__main__":
    main()
