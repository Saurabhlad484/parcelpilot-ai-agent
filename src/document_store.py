"""Persistent ChromaDB storage for ParcelPilot document chunks.

This module handles:
1. Creating embeddings using OpenAI
2. Storing embeddings in ChromaDB
3. Searching document chunks using vector similarity

Authority decisions and answer generation belong to later layers.
"""

from __future__ import annotations

import os
from typing import Any

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

from src.data_loader import PROJECT_ROOT
from src.document_chunker import DocumentChunk, chunk_all_documents
from src.document_processor import process_all_documents


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

CHROMA_DIRECTORY = PROJECT_ROOT / "data" / "chroma"

COLLECTION_NAME = "parcelpilot_documents"

EMBEDDING_PROVIDER = "openai"

# Cheap and suitable for this project
EMBEDDING_MODEL = "text-embedding-3-small"

EMBEDDING_BATCH_SIZE = 100

GLOBAL_ACCOUNT_ID = "__global__"

COLLECTION_METADATA = {
    "embedding_provider": EMBEDDING_PROVIDER,
    "embedding_model": EMBEDDING_MODEL,
}


# ---------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------

def get_openai_client() -> OpenAI:
    """
    Create and return an OpenAI client.

    The API key is loaded from the project's .env file.
    """

    load_dotenv(PROJECT_ROOT / ".env", override=True)

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required to create or query document "
            "embeddings. Add it to your local .env file."
        )

    return OpenAI(api_key=api_key)


# ---------------------------------------------------------
# Chroma collection
# ---------------------------------------------------------

def get_collection() -> Any:
    """
    Open the ChromaDB collection.

    If an existing collection was created using a different embedding
    provider or embedding model, delete it because embeddings from
    different models cannot be compared safely.
    """

    CHROMA_DIRECTORY.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIRECTORY)
    )

    collection_names = {
        collection.name
        for collection in client.list_collections()
    }

    if COLLECTION_NAME in collection_names:

        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=None,
        )

        existing_metadata = collection.metadata or {}

        provider_matches = (
            existing_metadata.get("embedding_provider")
            == EMBEDDING_PROVIDER
        )

        model_matches = (
            existing_metadata.get("embedding_model")
            == EMBEDDING_MODEL
        )

        if not provider_matches or not model_matches:

            print(
                "Existing embeddings use a different provider or model. "
                "Resetting the ChromaDB collection..."
            )

            client.delete_collection(COLLECTION_NAME)

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata=COLLECTION_METADATA,
        embedding_function=None,
    )


# ---------------------------------------------------------
# Metadata conversion
# ---------------------------------------------------------

def chunk_metadata(
    chunk: DocumentChunk,
) -> dict[str, str | int | bool]:
    """
    Convert DocumentChunk metadata into Chroma-compatible values.
    """

    is_account_specific = chunk.account_id is not None

    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "source_file": chunk.source_file,
        "document_type": chunk.document_type,
        "status": chunk.status,
        "applies_to": chunk.applies_to,

        # Chroma metadata should not use None.
        "account_id": (
            chunk.account_id
            if chunk.account_id is not None
            else GLOBAL_ACCOUNT_ID
        ),

        "is_account_specific": is_account_specific,
        "page_number": chunk.page_number,
        "page_range": chunk.page_range,
        "chunk_index": chunk.chunk_index,
    }


# ---------------------------------------------------------
# Generate OpenAI embeddings
# ---------------------------------------------------------

def generate_embeddings(
    texts: list[str],
    client: OpenAI | None = None,
) -> list[list[float]]:
    """
    Generate embeddings using OpenAI.

    Texts are processed in batches to avoid unnecessarily large
    API requests while preserving the original text order.
    """

    if not texts:
        return []

    if any(not text.strip() for text in texts):
        raise ValueError(
            "Cannot create embeddings for blank chunk text."
        )

    embedding_client = client or get_openai_client()

    embeddings: list[list[float]] = []

    for start in range(
        0,
        len(texts),
        EMBEDDING_BATCH_SIZE,
    ):

        batch = texts[
            start:start + EMBEDDING_BATCH_SIZE
        ]

        response = embedding_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )

        embeddings.extend(
            item.embedding
            for item in response.data
        )

    return embeddings


# ---------------------------------------------------------
# Index document chunks
# ---------------------------------------------------------

def index_chunks(
    chunks: list[DocumentChunk],
) -> dict[str, int]:
    """
    Synchronise current document chunks with ChromaDB.

    New chunks are embedded and stored.
    Stale chunks are removed.
    Existing valid chunks are preserved.
    """

    if not chunks:
        raise ValueError(
            "Cannot index an empty chunk list."
        )

    chunk_ids = {
        chunk.chunk_id
        for chunk in chunks
    }

    if len(chunk_ids) != len(chunks):
        raise ValueError(
            "Chunk IDs must be unique before indexing."
        )

    collection = get_collection()

    existing_ids = set(
        collection.get(include=[])["ids"]
    )

    current_ids = {
        chunk.chunk_id
        for chunk in chunks
    }

    # ---------------------------------------------------------
    # Remove stale chunks
    # ---------------------------------------------------------

    stale_ids = sorted(
        existing_ids - current_ids
    )

    if stale_ids:
        collection.delete(ids=stale_ids)

    # ---------------------------------------------------------
    # Add only new chunks
    # ---------------------------------------------------------

    chunks_to_upsert = [
        chunk
        for chunk in chunks
        if chunk.chunk_id not in existing_ids
    ]

    if chunks_to_upsert:

        print(
            f"Creating embeddings for "
            f"{len(chunks_to_upsert)} chunk(s)..."
        )

        embeddings = generate_embeddings(
            [
                chunk.chunk_text
                for chunk in chunks_to_upsert
            ]
        )

        collection.upsert(
            ids=[
                chunk.chunk_id
                for chunk in chunks_to_upsert
            ],
            documents=[
                chunk.chunk_text
                for chunk in chunks_to_upsert
            ],
            metadatas=[
                chunk_metadata(chunk)
                for chunk in chunks_to_upsert
            ],
            embeddings=embeddings,
        )

    return {
        "generated_chunks": len(chunks),
        "upserted_chunks": len(chunks_to_upsert),
        "deleted_stale_chunks": len(stale_ids),
        "collection_count": collection.count(),
    }


# ---------------------------------------------------------
# Search chunks
# ---------------------------------------------------------

def search_chunks(
    query: str,
    top_k: int = 5,
    where: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Search the ChromaDB collection using an OpenAI query embedding.

    Returns source-traceable evidence.
    """

    if not query.strip():
        raise ValueError(
            "Search query must not be blank."
        )

    if top_k < 1:
        raise ValueError(
            "top_k must be at least 1."
        )

    collection = get_collection()

    if collection.count() == 0:
        raise RuntimeError(
            "The document collection is empty. "
            "Run indexing before searching."
        )

    # Create embedding for user's query
    query_embedding = generate_embeddings(
        [query]
    )[0]

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(
            top_k,
            collection.count(),
        ),
        where=where,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    evidence: list[dict[str, Any]] = []

    ids = result.get("ids", [[]])[0]

    documents = result.get(
        "documents",
        [[]],
    )[0]

    metadatas = result.get(
        "metadatas",
        [[]],
    )[0]

    distances = result.get(
        "distances",
        [[]],
    )[0]

    for (
        chunk_id,
        chunk_text,
        metadata,
        distance,
    ) in zip(
        ids,
        documents,
        metadatas,
        distances,
    ):

        evidence.append(
            {
                "chunk_id": chunk_id,
                "chunk_text": chunk_text,

                "document_id": (
                    metadata["document_id"]
                ),

                "source_file": (
                    metadata["source_file"]
                ),

                "document_type": (
                    metadata["document_type"]
                ),

                "status": metadata["status"],

                "applies_to": (
                    metadata["applies_to"]
                ),

                "account_id": (
                    None
                    if metadata["account_id"]
                    == GLOBAL_ACCOUNT_ID
                    else metadata["account_id"]
                ),

                "page_number": (
                    metadata["page_number"]
                ),

                "page_range": (
                    metadata["page_range"]
                ),

                "chunk_index": (
                    metadata["chunk_index"]
                ),

                "distance": distance,
            }
        )

    return evidence


# ---------------------------------------------------------
# Metadata filters
# ---------------------------------------------------------

def global_documents_filter(
    document_type: str | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    """
    Return a Chroma filter for globally applicable documents.
    """

    clauses: list[dict[str, Any]] = [
        {
            "account_id": GLOBAL_ACCOUNT_ID
        }
    ]

    if document_type:
        clauses.append(
            {
                "document_type": document_type
            }
        )

    if status:
        clauses.append(
            {
                "status": status
            }
        )

    if len(clauses) == 1:
        return clauses[0]

    return {
        "$and": clauses
    }


def account_agreement_filter(
    account_id: str,
) -> dict[str, Any]:
    """
    Return a filter for exactly one customer's agreement.

    This prevents retrieving another customer's agreement.
    """

    if not account_id.strip():
        raise ValueError(
            "account_id must not be blank."
        )

    return {
        "$and": [
            {
                "account_id": account_id.upper()
            },
            {
                "is_account_specific": True
            },
            {
                "document_type": "customer_agreement"
            },
        ]
    }


# ---------------------------------------------------------
# Display helper
# ---------------------------------------------------------

def _print_search_result(
    label: str,
    evidence: list[dict[str, Any]],
) -> None:
    """
    Print a compact validation summary.
    """

    print(f"{label}: {len(evidence)} result(s)")

    for item in evidence:

        print(
            f"  {item['source_file']} | "
            f"page={item['page_range']} | "
            f"status={item['status']} | "
            f"distance={item['distance']:.4f}"
        )


# ---------------------------------------------------------
# Test / indexing entry point
# ---------------------------------------------------------

def main() -> None:
    """
    Process documents, create chunks, index them,
    and run sample searches.
    """

    print("\nProcessing documents...")

    documents = process_all_documents()

    chunks = chunk_all_documents(documents)

    print(f"Documents processed: {len(documents)}")
    print(f"Chunks generated: {len(chunks)}")

    try:

        report = index_chunks(chunks)

    except RuntimeError as error:

        raise SystemExit(
            f"Configuration error: {error}"
        ) from error

    print(
        f"Chunks upserted: "
        f"{report['upserted_chunks']}"
    )

    print(
        f"Chunks deleted: "
        f"{report['deleted_stale_chunks']}"
    )

    print(
        f"Collection count: "
        f"{report['collection_count']}"
    )

    print("\nRunning validation searches...\n")

    # A. Cancellation SOP
    _print_search_result(
        "A. Cancellation SOP",
        search_chunks(
            "What is the cancellation fee "
            "for a booked shipment?",
            where={
                "$and": [
                    {
                        "status": {
                            "$ne": "deprecated"
                        }
                    },
                    {
                        "document_type": (
                            "cancellation_service_credit_sop"
                        )
                    },
                ]
            },
        ),
    )

    # B. Product guide
    _print_search_result(
        "B. Product guide",
        search_chunks(
            "What is the bulk upload limit?",
            where=global_documents_filter(
                "product_operations_guide",
                "current",
            ),
        ),
    )

    # C. Northstar agreement
    _print_search_result(
        "C. Northstar agreement",
        search_chunks(
            "What are Northstar's cancellation terms?",
            where=account_agreement_filter(
                "ACCT-001"
            ),
        ),
    )

    # D. LumenWorks agreement
    _print_search_result(
        "D. LumenWorks agreement",
        search_chunks(
            "What are LumenWorks' service credit terms?",
            where=account_agreement_filter(
                "ACCT-002"
            ),
        ),
    )

    # E. Deprecated policy
    _print_search_result(
        "E. Deprecated policy",
        search_chunks(
            "What did the old support policy say?",
            where={
                "status": "deprecated"
            },
        ),
    )


if __name__ == "__main__":
    main()