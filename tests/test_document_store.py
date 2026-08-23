from unittest.mock import MagicMock, patch

import pytest

from src.document_chunker import DocumentChunk
from src.document_store import (
    COLLECTION_METADATA,
    EMBEDDING_MODEL,
    GLOBAL_ACCOUNT_ID,
    account_agreement_filter,
    chunk_metadata,
    generate_embeddings,
    global_documents_filter,
    index_chunks,
    search_chunks,
)


# =====================================================================
# HELPER: CREATE MOCK DOCUMENT CHUNK
# =====================================================================

def create_mock_chunk(
    chunk_id="chunk-1",
    document_id="doc-1",
    account_id=None,
):
    """
    Create a mock DocumentChunk for testing.
    """

    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        source_file="test_document.pdf",
        document_type="customer_agreement",
        status="current",
        applies_to="customer",
        account_id=account_id,
        page_number=1,
        page_range="1",
        chunk_index=0,
        chunk_text="This is a test document chunk.",
    )


# =====================================================================
# METADATA TESTS
# =====================================================================

def test_chunk_metadata_for_global_document():

    chunk = create_mock_chunk(
        account_id=None
    )

    metadata = chunk_metadata(chunk)

    assert metadata["chunk_id"] == "chunk-1"
    assert metadata["document_id"] == "doc-1"
    assert metadata["source_file"] == "test_document.pdf"

    assert metadata["account_id"] == GLOBAL_ACCOUNT_ID

    assert metadata["is_account_specific"] is False

    assert metadata["page_number"] == 1
    assert metadata["page_range"] == "1"
    assert metadata["chunk_index"] == 0


def test_chunk_metadata_for_account_specific_document():

    chunk = create_mock_chunk(
        account_id="ACCT-001"
    )

    metadata = chunk_metadata(chunk)

    assert metadata["account_id"] == "ACCT-001"

    assert metadata["is_account_specific"] is True


# =====================================================================
# EMBEDDING GENERATION TESTS
# =====================================================================

def test_generate_embeddings_returns_empty_list_for_empty_input():

    result = generate_embeddings(
        [],
        client=MagicMock(),
    )

    assert result == []


def test_generate_embeddings_rejects_blank_text():

    with pytest.raises(
        ValueError,
        match="Cannot create embeddings for blank chunk text",
    ):
        generate_embeddings(
            ["Valid text", "   "],
            client=MagicMock(),
        )


def test_generate_embeddings_uses_openai_client():

    mock_client = MagicMock()

    mock_embedding_1 = MagicMock()
    mock_embedding_1.embedding = [0.1, 0.2, 0.3]

    mock_embedding_2 = MagicMock()
    mock_embedding_2.embedding = [0.4, 0.5, 0.6]

    mock_response = MagicMock()
    mock_response.data = [
        mock_embedding_1,
        mock_embedding_2,
    ]

    mock_client.embeddings.create.return_value = (
        mock_response
    )

    texts = [
        "First document",
        "Second document",
    ]

    result = generate_embeddings(
        texts,
        client=mock_client,
    )

    assert result == [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]

    mock_client.embeddings.create.assert_called_once_with(
        model=EMBEDDING_MODEL,
        input=texts,
    )


def test_generate_embeddings_preserves_text_order():

    mock_client = MagicMock()

    mock_embedding_1 = MagicMock()
    mock_embedding_1.embedding = [1.0]

    mock_embedding_2 = MagicMock()
    mock_embedding_2.embedding = [2.0]

    mock_embedding_3 = MagicMock()
    mock_embedding_3.embedding = [3.0]

    mock_response = MagicMock()
    mock_response.data = [
        mock_embedding_1,
        mock_embedding_2,
        mock_embedding_3,
    ]

    mock_client.embeddings.create.return_value = (
        mock_response
    )

    result = generate_embeddings(
        [
            "first",
            "second",
            "third",
        ],
        client=mock_client,
    )

    assert result == [
        [1.0],
        [2.0],
        [3.0],
    ]


# =====================================================================
# GLOBAL DOCUMENT FILTER TESTS
# =====================================================================

def test_global_documents_filter_without_optional_filters():

    result = global_documents_filter()

    assert result == {
        "account_id": GLOBAL_ACCOUNT_ID
    }


def test_global_documents_filter_with_document_type_only():

    result = global_documents_filter(
        document_type="support_policy"
    )

    assert result == {
        "$and": [
            {
                "account_id": GLOBAL_ACCOUNT_ID
            },
            {
                "document_type": "support_policy"
            },
        ]
    }


def test_global_documents_filter_with_status_only():

    result = global_documents_filter(
        status="current"
    )

    assert result == {
        "$and": [
            {
                "account_id": GLOBAL_ACCOUNT_ID
            },
            {
                "status": "current"
            },
        ]
    }


def test_global_documents_filter_with_document_type_and_status():

    result = global_documents_filter(
        document_type="support_policy",
        status="current",
    )

    assert result == {
        "$and": [
            {
                "account_id": GLOBAL_ACCOUNT_ID
            },
            {
                "document_type": "support_policy"
            },
            {
                "status": "current"
            },
        ]
    }


# =====================================================================
# ACCOUNT AGREEMENT FILTER TESTS
# =====================================================================

def test_account_agreement_filter_converts_account_id_to_uppercase():

    result = account_agreement_filter(
        "acct-001"
    )

    assert result == {
        "$and": [
            {
                "account_id": "ACCT-001"
            },
            {
                "is_account_specific": True
            },
            {
                "document_type": "customer_agreement"
            },
        ]
    }


def test_account_agreement_filter_rejects_blank_account_id():

    with pytest.raises(
        ValueError,
        match="account_id must not be blank",
    ):
        account_agreement_filter("   ")


# =====================================================================
# INDEX CHUNKS TESTS
# =====================================================================

def test_index_chunks_rejects_empty_chunk_list():

    with pytest.raises(
        ValueError,
        match="Cannot index an empty chunk list",
    ):
        index_chunks([])


def test_index_chunks_rejects_duplicate_chunk_ids():

    chunk_1 = create_mock_chunk(
        chunk_id="duplicate"
    )

    chunk_2 = create_mock_chunk(
        chunk_id="duplicate"
    )

    with pytest.raises(
        ValueError,
        match="Chunk IDs must be unique",
    ):
        index_chunks([
            chunk_1,
            chunk_2,
        ])


@patch("src.document_store.generate_embeddings")
@patch("src.document_store.get_collection")
def test_index_chunks_adds_new_chunks(
    mock_get_collection,
    mock_generate_embeddings,
):

    mock_collection = MagicMock()

    mock_collection.get.return_value = {
        "ids": []
    }

    mock_collection.count.return_value = 2

    mock_get_collection.return_value = (
        mock_collection
    )

    mock_generate_embeddings.return_value = [
        [0.1, 0.2],
        [0.3, 0.4],
    ]

    chunk_1 = create_mock_chunk(
        chunk_id="chunk-1"
    )

    chunk_2 = create_mock_chunk(
        chunk_id="chunk-2"
    )

    result = index_chunks([
        chunk_1,
        chunk_2,
    ])

    assert result["generated_chunks"] == 2
    assert result["upserted_chunks"] == 2
    assert result["deleted_stale_chunks"] == 0
    assert result["collection_count"] == 2

    mock_generate_embeddings.assert_called_once()

    mock_collection.upsert.assert_called_once()

    mock_collection.delete.assert_not_called()


@patch("src.document_store.get_collection")
def test_index_chunks_deletes_stale_chunks(
    mock_get_collection,
):

    mock_collection = MagicMock()

    mock_collection.get.return_value = {
        "ids": [
            "old-chunk",
            "chunk-1",
        ]
    }

    mock_collection.count.return_value = 1

    mock_get_collection.return_value = (
        mock_collection
    )

    chunk = create_mock_chunk(
        chunk_id="chunk-1"
    )

    result = index_chunks([
        chunk
    ])

    assert result["generated_chunks"] == 1
    assert result["upserted_chunks"] == 0
    assert result["deleted_stale_chunks"] == 1
    assert result["collection_count"] == 1

    mock_collection.delete.assert_called_once_with(
        ids=["old-chunk"]
    )


@patch("src.document_store.generate_embeddings")
@patch("src.document_store.get_collection")
def test_index_chunks_preserves_existing_chunks(
    mock_get_collection,
    mock_generate_embeddings,
):

    mock_collection = MagicMock()

    mock_collection.get.return_value = {
        "ids": [
            "chunk-1"
        ]
    }

    mock_collection.count.return_value = 1

    mock_get_collection.return_value = (
        mock_collection
    )

    chunk = create_mock_chunk(
        chunk_id="chunk-1"
    )

    result = index_chunks([
        chunk
    ])

    assert result["generated_chunks"] == 1
    assert result["upserted_chunks"] == 0
    assert result["deleted_stale_chunks"] == 0
    assert result["collection_count"] == 1

    mock_generate_embeddings.assert_not_called()

    mock_collection.upsert.assert_not_called()

    mock_collection.delete.assert_not_called()


# =====================================================================
# SEARCH CHUNKS VALIDATION TESTS
# =====================================================================

def test_search_chunks_rejects_blank_query():

    with pytest.raises(
        ValueError,
        match="Search query must not be blank",
    ):
        search_chunks("   ")


def test_search_chunks_rejects_invalid_top_k():

    with pytest.raises(
        ValueError,
        match="top_k must be at least 1",
    ):
        search_chunks(
            "test query",
            top_k=0,
        )


@patch("src.document_store.get_collection")
def test_search_chunks_rejects_empty_collection(
    mock_get_collection,
):

    mock_collection = MagicMock()

    mock_collection.count.return_value = 0

    mock_get_collection.return_value = (
        mock_collection
    )

    with pytest.raises(
        RuntimeError,
        match="document collection is empty",
    ):
        search_chunks(
            "What is the cancellation policy?"
        )


# =====================================================================
# SEARCH RESULT TEST
# =====================================================================

@patch("src.document_store.generate_embeddings")
@patch("src.document_store.get_collection")
def test_search_chunks_returns_source_traceable_evidence(
    mock_get_collection,
    mock_generate_embeddings,
):

    mock_collection = MagicMock()

    mock_collection.count.return_value = 1

    mock_collection.query.return_value = {
        "ids": [
            ["chunk-1"]
        ],
        "documents": [
            ["Cancellation fee information"]
        ],
        "metadatas": [
            [
                {
                    "document_id": "doc-1",
                    "source_file": "policy.pdf",
                    "document_type": (
                        "cancellation_service_credit_sop"
                    ),
                    "status": "current",
                    "applies_to": "all_customers",
                    "account_id": GLOBAL_ACCOUNT_ID,
                    "page_number": 2,
                    "page_range": "2",
                    "chunk_index": 0,
                }
            ]
        ],
        "distances": [
            [0.123]
        ],
    }

    mock_get_collection.return_value = (
        mock_collection
    )

    mock_generate_embeddings.return_value = [
        [0.1, 0.2, 0.3]
    ]

    result = search_chunks(
        "What is the cancellation fee?",
        top_k=5,
    )

    assert len(result) == 1

    evidence = result[0]

    assert evidence["chunk_id"] == "chunk-1"

    assert evidence["chunk_text"] == (
        "Cancellation fee information"
    )

    assert evidence["document_id"] == "doc-1"

    assert evidence["source_file"] == "policy.pdf"

    assert evidence["document_type"] == (
        "cancellation_service_credit_sop"
    )

    assert evidence["status"] == "current"

    assert evidence["applies_to"] == "all_customers"

    # Global documents should be converted back to None.
    assert evidence["account_id"] is None

    assert evidence["page_number"] == 2
    assert evidence["page_range"] == "2"
    assert evidence["chunk_index"] == 0
    assert evidence["distance"] == 0.123

    mock_collection.query.assert_called_once()


@patch("src.document_store.generate_embeddings")
@patch("src.document_store.get_collection")
def test_search_chunks_respects_where_filter(
    mock_get_collection,
    mock_generate_embeddings,
):

    mock_collection = MagicMock()

    mock_collection.count.return_value = 3

    mock_collection.query.return_value = {
        "ids": [[]],
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }

    mock_get_collection.return_value = (
        mock_collection
    )

    mock_generate_embeddings.return_value = [
        [0.1, 0.2]
    ]

    where_filter = {
        "status": "current"
    }

    result = search_chunks(
        "test query",
        top_k=2,
        where=where_filter,
    )

    assert result == []

    call_kwargs = mock_collection.query.call_args.kwargs

    assert call_kwargs["where"] == where_filter

    assert call_kwargs["n_results"] == 2


# =====================================================================
# CONSTANT TEST
# =====================================================================

def test_collection_metadata_matches_embedding_configuration():

    assert COLLECTION_METADATA == {
        "embedding_provider": "openai",
        "embedding_model": EMBEDDING_MODEL,
    }