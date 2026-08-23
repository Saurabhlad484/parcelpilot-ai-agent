import pytest

from src.document_chunker import (
    MAX_CHUNK_CHARS,
    MIN_CHUNK_CHARS,
    DocumentChunk,
    _pack_sections,
    _split_oversized_section,
    chunk_all_documents,
    chunk_document,
    normalize_text,
    split_page_into_sections,
    validate_chunks,
)
from src.document_processor import (
    DocumentMetadata,
    ExtractedPage,
    ProcessedDocument,
    process_all_documents,
)


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def create_test_document(
    document_id="test-document",
    source_file="test.pdf",
    document_type="test_type",
    status="current",
    account_id=None,
    title="Test Document",
    pages=None,
):
    """
    Create a small ProcessedDocument for unit tests.
    """

    metadata = DocumentMetadata(
        source_file=source_file,
        document_id=document_id,
        document_title=title,
        document_type=document_type,
        status=status,
        authority_level="test_authority",
        effective_or_updated_date="2026-01-01",
        account_id=account_id,
        account_name=None,
        applicability=(
            f"{account_id} only"
            if account_id
            else "all_accounts"
        ),
        allowed_for_current_requests=True,
        notes="Test document.",
    )

    if pages is None:
        pages = [
            ExtractedPage(
                page_number=1,
                text="This is a test document.",
            )
        ]

    return ProcessedDocument(
        document_id=document_id,
        metadata=metadata,
        pages=pages,
    )


# ---------------------------------------------------------------------
# normalize_text tests
# ---------------------------------------------------------------------

def test_normalize_text_removes_extra_whitespace():
    """
    Multiple spaces, tabs, and newlines should become
    single spaces.
    """

    text = "Hello   world\n\nthis\t is   ParcelPilot."

    result = normalize_text(text)

    assert result == "Hello world this is ParcelPilot."


def test_normalize_text_returns_empty_string_for_whitespace():
    """
    Whitespace-only text should become an empty string.
    """

    result = normalize_text("   \n\t  ")

    assert result == ""


# ---------------------------------------------------------------------
# split_page_into_sections tests
# ---------------------------------------------------------------------

def test_split_page_into_sections_returns_empty_for_blank_text():
    """
    Blank page text should produce no sections.
    """

    result = split_page_into_sections("   \n\t ")

    assert result == []


def test_split_page_into_sections_returns_single_section_without_headings():
    """
    Text without recognised headings should remain one section.
    """

    text = (
        "This policy explains the general rules for "
        "ParcelPilot customers."
    )

    result = split_page_into_sections(text)

    assert result == [text]


def test_split_page_into_sections_splits_numbered_headings():
    """
    Numbered headings should create separate sections.
    """

    text = (
        "1. CANCELLATION Customers may cancel an order. "
        "2. FEES A cancellation fee may apply."
    )

    result = split_page_into_sections(text)

    assert len(result) == 2

    assert result[0].startswith("1. CANCELLATION")

    assert result[1].startswith("2. FEES")


def test_split_page_into_sections_splits_known_issue_headings():
    """
    KI headings should create separate sections.
    """

    text = (
        "KI-001 - Upload Delay CSV uploads may be delayed. "
        "KI-002 - Webhook Retry Some webhook events may retry."
    )

    result = split_page_into_sections(text)

    assert len(result) == 2

    assert result[0].startswith("KI-001")

    assert result[1].startswith("KI-002")


def test_split_page_into_sections_keeps_preamble():
    """
    Text before the first heading should be preserved.
    """

    text = (
        "Introduction to the policy. "
        "1. CANCELLATION Customers may cancel orders."
    )

    result = split_page_into_sections(text)

    assert len(result) == 2

    assert result[0] == "Introduction to the policy."

    assert result[1].startswith("1. CANCELLATION")


# ---------------------------------------------------------------------
# Oversized section splitting tests
# ---------------------------------------------------------------------

def test_split_oversized_section_keeps_small_section_unchanged():
    """
    A section within MAX_CHUNK_CHARS should not be split.
    """

    section = "A" * 500

    result = _split_oversized_section(section)

    assert result == [section]


def test_split_oversized_section_splits_long_text():
    """
    An oversized section with sentence boundaries should be split.
    """

    sentence = "This is a complete sentence about ParcelPilot."

    section = " ".join(
        [sentence] * 30
    )

    assert len(section) > MAX_CHUNK_CHARS

    result = _split_oversized_section(section)

    assert len(result) > 1

    assert all(
        len(piece) <= MAX_CHUNK_CHARS
        for piece in result
    )


def test_split_oversized_section_preserves_all_text():
    """
    Splitting should not lose meaningful text.
    """

    sentence = "ParcelPilot provides shipment support."

    section = " ".join(
        [sentence] * 40
    )

    result = _split_oversized_section(section)

    reconstructed = " ".join(result)

    assert normalize_text(reconstructed) == (
        normalize_text(section)
    )


# ---------------------------------------------------------------------
# Section packing tests
# ---------------------------------------------------------------------

def test_pack_sections_returns_empty_for_empty_input():
    """
    No sections should produce no chunks.
    """

    result = _pack_sections([])

    assert result == []


def test_pack_sections_keeps_text_within_maximum_size():
    """
    Packed chunks should not exceed MAX_CHUNK_CHARS when
    sections themselves can be split safely.
    """

    sections = [
        "First section. " * 20,
        "Second section. " * 20,
        "Third section. " * 20,
    ]

    result = _pack_sections(sections)

    assert all(
        len(chunk) <= MAX_CHUNK_CHARS
        for chunk in result
    )


def test_pack_sections_preserves_content():
    """
    Packing sections should preserve all supplied content.
    """

    sections = [
        "First ParcelPilot section.",
        "Second ParcelPilot section.",
        "Third ParcelPilot section.",
    ]

    result = _pack_sections(sections)

    combined = " ".join(result)

    for section in sections:
        assert section in combined


# ---------------------------------------------------------------------
# chunk_document tests
# ---------------------------------------------------------------------

def test_chunk_document_returns_document_chunks():
    """
    chunk_document should return DocumentChunk objects.
    """

    document = create_test_document()

    chunks = chunk_document(document)

    assert len(chunks) >= 1

    assert all(
        isinstance(chunk, DocumentChunk)
        for chunk in chunks
    )


def test_chunk_document_preserves_document_metadata():
    """
    Each chunk should retain its source document metadata.
    """

    document = create_test_document(
        document_id="test-policy",
        source_file="policy.pdf",
        document_type="support_policy",
        status="current",
    )

    chunks = chunk_document(document)

    for chunk in chunks:
        assert chunk.document_id == "test-policy"
        assert chunk.source_file == "policy.pdf"
        assert chunk.document_type == "support_policy"
        assert chunk.status == "current"


def test_chunk_document_preserves_account_id():
    """
    Account-specific documents should retain their account ID.
    """

    document = create_test_document(
        document_id="northstar-agreement",
        account_id="ACCT-001",
    )

    chunks = chunk_document(document)

    assert all(
        chunk.account_id == "ACCT-001"
        for chunk in chunks
    )


def test_chunk_document_preserves_page_traceability():
    """
    Every chunk should reference a real source page.
    """

    document = create_test_document(
        pages=[
            ExtractedPage(
                page_number=1,
                text="First page content.",
            ),
            ExtractedPage(
                page_number=2,
                text="Second page content.",
            ),
        ]
    )

    chunks = chunk_document(document)

    page_numbers = {
        chunk.page_number
        for chunk in chunks
    }

    assert page_numbers == {1, 2}

    for chunk in chunks:
        assert chunk.page_range == str(
            chunk.page_number
        )


def test_chunk_document_includes_document_title():
    """
    Each chunk should include the document title as context.
    """

    document = create_test_document(
        title="ParcelPilot Test Policy",
    )

    chunks = chunk_document(document)

    assert all(
        chunk.chunk_text.startswith(
            "ParcelPilot Test Policy\n"
        )
        for chunk in chunks
    )


def test_chunk_document_chunk_ids_are_unique():
    """
    Chunks from one document should have unique IDs.
    """

    document = create_test_document(
        pages=[
            ExtractedPage(
                page_number=1,
                text=(
                    "1. FIRST SECTION " + "A " * 150 +
                    "2. SECOND SECTION " + "B " * 150
                ),
            )
        ]
    )

    chunks = chunk_document(document)

    chunk_ids = [
        chunk.chunk_id
        for chunk in chunks
    ]

    assert len(chunk_ids) == len(set(chunk_ids))


def test_chunk_document_is_deterministic():
    """
    The same document should always generate identical chunks
    and chunk IDs.
    """

    document = create_test_document(
        pages=[
            ExtractedPage(
                page_number=1,
                text=(
                    "1. RULE ONE First rule. "
                    "2. RULE TWO Second rule."
                ),
            )
        ]
    )

    first_result = chunk_document(document)

    second_result = chunk_document(document)

    assert first_result == second_result


def test_chunk_document_chunk_indexes_are_sequential():
    """
    Chunk indexes should start from zero and remain sequential.
    """

    document = create_test_document(
        pages=[
            ExtractedPage(
                page_number=1,
                text=(
                    "1. FIRST SECTION " + "A " * 150 +
                    "2. SECOND SECTION " + "B " * 150
                ),
            )
        ]
    )

    chunks = chunk_document(document)

    indexes = [
        chunk.chunk_index
        for chunk in chunks
    ]

    assert indexes == list(
        range(len(chunks))
    )


def test_chunk_document_blank_page_creates_no_chunks():
    """
    A blank page should not produce an empty chunk.
    """

    document = create_test_document(
        pages=[
            ExtractedPage(
                page_number=1,
                text="   \n\t ",
            )
        ]
    )

    chunks = chunk_document(document)

    assert chunks == []


# ---------------------------------------------------------------------
# chunk_all_documents tests
# ---------------------------------------------------------------------

def test_chunk_all_documents_combines_document_chunks():
    """
    chunk_all_documents should combine chunks from every
    supplied document.
    """

    first_document = create_test_document(
        document_id="document-one",
        source_file="one.pdf",
    )

    second_document = create_test_document(
        document_id="document-two",
        source_file="two.pdf",
    )

    chunks = chunk_all_documents(
        [
            first_document,
            second_document,
        ]
    )

    document_ids = {
        chunk.document_id
        for chunk in chunks
    }

    assert document_ids == {
        "document-one",
        "document-two",
    }


def test_chunk_all_documents_matches_individual_chunk_counts():
    """
    Combined chunk count should equal the sum of chunks
    generated for each document individually.
    """

    documents = [
        create_test_document(
            document_id="document-one",
            source_file="one.pdf",
        ),
        create_test_document(
            document_id="document-two",
            source_file="two.pdf",
        ),
    ]

    expected_count = sum(
        len(chunk_document(document))
        for document in documents
    )

    chunks = chunk_all_documents(documents)

    assert len(chunks) == expected_count


# ---------------------------------------------------------------------
# Real ParcelPilot document integration tests
# ---------------------------------------------------------------------

def test_real_documents_can_be_chunked():
    """
    All real ParcelPilot documents should produce chunks.
    """

    documents = process_all_documents()

    chunks = chunk_all_documents(documents)

    assert len(chunks) > 0


def test_real_chunks_validate_successfully():
    """
    Real document chunks should pass all chunk validation rules.
    """

    documents = process_all_documents()

    chunks = chunk_all_documents(documents)

    validate_chunks(chunks, documents)


def test_deprecated_policy_chunks_remain_deprecated():
    """
    Chunks from Support Policy v2 must retain deprecated status.
    """

    documents = process_all_documents()

    chunks = chunk_all_documents(documents)

    deprecated_chunks = [
        chunk
        for chunk in chunks
        if chunk.document_id == "support-policy-v2"
    ]

    assert len(deprecated_chunks) > 0

    assert all(
        chunk.status == "deprecated"
        for chunk in deprecated_chunks
    )


def test_northstar_chunks_retain_correct_account():
    """
    Northstar agreement chunks must remain isolated to ACCT-001.
    """

    documents = process_all_documents()

    chunks = chunk_all_documents(documents)

    northstar_chunks = [
        chunk
        for chunk in chunks
        if (
            chunk.document_id
            == "northstar-logistics-enterprise-agreement"
        )
    ]

    assert len(northstar_chunks) > 0

    assert all(
        chunk.account_id == "ACCT-001"
        for chunk in northstar_chunks
    )


def test_lumenworks_chunks_retain_correct_account():
    """
    LumenWorks agreement chunks must remain isolated to ACCT-002.
    """

    documents = process_all_documents()

    chunks = chunk_all_documents(documents)

    lumenworks_chunks = [
        chunk
        for chunk in chunks
        if (
            chunk.document_id
            == "lumenworks-service-agreement"
        )
    ]

    assert len(lumenworks_chunks) > 0

    assert all(
        chunk.account_id == "ACCT-002"
        for chunk in lumenworks_chunks
    )


def test_global_documents_have_no_account_id():
    """
    Global policy documents should not contain a customer account ID.
    """

    documents = process_all_documents()

    chunks = chunk_all_documents(documents)

    global_chunks = [
        chunk
        for chunk in chunks
        if chunk.document_id in {
            "support-policy-v3",
            "support-policy-v2",
            "cancellation-service-credit-sop-v4",
            "product-operations-guide-known-issues",
        }
    ]

    assert len(global_chunks) > 0

    assert all(
        chunk.account_id is None
        for chunk in global_chunks
    )


# ---------------------------------------------------------------------
# validate_chunks failure tests
# ---------------------------------------------------------------------

def test_validate_chunks_rejects_empty_chunk_list():
    """
    Validation should reject an empty chunk list.
    """

    documents = [
        create_test_document()
    ]

    with pytest.raises(
        ValueError,
        match="Chunking produced no chunks",
    ):
        validate_chunks([], documents)


def test_validate_chunks_rejects_duplicate_chunk_ids():
    """
    Validation should reject duplicate chunk IDs.
    """

    documents = [
        create_test_document()
    ]

    chunks = chunk_document(documents[0])

    duplicate_chunks = [
        chunks[0],
        chunks[0],
    ]

    with pytest.raises(
        ValueError,
        match="Chunk IDs must be unique",
    ):
        validate_chunks(
            duplicate_chunks,
            documents,
        )


def test_validate_chunks_rejects_unknown_document():
    """
    A chunk referencing a document that was not supplied
    should be rejected.
    """

    document = create_test_document()

    chunks = chunk_document(document)

    invalid_chunk = DocumentChunk(
        chunk_id=chunks[0].chunk_id,
        document_id="unknown-document",
        source_file=chunks[0].source_file,
        document_type=chunks[0].document_type,
        status=chunks[0].status,
        applies_to=chunks[0].applies_to,
        account_id=chunks[0].account_id,
        page_number=chunks[0].page_number,
        page_range=chunks[0].page_range,
        chunk_index=chunks[0].chunk_index,
        chunk_text=chunks[0].chunk_text,
    )

    with pytest.raises(
        ValueError,
        match="references an unknown document",
    ):
        validate_chunks(
            [invalid_chunk],
            [document],
        )


def test_validate_chunks_rejects_invalid_page_number():
    """
    A chunk referencing a non-existent source page should
    be rejected.
    """

    document = create_test_document()

    chunks = chunk_document(document)

    invalid_chunk = DocumentChunk(
        chunk_id=chunks[0].chunk_id,
        document_id=chunks[0].document_id,
        source_file=chunks[0].source_file,
        document_type=chunks[0].document_type,
        status=chunks[0].status,
        applies_to=chunks[0].applies_to,
        account_id=chunks[0].account_id,
        page_number=999,
        page_range="999",
        chunk_index=chunks[0].chunk_index,
        chunk_text=chunks[0].chunk_text,
    )

    with pytest.raises(
        ValueError,
        match="invalid page metadata",
    ):
        validate_chunks(
            [invalid_chunk],
            [document],
        )


def test_validate_chunks_rejects_incorrect_account_id():
    """
    A chunk with an account ID different from its source
    document should be rejected.
    """

    document = create_test_document(
        account_id="ACCT-001",
    )

    chunks = chunk_document(document)

    invalid_chunk = DocumentChunk(
        chunk_id=chunks[0].chunk_id,
        document_id=chunks[0].document_id,
        source_file=chunks[0].source_file,
        document_type=chunks[0].document_type,
        status=chunks[0].status,
        applies_to=chunks[0].applies_to,
        account_id="ACCT-999",
        page_number=chunks[0].page_number,
        page_range=chunks[0].page_range,
        chunk_index=chunks[0].chunk_index,
        chunk_text=chunks[0].chunk_text,
    )

    with pytest.raises(
        ValueError,
        match="incorrect account applicability",
    ):
        validate_chunks(
            [invalid_chunk],
            [document],
        )


def test_validate_chunks_rejects_missing_required_field():
    """
    Validation should reject a chunk with blank required text.
    """

    document = create_test_document()

    chunks = chunk_document(document)

    invalid_chunk = DocumentChunk(
        chunk_id=chunks[0].chunk_id,
        document_id=chunks[0].document_id,
        source_file=chunks[0].source_file,
        document_type=chunks[0].document_type,
        status=chunks[0].status,
        applies_to=chunks[0].applies_to,
        account_id=chunks[0].account_id,
        page_number=chunks[0].page_number,
        page_range=chunks[0].page_range,
        chunk_index=chunks[0].chunk_index,
        chunk_text="   ",
    )

    with pytest.raises(
        ValueError,
        match="missing chunk_text",
    ):
        validate_chunks(
            [invalid_chunk],
            [document],
        )


# ---------------------------------------------------------------------
# Chunk size tests
# ---------------------------------------------------------------------

def test_real_chunks_contain_text():
    """
    Every generated real chunk should contain meaningful text.
    """

    documents = process_all_documents()

    chunks = chunk_all_documents(documents)

    assert all(
        chunk.chunk_text.strip()
        for chunk in chunks
    )


def test_real_chunks_do_not_exceed_expected_size_by_large_amount():
    """
    Chunk body generation should generally stay within the
    configured maximum, allowing for the document title prefix.
    """

    documents = process_all_documents()

    chunks = chunk_all_documents(documents)

    for chunk in chunks:

        document = next(
            document
            for document in documents
            if document.document_id == chunk.document_id
        )

        title_length = len(
            document.metadata.document_title
        ) + 1

        body_length = (
            len(chunk.chunk_text)
            - title_length
        )

        assert body_length <= MAX_CHUNK_CHARS


def test_minimum_chunk_constant_is_positive():
    """
    MIN_CHUNK_CHARS should be a positive value smaller than
    MAX_CHUNK_CHARS.
    """

    assert MIN_CHUNK_CHARS > 0

    assert MIN_CHUNK_CHARS < MAX_CHUNK_CHARS