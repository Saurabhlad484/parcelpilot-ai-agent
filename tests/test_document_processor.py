from pathlib import Path

import pytest

from src.document_processor import (
    DOCUMENT_CATALOG,
    DocumentMetadata,
    ExtractedPage,
    ProcessedDocument,
    build_validation_report,
    discover_documents,
    extract_document,
    process_all_documents,
    validate_processed_documents,
)


# ---------------------------------------------------------------------
# Document discovery tests
# ---------------------------------------------------------------------

def test_discover_documents_returns_all_expected_documents():
    """
    All catalogued PDF documents should be discovered.
    """

    documents = discover_documents()

    assert len(documents) == len(DOCUMENT_CATALOG)

    discovered_names = [
        document.name
        for document in documents
    ]

    assert discovered_names == list(DOCUMENT_CATALOG.keys())


def test_discover_documents_missing_directory_raises_error(
    tmp_path,
):
    """
    A missing documents directory should raise FileNotFoundError.
    """

    missing_directory = tmp_path / "missing_documents"

    with pytest.raises(
        FileNotFoundError,
        match="Documents directory not found",
    ):
        discover_documents(missing_directory)


def test_discover_documents_missing_expected_pdf_raises_error(
    tmp_path,
):
    """
    If an expected PDF is missing, discovery should fail.
    """

    # Create the directory but do not add the required PDFs.
    documents_directory = tmp_path / "documents"
    documents_directory.mkdir()

    with pytest.raises(
        FileNotFoundError,
        match="Missing expected PDF document",
    ):
        discover_documents(documents_directory)


def test_discover_documents_unexpected_pdf_raises_error(
    tmp_path,
):
    """
    A PDF without catalog metadata should be rejected.
    """

    documents_directory = tmp_path / "documents"
    documents_directory.mkdir()

    # Create all expected PDF filenames.
    for filename in DOCUMENT_CATALOG:
        (documents_directory / filename).touch()

    # Add one unexpected PDF.
    (
        documents_directory / "unexpected_document.pdf"
    ).touch()

    with pytest.raises(
        ValueError,
        match="missing catalog metadata",
    ):
        discover_documents(documents_directory)


# ---------------------------------------------------------------------
# Single document extraction tests
# ---------------------------------------------------------------------

def test_extract_document_returns_processed_document():
    """
    A valid catalogued PDF should be extracted into a
    ProcessedDocument.
    """

    documents = discover_documents()

    document = extract_document(documents[0])

    assert isinstance(document, ProcessedDocument)

    assert document.document_id == (
        DOCUMENT_CATALOG[
            documents[0].name
        ].document_id
    )

    assert document.metadata.source_file == documents[0].name


def test_extract_document_pages_have_valid_page_numbers():
    """
    Extracted pages should be numbered starting from 1
    and continue sequentially.
    """

    documents = discover_documents()

    document = extract_document(documents[0])

    page_numbers = [
        page.page_number
        for page in document.pages
    ]

    assert page_numbers == list(
        range(
            1,
            len(document.pages) + 1,
        )
    )


def test_extract_document_missing_file_raises_error(
    tmp_path,
):
    """
    Extracting a file that does not exist should raise
    FileNotFoundError.
    """

    missing_file = tmp_path / "missing.pdf"

    with pytest.raises(
        FileNotFoundError,
        match="PDF document not found",
    ):
        extract_document(missing_file)


def test_extract_document_without_catalog_entry_raises_error(
    tmp_path,
):
    """
    A PDF that is not present in the metadata catalog
    should raise ValueError.
    """

    unknown_pdf = tmp_path / "unknown.pdf"
    unknown_pdf.touch()

    with pytest.raises(
        ValueError,
        match="No metadata catalog entry",
    ):
        extract_document(unknown_pdf)


# ---------------------------------------------------------------------
# Process all documents tests
# ---------------------------------------------------------------------

def test_process_all_documents_returns_all_documents():
    """
    All expected ParcelPilot PDFs should be processed.
    """

    documents = process_all_documents()

    assert len(documents) == len(DOCUMENT_CATALOG)


def test_process_all_documents_returns_processed_document_objects():
    """
    Every processed item should be a ProcessedDocument.
    """

    documents = process_all_documents()

    assert all(
        isinstance(
            document,
            ProcessedDocument,
        )
        for document in documents
    )


def test_processed_documents_match_catalog_metadata():
    """
    Processed documents should use the metadata defined
    in DOCUMENT_CATALOG.
    """

    documents = process_all_documents()

    for document in documents:

        expected_metadata = DOCUMENT_CATALOG[
            document.metadata.source_file
        ]

        assert document.document_id == (
            expected_metadata.document_id
        )

        assert document.metadata.document_type == (
            expected_metadata.document_type
        )

        assert document.metadata.status == (
            expected_metadata.status
        )

        assert document.metadata.authority_level == (
            expected_metadata.authority_level
        )


# ---------------------------------------------------------------------
# Authority and account metadata tests
# ---------------------------------------------------------------------

def test_current_support_policy_is_allowed_for_current_requests():
    """
    Support Policy v3 should be available for current requests.
    """

    metadata = DOCUMENT_CATALOG[
        "01_Support_Policy_v3_CURRENT.pdf"
    ]

    assert metadata.status == "current"

    assert metadata.allowed_for_current_requests is True


def test_deprecated_support_policy_is_not_allowed_for_current_requests():
    """
    Support Policy v2 is historical and must not be used
    for current requests.
    """

    metadata = DOCUMENT_CATALOG[
        "02_Support_Policy_v2_DEPRECATED.pdf"
    ]

    assert metadata.status == "deprecated"

    assert metadata.allowed_for_current_requests is False


def test_northstar_agreement_is_linked_to_correct_account():
    """
    Northstar's agreement should belong only to ACCT-001.
    """

    metadata = DOCUMENT_CATALOG[
        "05_Northstar_Logistics_Enterprise_Agreement.pdf"
    ]

    assert metadata.account_id == "ACCT-001"

    assert metadata.document_type == "customer_agreement"

    assert metadata.allowed_for_current_requests is True


def test_lumenworks_agreement_is_linked_to_correct_account():
    """
    LumenWorks' agreement should belong only to ACCT-002.
    """

    metadata = DOCUMENT_CATALOG[
        "06_LumenWorks_Service_Agreement.pdf"
    ]

    assert metadata.account_id == "ACCT-002"

    assert metadata.document_type == "customer_agreement"

    assert metadata.allowed_for_current_requests is True


# ---------------------------------------------------------------------
# ProcessedDocument serialization tests
# ---------------------------------------------------------------------

def test_processed_document_to_dict_returns_expected_structure():
    """
    ProcessedDocument.to_dict should return a serializable
    structure containing metadata and pages.
    """

    metadata = DocumentMetadata(
        source_file="test.pdf",
        document_id="test-document",
        document_title="Test Document",
        document_type="test_type",
        status="current",
        authority_level="test_authority",
        effective_or_updated_date="2026-01-01",
        account_id=None,
        account_name=None,
        applicability="test",
        allowed_for_current_requests=True,
        notes="Test notes.",
    )

    pages = [
        ExtractedPage(
            page_number=1,
            text="Test page text.",
        )
    ]

    document = ProcessedDocument(
        document_id="test-document",
        metadata=metadata,
        pages=pages,
    )

    result = document.to_dict()

    assert result["document_id"] == "test-document"

    assert result["metadata"]["source_file"] == "test.pdf"

    assert result["metadata"][
        "allowed_for_current_requests"
    ] is True

    assert len(result["pages"]) == 1

    assert result["pages"][0]["page_number"] == 1

    assert result["pages"][0]["text"] == "Test page text."


# ---------------------------------------------------------------------
# Validation report tests
# ---------------------------------------------------------------------

def test_build_validation_report_returns_one_item_per_document():
    """
    The validation report should contain one summary item
    for every processed document.
    """

    documents = process_all_documents()

    report = build_validation_report(documents)

    assert len(report) == len(documents)


def test_build_validation_report_contains_expected_fields():
    """
    Each validation report item should contain the required
    extraction and metadata information.
    """

    documents = process_all_documents()

    report = build_validation_report(documents)

    expected_fields = {
        "filename",
        "document_id",
        "status",
        "document_type",
        "applicability",
        "pages",
        "extracted_characters",
        "blank_pages",
    }

    for item in report:
        assert expected_fields.issubset(item.keys())


def test_validation_report_page_count_matches_document():
    """
    The report page count should match the actual extracted
    page count.
    """

    documents = process_all_documents()

    report = build_validation_report(documents)

    for document, report_item in zip(
        documents,
        report,
    ):
        assert report_item["pages"] == len(
            document.pages
        )


# ---------------------------------------------------------------------
# Processed document validation tests
# ---------------------------------------------------------------------

def test_validate_processed_documents_passes_for_real_documents():
    """
    The complete set of real project documents should pass
    validation.
    """

    documents = process_all_documents()

    validate_processed_documents(documents)


def test_validate_processed_documents_wrong_count_raises_error():
    """
    Validation should fail when the number of documents
    does not match the expected catalog.
    """

    documents = process_all_documents()

    incomplete_documents = documents[:-1]

    with pytest.raises(
        ValueError,
        match="Expected",
    ):
        validate_processed_documents(
            incomplete_documents
        )


def test_validate_processed_documents_rejects_invalid_northstar_account():
    """
    Validation should reject a Northstar agreement linked
    to the wrong account.
    """

    documents = process_all_documents()

    modified_documents = []

    for document in documents:

        if (
            document.metadata.source_file
            == "05_Northstar_Logistics_Enterprise_Agreement.pdf"
        ):

            invalid_metadata = DocumentMetadata(
                source_file=document.metadata.source_file,
                document_id=document.metadata.document_id,
                document_title=document.metadata.document_title,
                document_type=document.metadata.document_type,
                status=document.metadata.status,
                authority_level=document.metadata.authority_level,
                effective_or_updated_date=(
                    document.metadata.effective_or_updated_date
                ),
                account_id="ACCT-999",
                account_name=document.metadata.account_name,
                applicability=document.metadata.applicability,
                allowed_for_current_requests=(
                    document.metadata.allowed_for_current_requests
                ),
                notes=document.metadata.notes,
            )

            modified_documents.append(
                ProcessedDocument(
                    document_id=document.document_id,
                    metadata=invalid_metadata,
                    pages=document.pages,
                )
            )

        else:
            modified_documents.append(document)

    with pytest.raises(
        ValueError,
        match="Northstar agreement must be linked",
    ):
        validate_processed_documents(
            modified_documents
        )


def test_validate_processed_documents_rejects_invalid_lumenworks_account():
    """
    Validation should reject a LumenWorks agreement linked
    to the wrong account.
    """

    documents = process_all_documents()

    modified_documents = []

    for document in documents:

        if (
            document.metadata.source_file
            == "06_LumenWorks_Service_Agreement.pdf"
        ):

            invalid_metadata = DocumentMetadata(
                source_file=document.metadata.source_file,
                document_id=document.metadata.document_id,
                document_title=document.metadata.document_title,
                document_type=document.metadata.document_type,
                status=document.metadata.status,
                authority_level=document.metadata.authority_level,
                effective_or_updated_date=(
                    document.metadata.effective_or_updated_date
                ),
                account_id="ACCT-999",
                account_name=document.metadata.account_name,
                applicability=document.metadata.applicability,
                allowed_for_current_requests=(
                    document.metadata.allowed_for_current_requests
                ),
                notes=document.metadata.notes,
            )

            modified_documents.append(
                ProcessedDocument(
                    document_id=document.document_id,
                    metadata=invalid_metadata,
                    pages=document.pages,
                )
            )

        else:
            modified_documents.append(document)

    with pytest.raises(
        ValueError,
        match="LumenWorks agreement must be linked",
    ):
        validate_processed_documents(
            modified_documents
        )


def test_validate_processed_documents_rejects_invalid_page_numbers():
    """
    Validation should reject documents whose page numbering
    is not sequential starting from 1.
    """

    documents = process_all_documents()

    modified_documents = []

    for index, document in enumerate(documents):

        if index == 0 and document.pages:

            invalid_pages = [
                ExtractedPage(
                    page_number=2,
                    text=document.pages[0].text,
                )
            ] + document.pages[1:]

            modified_documents.append(
                ProcessedDocument(
                    document_id=document.document_id,
                    metadata=document.metadata,
                    pages=invalid_pages,
                )
            )

        else:
            modified_documents.append(document)

    with pytest.raises(
        ValueError,
        match="Page numbering is invalid",
    ):
        validate_processed_documents(
            modified_documents
        )