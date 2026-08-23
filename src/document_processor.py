"""Extract ParcelPilot PDF documents into traceable, structured records.

This module deliberately stops at extraction and metadata.  Chunking, vector
storage, retrieval, and LLM integration belong to later implementation steps.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from src.data_loader import DOCUMENTS_DIR


@dataclass(frozen=True)
class DocumentMetadata:
    """Explicit, retrieval-ready metadata for one supplied source document."""

    source_file: str
    document_id: str
    document_title: str
    document_type: str
    status: str
    authority_level: str
    effective_or_updated_date: str | None
    account_id: str | None
    account_name: str | None
    applicability: str
    allowed_for_current_requests: bool
    notes: str


@dataclass(frozen=True)
class ExtractedPage:
    """Text extracted from one one-indexed PDF page."""

    page_number: int
    text: str


@dataclass(frozen=True)
class ProcessedDocument:
    """A source document with metadata and page-level extracted text."""

    document_id: str
    metadata: DocumentMetadata
    pages: list[ExtractedPage]

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable representation for future processing layers."""
        return {
            "document_id": self.document_id,
            "metadata": asdict(self.metadata),
            "pages": [asdict(page) for page in self.pages],
        }


# This catalog is intentionally explicit rather than inferred from filenames at
# retrieval time. Dates, status, scope, and authority are transcribed from the
# supplied documents and Phase 1 inspection findings.
DOCUMENT_CATALOG: dict[str, DocumentMetadata] = {
    "01_Support_Policy_v3_CURRENT.pdf": DocumentMetadata(
        source_file="01_Support_Policy_v3_CURRENT.pdf",
        document_id="support-policy-v3",
        document_title="ParcelPilot Support Policy v3",
        document_type="support_policy",
        status="current",
        authority_level="authoritative_current",
        effective_or_updated_date="2026-05-01",
        account_id=None,
        account_name=None,
        applicability="all_accounts_subject_to_agreement_override",
        allowed_for_current_requests=True,
        notes=(
            "Defines default severity and first-response targets. Signed customer "
            "agreements override defaults; historical tickets and internal notes are "
            "context only."
        ),
    ),
    "02_Support_Policy_v2_DEPRECATED.pdf": DocumentMetadata(
        source_file="02_Support_Policy_v2_DEPRECATED.pdf",
        document_id="support-policy-v2",
        document_title="ParcelPilot Support Policy v2",
        document_type="support_policy",
        status="deprecated",
        authority_level="historical_only",
        effective_or_updated_date="2025-01-01",
        account_id=None,
        account_name=None,
        applicability="historical_reference_only",
        allowed_for_current_requests=False,
        notes="Superseded by Support Policy v3 on 2026-05-01; do not use for current requests.",
    ),
    "03_Cancellation_and_Service_Credit_SOP_v4.pdf": DocumentMetadata(
        source_file="03_Cancellation_and_Service_Credit_SOP_v4.pdf",
        document_id="cancellation-service-credit-sop-v4",
        document_title="ParcelPilot Cancellation & Service Credit SOP v4",
        document_type="cancellation_service_credit_sop",
        status="current",
        authority_level="authoritative_current",
        effective_or_updated_date="2026-06-15",
        account_id=None,
        account_name=None,
        applicability="all_accounts_subject_to_agreement_override",
        allowed_for_current_requests=True,
        notes="Current controlling SOP for cancellation and service-credit defaults.",
    ),
    "04_Product_Operations_Guide_and_Known_Issues.pdf": DocumentMetadata(
        source_file="04_Product_Operations_Guide_and_Known_Issues.pdf",
        document_id="product-operations-guide-known-issues",
        document_title="ParcelPilot Product Operations Guide and Known Issues",
        document_type="product_operations_guide",
        status="current",
        authority_level="authoritative_current",
        effective_or_updated_date="2026-08-14",
        account_id=None,
        account_name=None,
        applicability="all_accounts_by_plan",
        allowed_for_current_requests=True,
        notes="Current product capability, operational guidance, and known-issue source.",
    ),
    "05_Northstar_Logistics_Enterprise_Agreement.pdf": DocumentMetadata(
        source_file="05_Northstar_Logistics_Enterprise_Agreement.pdf",
        document_id="northstar-logistics-enterprise-agreement",
        document_title="ParcelPilot - Northstar Logistics Enterprise Agreement",
        document_type="customer_agreement",
        status="active",
        authority_level="account_specific",
        effective_or_updated_date="2026-01-01 to 2026-12-31",
        account_id="ACCT-001",
        account_name="Northstar Logistics",
        applicability="ACCT-001 only",
        allowed_for_current_requests=True,
        notes="Active agreement with custom support SLA, cancellation, and credit-cap terms.",
    ),
    "06_LumenWorks_Service_Agreement.pdf": DocumentMetadata(
        source_file="06_LumenWorks_Service_Agreement.pdf",
        document_id="lumenworks-service-agreement",
        document_title="ParcelPilot - LumenWorks Service Agreement",
        document_type="customer_agreement",
        status="active",
        authority_level="account_specific",
        effective_or_updated_date="2026-03-01 to 2027-02-28",
        account_id="ACCT-002",
        account_name="LumenWorks",
        applicability="ACCT-002 only",
        allowed_for_current_requests=True,
        notes="Active agreement with support coverage and failed-pickup credit terms.",
    ),
}


def discover_documents(documents_dir: Path = DOCUMENTS_DIR) -> list[Path]:
    """Return all expected PDFs in catalog order and reject catalog mismatches."""
    if not documents_dir.exists():
        raise FileNotFoundError(f"Documents directory not found: {documents_dir}")

    discovered_names = {path.name for path in documents_dir.glob("*.pdf")}
    expected_names = set(DOCUMENT_CATALOG)
    missing = sorted(expected_names - discovered_names)
    unexpected = sorted(discovered_names - expected_names)

    if missing:
        raise FileNotFoundError(f"Missing expected PDF document(s): {missing}")
    if unexpected:
        raise ValueError(f"PDF document(s) missing catalog metadata: {unexpected}")

    return [documents_dir / filename for filename in DOCUMENT_CATALOG]


def extract_document(pdf_path: Path) -> ProcessedDocument:
    """Extract every page of one catalogued PDF, retaining its one-indexed page."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF document not found: {pdf_path}")

    try:
        metadata = DOCUMENT_CATALOG[pdf_path.name]
    except KeyError as error:
        raise ValueError(f"No metadata catalog entry for PDF: {pdf_path.name}") from error

    try:
        reader = PdfReader(pdf_path)
        pages = [
            ExtractedPage(page_number=index, text=(page.extract_text() or "").strip())
            for index, page in enumerate(reader.pages, start=1)
        ]
    except Exception as error:
        raise RuntimeError(f"Failed to extract PDF '{pdf_path.name}': {error}") from error

    return ProcessedDocument(
        document_id=metadata.document_id,
        metadata=metadata,
        pages=pages,
    )


def process_all_documents(documents_dir: Path = DOCUMENTS_DIR) -> list[ProcessedDocument]:
    """Discover and extract every expected ParcelPilot PDF; extraction errors propagate."""
    return [extract_document(path) for path in discover_documents(documents_dir)]


def build_validation_report(documents: list[ProcessedDocument]) -> list[dict[str, Any]]:
    """Create concise, inspectable extraction metrics for each processed document."""
    return [
        {
            "filename": document.metadata.source_file,
            "document_id": document.document_id,
            "status": document.metadata.status,
            "document_type": document.metadata.document_type,
            "applicability": document.metadata.applicability,
            "pages": len(document.pages),
            "extracted_characters": sum(len(page.text) for page in document.pages),
            "blank_pages": sum(not page.text.strip() for page in document.pages),
        }
        for document in documents
    ]


def validate_processed_documents(documents: list[ProcessedDocument]) -> None:
    """Enforce the catalog invariants required before later retrieval work."""
    if len(documents) != len(DOCUMENT_CATALOG):
        raise ValueError(f"Expected {len(DOCUMENT_CATALOG)} documents, processed {len(documents)}.")

    by_file = {document.metadata.source_file: document for document in documents}
    if not by_file["01_Support_Policy_v3_CURRENT.pdf"].metadata.allowed_for_current_requests:
        raise ValueError("Current support policy must be allowed for current requests.")
    if by_file["02_Support_Policy_v2_DEPRECATED.pdf"].metadata.allowed_for_current_requests:
        raise ValueError("Deprecated support policy must not be allowed for current requests.")
    if by_file["05_Northstar_Logistics_Enterprise_Agreement.pdf"].metadata.account_id != "ACCT-001":
        raise ValueError("Northstar agreement must be linked to ACCT-001.")
    if by_file["06_LumenWorks_Service_Agreement.pdf"].metadata.account_id != "ACCT-002":
        raise ValueError("LumenWorks agreement must be linked to ACCT-002.")

    for document in documents:
        expected_page_numbers = list(range(1, len(document.pages) + 1))
        actual_page_numbers = [page.page_number for page in document.pages]
        if actual_page_numbers != expected_page_numbers:
            raise ValueError(f"Page numbering is invalid for {document.metadata.source_file}.")


def main() -> None:
    """Run the Step 1 extraction validation from the command line."""
    documents = process_all_documents()
    validate_processed_documents(documents)
    report = build_validation_report(documents)

    print(f"Documents processed: {len(documents)}")
    for item in report:
        print(
            " | ".join(
                [
                    item["filename"],
                    f"id={item['document_id']}",
                    f"status={item['status']}",
                    f"type={item['document_type']}",
                    f"applies={item['applicability']}",
                    f"pages={item['pages']}",
                    f"chars={item['extracted_characters']}",
                    f"blank_pages={item['blank_pages']}",
                ]
            )
        )
    print("Validation: PASS")


if __name__ == "__main__":
    main()
