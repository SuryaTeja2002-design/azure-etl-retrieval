

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Optional

import pdfplumber
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ExtractedDocument:
    """Canonical representation of a processed document."""

    blob_name: str          # Full path in the container, e.g. /manuals/deviceA.pdf
    source_url: str         # Public / SAS URL (if available)
    file_type: str          # "pdf" | "md" | "txt"
    text: str               # Full extracted text (UTF-8)
    page_count: int = 0     # PDFs only; 0 for others
    metadata: dict = field(default_factory=dict)
    extracted_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # Convenience helpers -------------------------------------------------------
    @property
    def doc_id(self) -> str:
        """Stable identifier derived from the blob path."""
        return PurePosixPath(self.blob_name).stem

    @property
    def category(self) -> str:
        """Top-level folder acts as a rough category label."""
        parts = PurePosixPath(self.blob_name).parts
        return parts[1] if len(parts) > 2 else "uncategorised"


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _extract_pdf(data: bytes, blob_name: str, ocr_client: Optional[DocumentAnalysisClient]) -> tuple[str, int]:
    """
    Try digital extraction first (pdfplumber).  If the page text is sparse
    (likely scanned), fall back to Azure Document Intelligence OCR.
    """
    text_pages: list[str] = []
    page_count = 0

    with pdfplumber.open(io.BytesIO(data)) as pdf:
        page_count = len(pdf.pages)
        for page in pdf.pages:
            raw = page.extract_text() or ""
            text_pages.append(raw)

    combined = "\n\n".join(text_pages).strip()

    # Heuristic: < 20 chars per page on average → probably scanned
    avg_chars = len(combined) / max(page_count, 1)
    if avg_chars < 20 and ocr_client is not None:
        logger.info("Sparse text in %s (avg %.1f chars/page) – using OCR fallback", blob_name, avg_chars)
        combined = _ocr_pdf(data, ocr_client)
    elif avg_chars < 20:
        logger.warning(
            "Sparse text in %s but no OCR client configured – result may be incomplete",
            blob_name,
        )

    return combined, page_count


def _ocr_pdf(data: bytes, client: DocumentAnalysisClient) -> str:
    """Use Azure Document Intelligence (Read model) to OCR a PDF blob."""
    poller = client.begin_analyze_document("prebuilt-read", document=io.BytesIO(data))
    result = poller.result()

    lines: list[str] = []
    for page in result.pages:
        for line in page.lines:
            lines.append(line.content)
        lines.append("")          # blank line between pages

    return "\n".join(lines).strip()


def _extract_markdown(data: bytes) -> tuple[str, dict]:
    """
    Parse Markdown with optional YAML/TOML front-matter.
    Returns (body_text, front_matter_dict).
    """
    raw = data.decode("utf-8", errors="replace")
    front_matter: dict = {}

    # Detect YAML front-matter (--- ... ---)
    fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", raw, re.DOTALL)
    if fm_match:
        try:
            import yaml  # optional dependency
            front_matter = yaml.safe_load(fm_match.group(1)) or {}
        except Exception:
            pass
        raw = raw[fm_match.end():]

    return raw.strip(), front_matter


def _extract_txt(data: bytes) -> str:
    return data.decode("utf-8", errors="replace").strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class DocumentExtractor:
    """
    Pulls documents from an Azure Blob Storage container and extracts
    text + metadata ready for downstream chunking.

    Parameters
    ----------
    blob_connection_string : str
        Connection string for the Azure Storage account.
    container_name : str
        Name of the blob container.
    doc_intelligence_endpoint : str | None
        Azure Document Intelligence endpoint URL (for OCR fallback).
    doc_intelligence_key : str | None
        Corresponding API key.
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt"}

    def __init__(
        self,
        blob_connection_string: str,
        container_name: str,
        doc_intelligence_endpoint: Optional[str] = None,
        doc_intelligence_key: Optional[str] = None,
    ) -> None:
        self._blob_service = BlobServiceClient.from_connection_string(blob_connection_string)
        self._container = container_name
        self._ocr_client: Optional[DocumentAnalysisClient] = None

        if doc_intelligence_endpoint and doc_intelligence_key:
            self._ocr_client = DocumentAnalysisClient(
                endpoint=doc_intelligence_endpoint,
                credential=AzureKeyCredential(doc_intelligence_key),
            )

    # ------------------------------------------------------------------

    def list_blobs(self, prefix: str = "") -> list[str]:
        """Return all supported blob paths under the given prefix."""
        container_client = self._blob_service.get_container_client(self._container)
        blobs = [
            b.name
            for b in container_client.list_blobs(name_starts_with=prefix)
            if PurePosixPath(b.name).suffix.lower() in self.SUPPORTED_EXTENSIONS
        ]
        logger.info("Found %d supported blobs under prefix '%s'", len(blobs), prefix)
        return blobs

    def extract_blob(self, blob_name: str) -> ExtractedDocument:
        """Download and extract a single blob."""
        container_client = self._blob_service.get_container_client(self._container)
        blob_client = container_client.get_blob_client(blob_name)

        logger.info("Downloading %s …", blob_name)
        data: bytes = blob_client.download_blob().readall()

        # Fetch blob properties for extra metadata
        props = blob_client.get_blob_properties()
        base_metadata = {
            "blob_name": blob_name,
            "content_type": props.content_settings.content_type or "",
            "last_modified": props.last_modified.isoformat() if props.last_modified else "",
            "size_bytes": props.size,
            **(props.metadata or {}),
        }

        suffix = PurePosixPath(blob_name).suffix.lower()

        if suffix == ".pdf":
            text, page_count = _extract_pdf(data, blob_name, self._ocr_client)
            file_type = "pdf"
            extra_meta: dict = {"page_count": page_count}

        elif suffix == ".md":
            text, front_matter = _extract_markdown(data)
            file_type = "md"
            extra_meta = front_matter
            page_count = 0

        else:  # .txt
            text = _extract_txt(data)
            file_type = "txt"
            extra_meta = {}
            page_count = 0

        return ExtractedDocument(
            blob_name=blob_name,
            source_url=blob_client.url,
            file_type=file_type,
            text=text,
            page_count=page_count,
            metadata={**base_metadata, **extra_meta},
        )

    def extract_all(self, prefix: str = "") -> list[ExtractedDocument]:
        """Extract every supported document in the container."""
        blobs = self.list_blobs(prefix)
        results: list[ExtractedDocument] = []
        for name in blobs:
            try:
                doc = self.extract_blob(name)
                results.append(doc)
                logger.info("Extracted %s  (%d chars)", name, len(doc.text))
            except Exception as exc:
                logger.error("Failed to extract %s: %s", name, exc, exc_info=True)
        return results