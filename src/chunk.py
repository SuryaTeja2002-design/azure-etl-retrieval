from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

from extract import ExtractedDocument

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class TextChunk:
    """A single retrieval unit produced by the chunker."""

    chunk_id: str           # "<doc_id>_<index>"  e.g. "deviceA_003"
    doc_id: str             # Parent document identifier
    text: str               # The actual chunk text
    token_count: int        # Approximate token count (word-based estimate)
    chunk_index: int        # 0-based position within the document
    page_hint: Optional[int] = None      # Best-guess source page (PDFs)
    heading: Optional[str] = None        # Nearest ancestor heading
    metadata: dict = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return len(self.text)


# ---------------------------------------------------------------------------
# Token counting (fast approximation – avoids heavy tokenizer dependency)
# ---------------------------------------------------------------------------

def _approx_tokens(text: str) -> int:
    """
    Rough token count: ~0.75 words per token for English text.
    Replace with tiktoken or HuggingFace tokenizer for precision.
    """
    words = len(text.split())
    return max(1, int(words / 0.75))


# ---------------------------------------------------------------------------
# Structural splitters
# ---------------------------------------------------------------------------

# Matches ATX headings (#, ##, ###) used in Markdown
_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

# Sentence boundary: period / ! / ? followed by whitespace and uppercase or end-of-string
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\u00C0-\u024F])")


def _split_by_markdown_headings(text: str) -> list[tuple[str | None, str]]:
    """
    Split Markdown into (heading, section_text) pairs.
    The first element may have heading=None for any preamble before the first heading.
    """
    sections: list[tuple[str | None, str]] = []
    last_end = 0
    last_heading: str | None = None

    for m in _MD_HEADING_RE.finditer(text):
        # Save everything between the last heading and this one
        body = text[last_end : m.start()].strip()
        if body:
            sections.append((last_heading, body))
        last_heading = m.group(2).strip()
        last_end = m.end()

    # Tail after last heading
    tail = text[last_end:].strip()
    if tail:
        sections.append((last_heading, tail))

    return sections


def _split_into_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


# ---------------------------------------------------------------------------
# Core chunker
# ---------------------------------------------------------------------------

class DocumentChunker:
    """
    Chunks an ExtractedDocument into TextChunk objects.

    Parameters
    ----------
    target_tokens : int
        Target maximum tokens per chunk (default 512).
    overlap_tokens : int
        Overlap between adjacent chunks (default 50).
    min_tokens : int
        Chunks smaller than this are merged with the previous one (default 50).
    """

    def __init__(
        self,
        target_tokens: int = 512,
        overlap_tokens: int = 50,
        min_tokens: int = 50,
    ) -> None:
        if overlap_tokens >= target_tokens:
            raise ValueError("overlap_tokens must be less than target_tokens")
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.min_tokens = min_tokens

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def chunk_document(self, doc: ExtractedDocument) -> list[TextChunk]:
        """Dispatch to the appropriate strategy based on file type."""
        if doc.file_type == "md":
            raw_chunks = self._chunk_markdown(doc.text)
        elif doc.file_type == "pdf":
            raw_chunks = self._chunk_pdf(doc.text, doc.page_count)
        else:
            raw_chunks = self._chunk_plain(doc.text)

        # Finalise: assign IDs and attach parent metadata
        chunks: list[TextChunk] = []
        for i, (text, heading, page_hint) in enumerate(raw_chunks):
            t_count = _approx_tokens(text)
            chunks.append(
                TextChunk(
                    chunk_id=f"{doc.doc_id}_{i:04d}",
                    doc_id=doc.doc_id,
                    text=text,
                    token_count=t_count,
                    chunk_index=i,
                    page_hint=page_hint,
                    heading=heading,
                    metadata={
                        **doc.metadata,
                        "blob_name": doc.blob_name,
                        "source_url": doc.source_url,
                        "file_type": doc.file_type,
                        "category": doc.category,
                        "chunk_index": i,
                        "heading": heading,
                        "page_hint": page_hint,
                        "token_count": t_count,
                    },
                )
            )

        logger.info(
            "Chunked '%s' into %d chunks (target=%d tokens)",
            doc.doc_id, len(chunks), self.target_tokens,
        )
        return chunks

    # ------------------------------------------------------------------
    # Markdown strategy
    # ------------------------------------------------------------------

    def _chunk_markdown(self, text: str) -> list[tuple[str, str | None, None]]:
        """
        Split by headings first, then slide a window within each section.
        Returns list of (chunk_text, heading, page_hint=None).
        """
        raw: list[tuple[str, str | None, None]] = []
        sections = _split_by_markdown_headings(text)

        for heading, body in sections:
            for chunk_text in self._sliding_window(body):
                raw.append((chunk_text, heading, None))

        return raw

    # ------------------------------------------------------------------
    # PDF strategy
    # ------------------------------------------------------------------

    def _chunk_pdf(self, text: str, page_count: int) -> list[tuple[str, None, int | None]]:
        """
        PDFs are chunked at page boundaries first (pdfplumber separates pages
        with double newlines), then slid within each page.
        """
        raw: list[tuple[str, None, int | None]] = []

        # Pages were joined with "\n\n" in extract.py
        pages = re.split(r"\n{2,}", text)

        for page_idx, page_text in enumerate(pages):
            page_num = page_idx + 1
            for chunk_text in self._sliding_window(page_text.strip()):
                raw.append((chunk_text, None, page_num))

        return raw

    # ------------------------------------------------------------------
    # Plain text strategy
    # ------------------------------------------------------------------

    def _chunk_plain(self, text: str) -> list[tuple[str, None, None]]:
        return [(ct, None, None) for ct in self._sliding_window(text)]

    # ------------------------------------------------------------------
    # Sliding window (sentence-aware)
    # ------------------------------------------------------------------

    def _sliding_window(self, text: str) -> list[str]:
        """
        Split text into overlapping windows of ~target_tokens tokens.
        Tries to break at sentence boundaries to avoid mid-sentence cuts.
        """
        if not text.strip():
            return []

        sentences = _split_into_sentences(text)
        if not sentences:
            sentences = [text]

        chunks: list[str] = []
        buffer: list[str] = []
        buffer_tokens = 0
        overlap_buffer: list[str] = []

        for sent in sentences:
            sent_tokens = _approx_tokens(sent)

            # If a single sentence exceeds the target, hard-split it
            if sent_tokens > self.target_tokens:
                # Flush current buffer first
                if buffer:
                    chunk_text = " ".join(buffer).strip()
                    if _approx_tokens(chunk_text) >= self.min_tokens:
                        chunks.append(chunk_text)
                    buffer, buffer_tokens = [], 0

                # Hard-split the long sentence by word count
                words = sent.split()
                step = int(self.target_tokens * 0.75)  # convert back to words
                for start in range(0, len(words), step - int(self.overlap_tokens * 0.75)):
                    segment = " ".join(words[start : start + step])
                    if segment.strip():
                        chunks.append(segment.strip())
                continue

            # Would adding this sentence overflow the window?
            if buffer_tokens + sent_tokens > self.target_tokens and buffer:
                chunk_text = " ".join(buffer).strip()
                if _approx_tokens(chunk_text) >= self.min_tokens:
                    chunks.append(chunk_text)

                # Build overlap: keep tail sentences up to overlap_tokens
                overlap_buffer = []
                overlap_count = 0
                for s in reversed(buffer):
                    st = _approx_tokens(s)
                    if overlap_count + st > self.overlap_tokens:
                        break
                    overlap_buffer.insert(0, s)
                    overlap_count += st

                buffer = overlap_buffer[:]
                buffer_tokens = sum(_approx_tokens(s) for s in buffer)

            buffer.append(sent)
            buffer_tokens += sent_tokens

        # Flush the last buffer
        if buffer:
            chunk_text = " ".join(buffer).strip()
            if _approx_tokens(chunk_text) >= self.min_tokens:
                chunks.append(chunk_text)
            elif chunks:
                # Merge tiny tail into the previous chunk
                chunks[-1] = chunks[-1] + " " + chunk_text

        return chunks


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def chunk_documents(
    documents: list[ExtractedDocument],
    target_tokens: int = 512,
    overlap_tokens: int = 50,
) -> list[TextChunk]:
    """Chunk a list of extracted documents and return all chunks."""
    chunker = DocumentChunker(target_tokens=target_tokens, overlap_tokens=overlap_tokens)
    all_chunks: list[TextChunk] = []
    for doc in documents:
        all_chunks.extend(chunker.chunk_document(doc))
    logger.info("Total chunks produced: %d", len(all_chunks))
    return all_chunks